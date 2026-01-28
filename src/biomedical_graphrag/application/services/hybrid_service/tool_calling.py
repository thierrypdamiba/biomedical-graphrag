import json
import asyncio
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from biomedical_graphrag.application.services.hybrid_service.neo4j_query import Neo4jGraphQuery
from biomedical_graphrag.application.services.hybrid_service.qdrant_query import AsyncQdrantQuery

from biomedical_graphrag.application.services.hybrid_service.prompts.hybrid_prompts import (
    QDRANT_PROMPT,
    NEO4J_PROMPT,
    fusion_summary_prompt,
)
from biomedical_graphrag.application.services.hybrid_service.tools.enrichment_tools import (
    NEO4J_ENRICHMENT_TOOLS,
)
from biomedical_graphrag.application.services.hybrid_service.tools.qdrant_tools import (
    QDRANT_TOOLS,
)
from biomedical_graphrag.config import settings
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


openai_client = OpenAI(api_key=settings.openai.api_key.get_secret_value())


def get_neo4j_schema() -> str:
    """Retrieve the Neo4j schema dynamically."""
    neo4j = Neo4jGraphQuery()
    schema = neo4j.get_schema()
    logger.info(f"Retrieved Neo4j schema length: {len(schema)}")
    return schema

@dataclass
class ToolExecution:
    """Record of a tool execution."""
    name: str
    arguments: dict[str, Any] | None = None
    result_count: int | None = None
    results: Any = None


@dataclass
class QdrantSearchResult:
    """Result from Qdrant vector search."""
    results: list[dict]
    tool: ToolExecution


# --------------------------------------------------------------------
# Phase 1 — Qdrant tools selection + execution
# --------------------------------------------------------------------
async def run_qdrant_vector_search(question: str, limit: int = 5) -> QdrantSearchResult:
    """Run Qdrant vector search.
    Args:
        question: The user question.
        limit: Maximum number of results to return.
    Returns:
        QdrantSearchResult with results and tool execution info.
    """
    prompt = QDRANT_PROMPT.format(question=question)
    qdrant = AsyncQdrantQuery()
    tool_name = "unknown"
    tool_args: dict[str, Any] = {}

    try:
        response = await asyncio.to_thread(
            openai_client.responses.create,  # type: ignore[call-overload]
            model=settings.openai.model,
            tools=QDRANT_TOOLS,
            input=[{"role": "user", "content": prompt}],
            tool_choice="required",
        )

        results = []
        if response.output:
            for tool_call in response.output:
                if tool_call.type == "function_call":
                    tool_name = tool_call.name
                    args = (
                        json.loads(tool_call.arguments)
                        if isinstance(tool_call.arguments, str)
                        else tool_call.arguments
                    )
                    # Force the limit from user setting
                    args["top_k"] = limit
                    tool_args = args.copy()
                    func = getattr(qdrant, tool_name, None)
                    if func:
                        logger.info(f"Executing Qdrant tool: {tool_name} with args: {args}")
                        results = await func(**args)
                        logger.info(f"Qdrant results count: {len(results)}")
    finally:
        await qdrant.close()

    return QdrantSearchResult(
        results=results,
        tool=ToolExecution(name=tool_name, arguments=tool_args, result_count=len(results), results=results),
    )

@dataclass
class Neo4jEnrichmentResult:
    """Result from Neo4j graph enrichment."""
    results: dict[str, Any]
    tools: list[ToolExecution]


# --------------------------------------------------------------------
# Phase 2 — Neo4j enrichment tools selection + execution
# --------------------------------------------------------------------
def run_graph_enrichment(question: str, qdrant_results: list[dict]) -> Neo4jEnrichmentResult:
    """Run graph enrichment.

    Args:
        question: The user question.
        qdrant_results: Qdrant payloads retrieved by a selected Qdrant tool.

    Returns:
        Neo4jEnrichmentResult with results and tool execution info.
    """
    schema = get_neo4j_schema()
    neo4j = Neo4jGraphQuery()
    tools_executed: list[ToolExecution] = []

    try:
        prompt = NEO4J_PROMPT.format(
            schema=schema,
            question=question,
            qdrant_points_metadata=str(qdrant_results),
        )

        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=settings.openai.model,
            tools=NEO4J_ENRICHMENT_TOOLS,
            input=[{"role": "user", "content": prompt}],
            tool_choice="auto",
        )

        results: dict[str, Any] = {}
        if response.output:
            for tool_call in response.output:
                if tool_call.type == "function_call":
                    name = tool_call.name
                    args = (
                        json.loads(tool_call.arguments)
                        if isinstance(tool_call.arguments, str)
                        else tool_call.arguments
                    )
                    func = getattr(neo4j, name, None)
                    if func:
                        try:
                            logger.info(f"Executing Neo4j tool: {name} with args: {args}")
                            result = func(**args)
                            results[name] = result
                            count = len(result) if isinstance(result, list) else None
                        except Exception as e:
                            logger.error(f"Neo4j tool {name} failed: {e}")
                            results[name] = f"Error: {e}"
                            result = None
                            count = 0
                        tools_executed.append(ToolExecution(name=name, arguments=args, result_count=count, results=result))

        logger.info(f"Neo4j tools executed: {[t.name for t in tools_executed]}")
        return Neo4jEnrichmentResult(results=results, tools=tools_executed)
    finally:
        neo4j.close()


async def run_graph_enrichment_async(question: str, qdrant_results: list[dict]) -> Neo4jEnrichmentResult:
    """Async wrapper for run_graph_enrichment to avoid blocking the event loop."""
    return await asyncio.to_thread(run_graph_enrichment, question, qdrant_results)


# --------------------------------------------------------------------
# Phase 3 — Fusion summarization
# --------------------------------------------------------------------
def summarize_fused_results(
    question: str, qdrant_results: list[dict], neo4j_results: dict[str, Any], limit: int = 5
) -> str:
    """Fuse semantic and graph evidence into one final biomedical summary.

    Args:
        question: The user question.
        qdrant_results: Qdrant tool results.
        neo4j_results: The Neo4j results.
        limit: Number of papers retrieved (controls Key Findings count).

    Returns:
        The summarized results.
    """
    prompt = fusion_summary_prompt(question, qdrant_results, neo4j_results, limit=limit) #TBD: handle when errors in results
    resp = openai_client.responses.create(
        model=settings.openai.model,
        input=prompt,
        temperature=settings.openai.temperature,
        max_output_tokens=settings.openai.max_tokens,
    )
    return resp.output_text.strip()


async def summarize_fused_results_async(
    question: str, qdrant_results: list[dict], neo4j_results: dict[str, Any], limit: int = 5
) -> str:
    """Async wrapper for summarize_fused_results to avoid blocking the event loop."""
    return await asyncio.to_thread(
        summarize_fused_results, question, qdrant_results, neo4j_results, limit
    )

@dataclass
class GraphRAGResult:
    """Result container for GraphRAG search."""
    summary: str
    qdrant_results: list[dict]
    neo4j_results: dict[str, Any]
    trace: list[ToolExecution] = field(default_factory=list)


# --------------------------------------------------------------------
# Unified helper
# --------------------------------------------------------------------
async def run_tools_sequence_and_summarize(question: str, limit: int = 5) -> GraphRAGResult:
    """Run graph enrichment and summarize the results.

    Args:
        question: The user question.

    Returns:
        GraphRAGResult containing summary, results, and trace.
    """
    trace: list[ToolExecution] = []

    # Phase 1: Qdrant vector search
    qdrant_result = await run_qdrant_vector_search(question, limit=limit)
    trace.append(qdrant_result.tool)

    # Phase 2: Neo4j enrichment
    neo4j_result = await run_graph_enrichment_async(question, qdrant_result.results)
    trace.extend(neo4j_result.tools)

    # Phase 3: Summarization
    summary = await summarize_fused_results_async(
        question, qdrant_result.results, neo4j_result.results, limit=limit
    )
    trace.append(ToolExecution(name="summarize"))

    return GraphRAGResult(
        summary=summary,
        qdrant_results=qdrant_result.results,
        neo4j_results=neo4j_result.results,
        trace=trace,
    )
