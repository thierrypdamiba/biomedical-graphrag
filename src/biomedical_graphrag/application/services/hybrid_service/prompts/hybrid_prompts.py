"""Prompt templates for hybrid GraphRAG querying."""

import json
from typing import Any

 #TO DO: improve prompts
QDRANT_PROMPT = """
You are a biomedical query router for PubMed papers retrieval.

Your ONLY task is to choose and call exactly one Qdrant tool.
This tool will help retrieve relevant academic papers based on the user's query.
Do NOT generate any text.

Rules:
- Use `retrieve_papers_hybrid` if the user is simply searching for papers.
- Use `recommend_papers_based_on_examples` if the user in its query provides examples ('like', 'should be about') and ESPECIALLY counter-examples
  ('excluding', 'but not about', 'not like', 'shouldn't be about') regarding what the paper should or should not be about.

Argument rules:
- For `retrieve_papers_hybrid`, build `query` by lightly rewriting the user input
  to improve hybrid (dense + BM25) search. Preserve meaning. Do not add new topics.
- For `recommend_papers_based_on_examples`:
  - `positive_examples` = topics the paper should match
  - `negative_examples` = topics the paper should avoid
  Use only information from the user.

Call the selected tool and return nothing else.

User Question:
{question}
"""

NEO4J_PROMPT = """
You are a biomedical reasoning assistant.
You CANNOT answer the question using Qdrant context alone.
Always call at least one Neo4j enrichment tool before producing text.

Steps:
1. Read the user question and Qdrant context.
2. Identify biomedical entities (PMIDs, authors, MeSH terms, institutions).
3. Decide which Neo4j enrichment tool(s) to call, using the schema below.
4. Provide tool arguments based on the context.
5. Call the tool(s); after all calls are complete, you may generate text.

Neo4j Graph Schema:
{schema}

User Question:
{question}

Retrieved Qdrant Context:
{qdrant_points_metadata}
"""

FUSION_SUMMARY_PROMPT = """
You are a biomedical research assistant combining two data sources:

- Qdrant (vector search results)
- Neo4j (structured graph results)

Your goal:
Synthesize both sources into one concise, factual answer for a biomedical researcher.

Instructions:
- Mention how Neo4j results confirm, extend, or contradict Qdrant context.
- Highlight novel relationships discovered through the graph.
- Avoid repetition; prefer clarity and precision.

User Question:
{question}

Retrieved Qdrant Context:
{qdrant_context}

Neo4j Enrichment Results (JSON):
{neo4j_results}
"""


def _format_qdrant_points(qdrant_points_metadata: dict[str, Any]) -> str: #TO DO: change to just concatenating paper titles & abstracts
    """Render Qdrant tool results into a compact, LLM-friendly text block."""
    try:
        return json.dumps(qdrant_points_metadata, indent=2, ensure_ascii=False)
    except TypeError:
        # If some objects aren't JSON serializable, fall back to a safe string.
        return str(qdrant_points_metadata)


def neo4j_prompt(schema: str, question: str, qdrant_results: dict[str, Any]) -> str:
    """Generate the neo4j_prompt prompt.

    Args:
        schema: The Neo4j schema.
        question: The user question.
        qdrant_results: The Qdrant tool results / points metadata.

    Returns:
        The hybrid prompt.
    """
    return NEO4J_PROMPT.format(
        schema=schema,
        question=question,
        qdrant_points_metadata=str(qdrant_results),
    )


def fusion_summary_prompt(
    question: str, qdrant_results: dict[str, Any], neo4j_results: dict[str, Any]
) -> str:
    """Generate the fusion summary prompt.

    Args:
        question: The user question.
        qdrant_results: The Qdrant tool results / points metadata.
        neo4j_results: The Neo4j results.

    Returns:
        The fusion summary prompt.
    """
    return FUSION_SUMMARY_PROMPT.format(
        question=question,
        qdrant_context=_format_qdrant_points(qdrant_results),
        neo4j_results=json.dumps(neo4j_results, indent=2),
    )
