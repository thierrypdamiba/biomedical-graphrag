"""
FastAPI server for Biomedical GraphRAG API.

Provides endpoints for:
- Health check
- Neo4j graph statistics
- Hybrid GraphRAG search (Qdrant + Neo4j)
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

# Lazy-loaded modules (for faster startup / health checks)
_services_loaded = False
_neo4j_query_class = None
_run_tools_sequence = None


def _load_services() -> None:
    """Lazily load heavy services on first request."""
    global _services_loaded, _neo4j_query_class, _run_tools_sequence

    if _services_loaded:
        return

    logger.info("Loading GraphRAG services...")
    from biomedical_graphrag.application.services.hybrid_service.neo4j_query import (
        Neo4jGraphQuery,
    )
    from biomedical_graphrag.application.services.hybrid_service.tool_calling import (
        run_tools_sequence_and_summarize,
    )

    _neo4j_query_class = Neo4jGraphQuery
    _run_tools_sequence = run_tools_sequence_and_summarize
    _services_loaded = True
    logger.info("GraphRAG services loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting Biomedical GraphRAG API server")
    # Preload services in background after startup
    asyncio.create_task(_preload_services())
    yield
    logger.info("Shutting down Biomedical GraphRAG API server")


async def _preload_services() -> None:
    """Preload services in background after health check passes."""
    await asyncio.sleep(2)  # Wait for health check to pass first
    await asyncio.to_thread(_load_services)


app = FastAPI(
    title="Biomedical GraphRAG API",
    description="Hybrid search API combining Qdrant vector search with Neo4j knowledge graph",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SearchRequest(BaseModel):
    """Search request body."""

    query: str = Field(..., description="The search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    mode: str = Field(default="graphrag", description="Search mode: graphrag, dense, sparse, hybrid")


class TraceStep(BaseModel):
    """A single step in the execution trace."""

    name: str
    result_count: int | None = None
    results: Any = None


class SearchResponse(BaseModel):
    """Search response body."""

    summary: str | None = None
    results: list[dict[str, Any]] = []
    trace: list[TraceStep] = []
    metadata: dict[str, Any] = {}


class Neo4jStatsResponse(BaseModel):
    """Neo4j statistics response."""

    nodeLabels: list[dict[str, Any]] = []
    relationshipTypes: list[dict[str, Any]] = []
    totalNodes: int = 0
    totalRelationships: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.get("/api/neo4j/stats", response_model=Neo4jStatsResponse)
async def get_neo4j_stats() -> Neo4jStatsResponse:
    """Get Neo4j graph statistics."""
    try:
        _load_services()
        neo4j = _neo4j_query_class()

        try:
            # Get node counts by label
            node_labels_result = neo4j.query("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {}) YIELD value
                RETURN label, value.count as count
                ORDER BY count DESC
            """)

            # Fallback if APOC not available
            if not node_labels_result:
                node_labels_result = neo4j.query("""
                    MATCH (n)
                    WITH labels(n) as nodeLabels
                    UNWIND nodeLabels as label
                    RETURN label, count(*) as count
                    ORDER BY count DESC
                """)

            # Get relationship counts by type
            rel_types_result = neo4j.query("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                ORDER BY count DESC
            """)

            # Get totals
            totals = neo4j.query("""
                MATCH (n) WITH count(n) as nodes
                MATCH ()-[r]->() WITH nodes, count(r) as rels
                RETURN nodes, rels
            """)

            total_nodes = totals[0]["nodes"] if totals else 0
            total_rels = totals[0]["rels"] if totals else 0

            return Neo4jStatsResponse(
                nodeLabels=[{"label": r["label"], "count": r["count"]} for r in node_labels_result],
                relationshipTypes=[{"type": r["type"], "count": r["count"]} for r in rel_types_result],
                totalNodes=total_nodes,
                totalRelationships=total_rels,
            )
        finally:
            neo4j.close()

    except Exception as e:
        logger.error(f"Error fetching Neo4j stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch Neo4j stats: {str(e)}")


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Perform hybrid GraphRAG search.

    Combines Qdrant vector search with Neo4j knowledge graph enrichment.
    """
    try:
        _load_services()

        # Run the async hybrid search (returns GraphRAGResult with trace)
        graphrag_result = await _run_tools_sequence(request.query)

        # Build trace from tool executions (just names)
        trace = [TraceStep(name=t.name, result_count=t.result_count, results=t.results) for t in graphrag_result.trace]

        # Format Qdrant results for frontend (limit to 5)
        formatted_results = []
        for idx, result in enumerate(graphrag_result.qdrant_results[:5]):
            formatted_results.append({
                "id": result.get("pmid", f"result-{idx}"),
                "title": result.get("title", "Untitled"),
                "abstract": result.get("abstract", ""),
                "authors": result.get("authors", []),
                "journal": result.get("journal", ""),
                "year": result.get("year", ""),
                "pmid": result.get("pmid", ""),
                "score": result.get("score", 0),
            })

        return SearchResponse(
            summary=graphrag_result.summary,
            results=formatted_results,
            trace=trace,
            metadata={"query": request.query},
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def main() -> None:
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "biomedical_graphrag.api.server:app",
        host="0.0.0.0",
        port=8765,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
