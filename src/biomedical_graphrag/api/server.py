"""
FastAPI server for GraphRAG search API.

Uses lazy loading to ensure fast startup and health check response times.
This is critical for AWS App Runner deployments which have strict health check timeouts.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Use basic logging at startup - lazy load full logger later
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup: preload heavy modules in background
    async def _preload():
        await asyncio.sleep(1)  # Let health check pass first
        logger.info("Preloading modules...")
        try:
            from biomedical_graphrag.config import settings  # noqa: F401
            from biomedical_graphrag.application.services.hybrid_service.tool_calling import (
                run_tools_sequence_and_summarize,  # noqa: F401
            )
            logger.info("Modules preloaded successfully")
        except Exception as e:
            logger.warning(f"Module preload failed (non-fatal): {e}")

    asyncio.create_task(_preload())
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="Biomedical GraphRAG API",
    description="GraphRAG-powered search for biomedical papers",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    mode: str = "graphrag"


class SearchResponse(BaseModel):
    summary: str
    results: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    metadata: dict[str, Any]


@app.get("/health")
async def health_check():
    """Health check endpoint - returns immediately for fast startup."""
    return {"status": "healthy"}


@app.get("/api/neo4j/stats")
async def neo4j_stats():
    """Get Neo4j graph statistics."""
    # Lazy imports to avoid slow startup
    from neo4j import AsyncGraphDatabase
    from biomedical_graphrag.config import settings  # noqa: F811
    
    try:
        uri = settings.neo4j.uri
        user = settings.neo4j.username
        password = settings.neo4j.password.get_secret_value()
        database = settings.neo4j.database
        
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        
        async with driver.session(database=database) as session:
            # Get node counts by label
            node_labels = []
            labels_result = await session.run("CALL db.labels() YIELD label RETURN label")
            labels = [record["label"] async for record in labels_result]
            
            for label in labels:
                count_result = await session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                record = await count_result.single()
                count = record["count"] if record else 0
                node_labels.append({"label": label, "count": count})
            
            # Sort by count descending
            node_labels.sort(key=lambda x: x["count"], reverse=True)
            
            # Get relationship counts by type
            rel_types = []
            rels_result = await session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            rel_type_names = [record["relationshipType"] async for record in rels_result]
            
            for rel_type in rel_type_names:
                count_result = await session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                record = await count_result.single()
                count = record["count"] if record else 0
                rel_types.append({"type": rel_type, "count": count})
            
            # Sort by count descending
            rel_types.sort(key=lambda x: x["count"], reverse=True)
            
            # Get totals
            total_nodes_result = await session.run("MATCH (n) RETURN count(n) as count")
            total_nodes_record = await total_nodes_result.single()
            total_nodes = total_nodes_record["count"] if total_nodes_record else 0
            
            total_rels_result = await session.run("MATCH ()-[r]->() RETURN count(r) as count")
            total_rels_record = await total_rels_result.single()
            total_rels = total_rels_record["count"] if total_rels_record else 0
        
        await driver.close()
        
        return {
            "nodeLabels": node_labels,
            "relationshipTypes": rel_types,
            "totalNodes": total_nodes,
            "totalRelationships": total_rels,
        }
        
    except Exception as e:
        logger.error(f"Neo4j stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> dict[str, Any]:
    """
    Run GraphRAG search pipeline.

    Args:
        request: SearchRequest with query, limit, and mode

    Returns:
        SearchResponse with summary, results, trace, and metadata
    """
    try:
        logger.info(f"Received search request: {request.query}")

        # Lazy import to avoid slow startup (critical for App Runner health checks)
        from biomedical_graphrag.application.services.hybrid_service.tool_calling import (
            run_tools_sequence_and_summarize,
        )

        # Run the GraphRAG pipeline - returns a summary string
        summary = await run_tools_sequence_and_summarize(request.query)

        logger.info("Search completed successfully")

        return {
            "summary": summary,
            "results": [],  # Results are embedded in the summary
            "trace": [],
            "metadata": {"query": request.query, "mode": request.mode},
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
