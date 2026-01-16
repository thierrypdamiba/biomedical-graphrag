"""
CLI for hybrid GraphRAG querying:
1. Retrieve relevant chunks from Qdrant.
2. Use LLM to select and run Neo4j enrichment tools.
3. Fuse both sources into one concise biomedical summary.
"""

import asyncio
import sys

from biomedical_graphrag.application.services.hybrid_service.tool_calling import (
    run_graph_enrichment_and_summarize,
)
from biomedical_graphrag.application.services.query_vectorstore_service.qdrant_query import (
    AsyncQdrantQuery,
)
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


async def main() -> None:
    """Main function to run the fusion query.
    Args:
        None

    Returns:
        None
    """
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        logger.info("Please enter your question:")
        question = input().strip()

    logger.info(f"Processing question: {question}")

    # --- Step 1: Retrieve semantic context from Qdrant ---
    qdrant_query = AsyncQdrantQuery()
    try:
        documents = await qdrant_query.retrieve_documents_dense(question)
        chunks = []
        for doc in documents:
            payload = doc.get("payload", {})
            if isinstance(payload, dict) and "content" in payload: # which content? there's no key like this, is there
                chunks.append(str(payload["content"]))
            else:
                chunks.append(str(payload))

        # --- Step 2: Enrichment + Fusion summary (two-phase internally) ---
        answer = run_graph_enrichment_and_summarize(question, chunks)

        print("\n=== Unified Biomedical Answer ===\n")
        print(answer)
    finally:
        await qdrant_query.close()


if __name__ == "__main__":
    asyncio.run(main())
