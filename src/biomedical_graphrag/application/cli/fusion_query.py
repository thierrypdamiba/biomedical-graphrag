"""
CLI for hybrid GraphRAG querying:
1. Retrieve relevant papers from Qdrant based on the tool selected.
2. Use LLM to select and run Neo4j enrichment tools.
3. Fuse both sources into one concise biomedical summary.
"""

import asyncio
import sys

from biomedical_graphrag.application.services.hybrid_service.tool_calling import (
    run_tools_sequence_and_summarize,
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

    try:
        answer = await run_tools_sequence_and_summarize(question)

        print("\n=== Unified Biomedical Answer ===\n")
        print(answer)
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
