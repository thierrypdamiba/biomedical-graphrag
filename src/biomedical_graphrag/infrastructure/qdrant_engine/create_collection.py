import asyncio

from biomedical_graphrag.infrastructure.qdrant_engine.qdrant_vectorstore import AsyncQdrantVectorStore


async def create_collection() -> None:
    """Create a Qdrant collection based on settings (async).

    Args:
            None
    Returns:
            None
    """
    vector_store = AsyncQdrantVectorStore()
    try:
        await vector_store.create_collection()
    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(create_collection())
