import asyncio

from biomedical_graphrag.infrastructure.qdrant_engine.qdrant_vectorstore import AsyncQdrantVectorStore


async def delete_collection() -> None:
    """Delete a Qdrant collection based on settings (async).

    Args:
            None
    Returns:
            None
    """
    vector_store = AsyncQdrantVectorStore()
    try:
        await vector_store.delete_collection()
    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(delete_collection())
