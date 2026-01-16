import asyncio

from biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore import AsyncQdrantVectorStore
from biomedical_graphrag.utils.json_util import load_gene_json, load_pubmed_json
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


async def ingest_data(recreate: bool = False) -> None:
    """
    Ingest data from datasets into the Qdrant vector store (async).
    Optionally recreate the collection before ingesting.
    """
    logger.info("🚀 Starting Qdrant data ingestion...")

    vector_store = AsyncQdrantVectorStore()
    try:
        if await vector_store.client.collection_exists(vector_store.collection_name):
            if recreate:
                logger.info("🔄 Recreating collection...")
                # recreate collection to ensure clean schema/state
                await vector_store.delete_collection()
                await vector_store.create_collection()
        else:
            logger.info(f"Collection with name {vector_store.collection_name} does not exist. Creating...")
            await vector_store.create_collection()

        # Load datasets
        logger.info("📂 Loading datasets...")
        pubmed_data = load_pubmed_json()
        gene_data = load_gene_json()
        logger.info(
            f"📊 Loaded {len(pubmed_data.get('papers', []))} \
            papers and {len(gene_data.get('genes', []))} genes"
        )

        # Upsert papers with attached genes in payload
        logger.info("📤 Starting data upsertion...")
        await vector_store.upsert_points(pubmed_data, gene_data)
        logger.info("✅ Embeddings ingestion complete.")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        raise
    finally:
        await vector_store.close()
        logger.info("🔌 Qdrant client connection closed")


if __name__ == "__main__":
    asyncio.run(ingest_data(recreate=True))
