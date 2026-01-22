from qdrant_client.models import models

from biomedical_graphrag.infrastructure.qdrant_db.qdrant_vectorstore import AsyncQdrantVectorStore
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()


class AsyncQdrantQuery:
    """Handles querying Qdrant vector search engine for natural language questions (async)."""

    def __init__(self) -> None:
        """
        Initialize the async Qdrant client with connection parameters.
        """
        self.qdrant_client = AsyncQdrantVectorStore()

    async def close(self) -> None:
        """Close the async Qdrant client."""
        await self.qdrant_client.close()

    async def retrieve_papers_dense(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Query the Qdrant vector store for similar papers (async).
        Vanilla dense search on quantized openAI embeddings.

        Args:
            query (str): Input query to query.
            top_k (int): Number of top similar papers to retrieve.
        Returns:
            List of dictionaries containing the top_k similar papers.
        """
        if self.qdrant_client.cloud_inference:
            dense_vector = self.qdrant_client._define_openai_vectors(
                query, dimensions=self.qdrant_client.embedding_dimension
            )
        else:
            dense_vector = await self.qdrant_client._get_openai_vectors(
                query, dimensions=self.qdrant_client.embedding_dimension
            )

        search_result = await self.qdrant_client.client.query_points(
            collection_name=self.qdrant_client.collection_name,
            query=dense_vector,
            using="Dense",
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    oversampling=3.0,  # retrieve 3 * top_k quantized vectors
                    rescore=True,  # to rescore with original vectors
                )
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        results = [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in search_result.points
        ]
        return results

    async def retrieve_papers_hybrid(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Query the Qdrant vector search engine (async).
        Hybrid dense + lexical search, fused by reranking with text-embedding-3-large.

        Args:
            query (str): Input query.
            top_k (int): Number of top similar papers to retrieve.
        Returns:
            List of dictionaries containing the top_k similar papers.
        """
        if self.qdrant_client.cloud_inference:
            retriever_vector = self.qdrant_client._define_openai_vectors(
                query, dimensions=self.qdrant_client.embedding_dimension
            )
            reranker_vector = self.qdrant_client._define_openai_vectors(
                query, dimensions=self.qdrant_client.reranker_embedding_dimension
            )
        else:
            openai_vector = await self.qdrant_client._get_openai_vectors(
                query, dimensions=self.qdrant_client.reranker_embedding_dimension
            )
            retriever_vector = openai_vector[
                : self.qdrant_client.embedding_dimension
            ]  # Qdrant normalizes vectors used with COSINE automatically on upsert/query
            reranker_vector = openai_vector  # full precision for reranking

        sparse_vector = self.qdrant_client._define_bm25_vectors(query)

        search_result = await self.qdrant_client.client.query_points(
            collection_name=self.qdrant_client.collection_name,
            prefetch=[
                models.Prefetch(
                    query=retriever_vector,
                    using="Dense",
                    params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            oversampling=3.0,  # retrieve 3 * top_k quantized vectors
                            rescore=True,  # to rescore with original vectors
                        )
                    ),
                    limit=top_k,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="Lexical",
                    limit=top_k,
                ),
            ],
            # query=models.RrfQuery(rrf=models.Rrf(k=60)),
            query=reranker_vector,
            using="Reranker",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in search_result.points
        ]
        return results

    async def recommend_papers_based_on_examples(
        self,
        positive_examples: list[str] | None,
        negative_examples: list[str] | None,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Recommend papers based on positive and negative examples (async).
        FOR NOW, only covering text descriptions as input (not mixed with PMIDs)

        Args:
            positive_examples (list[str]): List of positive abstracts (similar to this).
            negative_examples (list[str]): List of negative abstracts (dissimilar to this).
            top_k (int): Number of top recommended papers to retrieve.
        Returns:
            List of dictionaries containing the top_k recommended papers.
        """

        if self.qdrant_client.cloud_inference:
            if positive_examples:
                positive_vectors = [
                    self.qdrant_client._define_openai_vectors(
                        pos_example, dimensions=self.qdrant_client.embedding_dimension
                    )
                    for pos_example in positive_examples
                ]
            else:
                positive_vectors = None
            if negative_examples:
                negative_vectors = [
                    self.qdrant_client._define_openai_vectors(
                        neg_example, dimensions=self.qdrant_client.embedding_dimension
                    )
                    for neg_example in negative_examples
                ]
            else:
                negative_vectors = None
        else:
            if positive_examples:
                positive_vectors = [
                    await self.qdrant_client._get_openai_vectors(
                        pos_example, dimensions=self.qdrant_client.embedding_dimension
                    )
                    for pos_example in positive_examples
                ]
            else:
                positive_vectors = None
            if negative_examples:
                negative_vectors = [
                    await self.qdrant_client._get_openai_vectors(
                        neg_example, dimensions=self.qdrant_client.embedding_dimension
                    )
                    for neg_example in negative_examples
                ]
            else:
                negative_vectors = None

        recommendation_result = await self.qdrant_client.client.query_points(
            collection_name=self.qdrant_client.collection_name,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=positive_vectors,
                    negative=negative_vectors,
                    strategy=models.RecommendStrategy.AVERAGE_VECTOR,
                )
            ),
            using="Dense",
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in recommendation_result.points
        ]
        return results
