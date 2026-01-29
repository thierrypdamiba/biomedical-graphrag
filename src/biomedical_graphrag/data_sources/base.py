import asyncio
from abc import ABC, abstractmethod
from typing import Any

from Bio import Entrez

from biomedical_graphrag.config import settings
from biomedical_graphrag.domain.paper import Paper
from biomedical_graphrag.utils.logger_util import setup_logging

logger = setup_logging()

_GLOBAL_RATE_LOCK = asyncio.Lock()
_LAST_REQUEST_TIME = 0.0


class BaseDataSource(ABC):
    """Abstract base class for all biomedical data sources with async support"""

    def __init__(self) -> None:
        self.email = settings.pubmed.email.get_secret_value()
        self.api_key = settings.pubmed.api_key.get_secret_value()
        logger.info(
            f"Entrez setup from BaseDataSource: email={'set' if self.email else 'not set'}, "
            f"api_key={'set' if self.api_key else 'not set'}"
        )
        if self.email:
            Entrez.email = str(self.email)  # type: ignore[assignment]
        if self.api_key:
            Entrez.api_key = str(self.api_key)  # type: ignore[assignment]

    async def _rate_limit(self) -> None:
        """Global rate limiter: ensures max 10 requests per second across all sources."""
        global _LAST_REQUEST_TIME
        async with _GLOBAL_RATE_LOCK:
            now = asyncio.get_event_loop().time()
            time_since_last = now - _LAST_REQUEST_TIME
            if time_since_last < 1.0 / 8.0:  # 10 requests per second
                await asyncio.sleep((1.0 / 8.0) - time_since_last)
            _LAST_REQUEST_TIME = asyncio.get_event_loop().time()

    async def search(self, query: str, max_results: int) -> list[str]:
        """Optional: Search the underlying source and return a list of IDs.

        Default implementation returns an empty list so subclasses that
        don't require searching (e.g., PMID-driven flows) don't need to
        implement it.
        """
        return []

    @abstractmethod
    async def fetch_entities(self, entity_ids: list[str]) -> list[Any]:
        """Fetch typed entities (papers, genes, etc.) for given IDs."""
        pass

    # Optional for paper-based sources
    async def fetch_papers(self, paper_ids: list[str]) -> list[Paper]:  # type: ignore[name-defined]
        """Fetch paper details for given paper IDs.

        Args:
            paper_ids: List of paper IDs to fetch.

        Returns:
            List of Paper objects.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    # Optional for paper-based sources
    async def fetch_citations(self, paper_id: str) -> dict:
        """Fetch citations for a given paper ID.

        Args:
            paper_id: The paper ID to fetch citations for.

        Returns:
            Dictionary containing citation information.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def collect_dataset(self, query: str, max_results: int) -> Any:
        """Template method to collect a dataset for the given source."""
        pass
