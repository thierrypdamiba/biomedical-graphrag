"""Test configuration and fixtures."""

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import SecretStr

from biomedical_graphrag.config import OpenAISettings, Settings
from biomedical_graphrag.domain.author import Author
from biomedical_graphrag.domain.meshterm import MeSHTerm
from biomedical_graphrag.domain.paper import Paper
from biomedical_graphrag.infrastructure.neo4j_db.neo4j_client import AsyncNeo4jClient


@pytest.fixture
def sample_author() -> Author:
    """Create a sample author for testing."""
    return Author(
        name="John Doe",
        first_name="John",
        last_name="Doe",
        affiliations=["Test University", "Research Institute"],
    )


@pytest.fixture
def sample_mesh_term() -> MeSHTerm:
    """Create a sample MeSH term for testing."""
    return MeSHTerm(
        term="Test Term", ui="D123456", major_topic=True, qualifiers=["therapy", "diagnosis"]
    )


@pytest.fixture
def sample_paper(sample_author: Author, sample_mesh_term: MeSHTerm) -> Paper:
    """Create a sample paper for testing."""
    return Paper(
        pmid="12345678",
        title="Test Paper Title",
        abstract="This is a test abstract for the paper.",
        authors=[sample_author],
        mesh_terms=[sample_mesh_term],
        publication_date="2024-01-01",
        journal="Test Journal",
        doi="10.1000/test.doi",
    )


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings with default values."""
    return Settings(
        openai=OpenAISettings(
            api_key=SecretStr("test-key"), model="gpt-4o-mini", temperature=0.0, max_tokens=1500
        )
    )


@pytest.fixture
def mock_qdrant_client() -> AsyncMock:
    """Create a mock Qdrant client for testing."""
    mock_client = AsyncMock()
    # Set up common async methods
    for method in ["create_collection", "delete_collection", "upsert", "search", "close"]:
        setattr(mock_client, method, AsyncMock())
    return mock_client


@pytest.fixture
def mock_neo4j_driver() -> AsyncMock:
    """Create a mock Neo4j driver for testing."""
    mock_driver = AsyncMock()
    mock_session = AsyncMock()

    # Set up session context manager behavior
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_driver.session.return_value.__aexit__.return_value = None

    # Set up async methods
    for method in ["close"]:
        setattr(mock_driver, method, AsyncMock())

    return mock_driver


@pytest.fixture
def mock_qdrant_vectorstore(mock_qdrant_client: AsyncMock) -> Mock:
    """Create a mock QdrantVectorStore for testing."""
    mock_vectorstore = Mock()
    mock_vectorstore.client = mock_qdrant_client
    # Set up async methods
    for method in ["create_collection", "close"]:
        setattr(mock_vectorstore, method, AsyncMock())
    return mock_vectorstore


@pytest.fixture
def mock_neo4j_client(mock_neo4j_driver: AsyncMock) -> AsyncMock:
    """Create a mock Neo4jClient for testing."""
    mock_client = AsyncMock(spec=AsyncNeo4jClient)
    mock_client.driver = mock_neo4j_driver
    # Set up async methods
    for method in ["execute", "create_graph", "close"]:
        setattr(mock_client, method, AsyncMock())
    return mock_client


@pytest.fixture
def generic_async_mock() -> AsyncMock:
    """Generic AsyncMock fixture for simple async method mocking."""
    return AsyncMock()
