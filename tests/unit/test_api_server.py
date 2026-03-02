"""Unit tests for FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient

from biomedical_graphrag.api.server import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    TraceStep,
    app,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_check_returns_healthy(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestSearchRequestModel:
    def test_default_values(self) -> None:
        req = SearchRequest(query="test")
        assert req.query == "test"
        assert req.limit == 5
        assert req.mode == "graphrag"

    def test_custom_values(self) -> None:
        req = SearchRequest(query="BRCA1", limit=3, mode="dense")
        assert req.limit == 3
        assert req.mode == "dense"

    def test_limit_validation_max(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=10)

    def test_limit_validation_min(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)


class TestResponseModels:
    def test_search_response_defaults(self) -> None:
        resp = SearchResponse()
        assert resp.summary is None
        assert resp.results == []
        assert resp.trace == []
        assert resp.metadata == {}

    def test_trace_step_creation(self) -> None:
        step = TraceStep(name="qdrant_search", arguments={"query": "test"}, result_count=5)
        assert step.name == "qdrant_search"
        assert step.arguments == {"query": "test"}
        assert step.result_count == 5

    def test_health_response(self) -> None:
        resp = HealthResponse()
        assert resp.status == "healthy"


class TestSearchEndpoint:
    def test_search_missing_query(self, client: TestClient) -> None:
        response = client.post("/api/graphrag-query", json={})
        assert response.status_code == 422

    def test_search_invalid_limit(self, client: TestClient) -> None:
        response = client.post("/api/graphrag-query", json={"query": "test", "limit": -1})
        assert response.status_code == 422
