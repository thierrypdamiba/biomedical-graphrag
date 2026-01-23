"""
Qdrant vector-search tool definitions.
"""

# ----------------------------
# Qdrant Vector Search tools definition
# ----------------------------

QDRANT_TOOLS = [
    {
        "type": "function",
        "name": "recommend_papers_based_on_constraints",
        "description": (
            "Recommend papers based on provided examples derived from user's query. "
            "extract to positive_examples what the paper should match, "
            "and to negative_examples - what the paper should avoid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positive_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Examples the recommended papers should be similar to.",
                },
                "negative_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Examples the recommended papers should be dissimilar to or avoid.",
                },
            },
            "required": ["negative_examples"],
        },
    },
    {
        "type": "function",
        "name": "retrieve_papers_hybrid",
        "description": "Search for academic papers matching a single topic or description using hybrid (dense + BM25) search.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single textual description of the paper topic to search for.",
                },
            },
            "required": ["query"],
        },
    },
]
