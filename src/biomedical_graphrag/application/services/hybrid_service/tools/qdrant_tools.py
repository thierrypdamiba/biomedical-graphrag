"""
Qdrant vector-search tool definitions.
"""

# ----------------------------
# Qdrant Vector Search tools definition
# ----------------------------

QDRANT_TOOLS = [
    {
        "type": "function",
        "name": "recommend_papers_based_on_examples",
        "description": (
            "Recommend papers based on example descriptions. "
            "Use positive_examples for topics the paper should match, "
            "and negative_examples for topics the paper should avoid."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "positive_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Descriptions of topics the recommended papers should be similar to.",
                },
                "negative_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Descriptions of topics the recommended papers should be dissimilar to or avoid.",
                },
            },
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
