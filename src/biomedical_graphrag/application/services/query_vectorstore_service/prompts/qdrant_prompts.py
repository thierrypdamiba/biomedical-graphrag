"""Prompt templates for OpenAI Qdrant generation."""

QDRANT_GENERATION_PROMPT = """
You are a biomedical research assistant.

Generate an answer based on the following context from biomedical research papers.

Use the context to answer the question as accurately as possible.

Context: {context}
Question: {question}
Answer:
"""
