"""Prompt templates for hybrid GraphRAG querying."""

import json
from typing import Any

 #TO DO: improve prompts
QDRANT_PROMPT = """
You are a biomedical query router for PubMed papers retrieval.

Your ONLY task is to choose and call exactly one Qdrant tool, 
Either similarity search or recommendations based on constraints.
This tool should get relevant academic papers based on the user's intent.
Do NOT generate any text.

Rules:
- Use `retrieve_papers_hybrid` if you need to leverage similarity search (a combination of semantic and lexical).
- Use `recommend_papers_based_on_constraints` if the user in its query provides COUNTER-examples
  ('excluding', 'but not about', 'not like', 'shouldn't be about') regarding what the paper should NOT be about.

Argument rules:
- For `retrieve_papers_hybrid`, build `query` from the user input
  for hybrid (dense + BM25) search. Preserve meaning. Do not add new topics.
- For `recommend_papers_based_on_constraints`:
  - `positive_examples` = what user wants papers to BE about
  - `negative_examples` = what user wants papers to NOT BE about
  Use only information from the user.

Call the selected tool and return nothing else.

User Question:
{question}
"""

NEO4J_PROMPT = """
You are a biomedical reasoning assistant.
You CANNOT answer the question using Qdrant context alone.
Always call at least one Neo4j enrichment tool before producing text.

Steps:
1. Read the user question and Qdrant context.
2. Identify biomedical entities (PMIDs, authors, MeSH terms, institutions).
3. Decide which Neo4j enrichment tool(s) to call, using the schema below.
4. Provide tool arguments based on the context.
5. Call the tool(s); after all calls are complete, you may generate text.

Neo4j Graph Schema:
{schema}

User Question:
{question}

Retrieved Qdrant Context:
{qdrant_points_metadata}
"""

FUSION_SUMMARY_PROMPT = """
You are a biomedical research assistant combining two data sources:

- Qdrant (vector search results from PubMed papers)
- Neo4j (structured knowledge graph with genes, diseases, authors, institutions)

Your goal:
Synthesize both sources into a well-structured, factual answer for a biomedical researcher.

Format your response using this exact markdown structure:

### Key Findings
A numbered list of exactly {limit} finding(s) — one per retrieved paper. Each finding should reference the source paper by PMID (e.g. PMID: 12345678). Be specific and cite data points. Do NOT add extra findings beyond the {limit} retrieved paper(s).

### Graph Insights
Describe what the Neo4j knowledge graph revealed — gene relationships, author networks, institutional connections, or disease associations that extend beyond what the papers alone show. Use bullet points (- ) for each insight.

### Synthesis
A concise paragraph combining both sources into a cohesive answer. Mention how graph data confirms, extends, or adds context to the paper findings. End with any limitations or gaps.

Rules:
- Use **bold** for gene names, drug names, and key terms
- Always cite PMIDs inline like (PMID: 12345678)
- Be precise and factual — no speculation
- Keep each section focused and concise
- If Neo4j returned no useful results, skip the Graph Insights section
- Do NOT mention "Qdrant" or "Neo4j" by name — refer to them as "retrieved literature" and "knowledge graph"

User Question:
{question}

Retrieved Qdrant Context (JSON):
{qdrant_context}

Neo4j Enrichment Results (JSON):
{neo4j_results}
"""


def _format_qdrant_points(qdrant_points_metadata: list[dict]) -> str: #TO DO: change to just concatenating paper titles & abstracts
    """Render Qdrant tool results into a compact, LLM-friendly text block."""
    try:
        papers = [point["payload"]["paper"] for point in qdrant_points_metadata]
        return json.dumps(papers, indent=2, ensure_ascii=False)
    except TypeError:
        # If some objects aren't JSON serializable, fall back to a safe string.
        return str(qdrant_points_metadata)


def fusion_summary_prompt(
    question: str, qdrant_results: list[dict], neo4j_results: dict[str, Any], limit: int = 5
) -> str:
    """Generate the fusion summary prompt.

    Args:
        question: The user question.
        qdrant_results: The Qdrant tool results / points metadata.
        neo4j_results: The Neo4j results.
        limit: Number of papers retrieved (controls Key Findings count).

    Returns:
        The fusion summary prompt.
    """
    return FUSION_SUMMARY_PROMPT.format(
        question=question,
        qdrant_context=_format_qdrant_points(qdrant_results),
        neo4j_results=json.dumps(neo4j_results, indent=2),
        limit=limit,
    )
