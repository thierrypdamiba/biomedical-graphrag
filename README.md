# Biomedical GraphRAG

![Neo4j UI](static/image.png)

<div align="center">

<!-- Project Status -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python version](https://img.shields.io/badge/python-3.13.8-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

<!-- Providers -->

[![Qdrant](https://img.shields.io/badge/Qdrant-1.15.1-5A31F4?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.28.2-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-2.3.0-412991?logo=openai&logoColor=white)](https://openai.com/)

</div>

## Table of Contents

- [Biomedical GraphRAG](#biomedical-graphrag)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Data Collection](#data-collection)
    - [Infrastructure Setup](#infrastructure-setup)
      - [Neo4j Graph Database](#neo4j-graph-database)
      - [Qdrant Vector Search Engine](#qdrant-vector-search-engine)
    - [Query Commands](#query-commands)
      - [Hybrid Neo4j + Qdrant Queries](#hybrid-neo4j--qdrant-queries)
      - [Sample Queries](#sample-queries)
    - [API Server](#api-server)
    - [Frontend](#frontend)
    - [Docker](#docker)
    - [Testing](#testing)
    - [Quality Checks](#quality-checks)
  - [License](#license)

## Overview

A biomedical context engineering system. An agent uses Qdrant vector search engine tools (hybrid retrieval, recommendations) and Neo4j graph database tools (graph enrichment) to gather context, then fuses it into a single biomedical answer.

> Originally forked from [benitomartin/biomedical-graphrag](https://github.com/benitomartin/biomedical-graphrag).

**References:**
- Video: [PubMed Navigator](https://www.youtube.com/watch?v=3NWTi90i6C4)
- Article: [Building a Biomedical GraphRAG: When Knowledge Graphs Meet Vector Search](https://aiechoes.substack.com/p/building-a-biomedical-graphrag-when)

**Key Features:**

- **Context Engineering**: Agent orchestrates Qdrant and Neo4j tools, fusing results into a single answer
- **Qdrant Vector Search Engine**: Hybrid retrieval (dense + BM25 with reranking) and constraint-based recommendations
- **Neo4j Graph Database**: Graph enrichment via ontology-based tools (collaborator networks, MeSH relations, gene co-mentions)
- **Data Integration**: Processes PubMed papers, gene data, and research citations
- **Biomedical Schema**: Specialized graph schema for papers, authors, institutions, genes, and MeSH terms
- **Async Processing**: High-performance async data collection and processing

## Project Structure

```text
biomedical-graphrag/
├── .github/                    # GitHub workflows and templates
├── data/                       # Dataset storage (PubMed, Gene data)
├── src/
│   └── biomedical_graphrag/
│       ├── api/                # FastAPI server
│       │   └── server.py       # PubMed Navigator API endpoints
│       ├── application/        # Application layer
│       │   ├── cli/            # Command-line interfaces
│       │   └── services/       # Business logic services
│       ├── config.py           # Configuration management
│       ├── data_sources/       # Data collection modules
│       ├── domain/             # Domain models and entities
│       ├── infrastructure/     # Database and external service adapters
│       └── utils/              # Utility functions
├── static/                     # Static assets (images, etc.)
├── tests/                      # Test suite
├── Dockerfile                  # Docker build configuration
├── LICENSE                     # MIT License
├── Makefile                    # Build and development commands
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
└── uv.lock                     # Dependency lock file
```

## Prerequisites

| Requirement                                            | Description                             |
| ------------------------------------------------------ | --------------------------------------- |
| [Python 3.13+](https://www.python.org/downloads/)      | Programming language                    |
| [uv](https://docs.astral.sh/uv/)                       | Package and dependency manager          |
| [Neo4j](https://neo4j.com/)                            | Graph database for knowledge graphs     |
| [Qdrant](https://qdrant.tech/)                         | Vector search engine for embeddings     |
| [OpenAI](https://openai.com/)                          | LLM provider for queries and embeddings |
| [PubMed](https://www.ncbi.nlm.nih.gov/books/NBK25501/) | Biomedical literature database          |

## Installation

1. Clone the repository:

   ```bash
   git clone git@github.com:thierrypdamiba/biomedical-graphrag.git
   cd biomedical-graphrag
   ```

1. Create a virtual environment:

   ```bash
   uv venv
   ```

1. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

1. Install the required packages:

   ```bash
  uv sync --all-groups --all-extras
   ```

1. Create a `.env` file in the root directory:

   ```bash
    cp .env.example .env
   ```

## Usage

### Configuration

Configure API keys, model names, and other settings by editing the `.env` file:

```bash
# OpenAI Configuration
OPENAI__API_KEY=your_openai_api_key_here
OPENAI__MODEL=gpt-4o-mini
OPENAI__TEMPERATURE=0.0
OPENAI__MAX_TOKENS=1500

# Neo4j Configuration
NEO4J__URI=bolt://localhost:7687
NEO4J__USERNAME=neo4j
NEO4J__PASSWORD=your_neo4j_password
NEO4J__DATABASE=neo4j

# Qdrant Configuration
QDRANT__URL=http://localhost:6333
QDRANT__API_KEY=your_qdrant_api_key
QDRANT__COLLECTION_NAME=biomedical_papers
QDRANT__EMBEDDING_MODEL=text-embedding-3-large
QDRANT__EMBEDDING_DIMENSION=1536
QDRANT__RERANKER_EMBEDDING_DIMENSION=3072
QDRANT__ESTIMATE_BM25_AVG_LEN_ON_X_DOCS=300
QDRANT__CLOUD_INFERENCE=false

# PubMed Configuration (optional)
PUBMED__EMAIL=your_email@example.com
PUBMED__API_KEY=your_pubmed_api_key

# Data Paths
JSON_DATA__PUBMED_JSON_PATH=data/pubmed_dataset.json
JSON_DATA__GENE_JSON_PATH=data/gene_dataset.json
```

### Data Collection

The system includes data collectors for biomedical and gene datasets:

```bash
# Collect PubMed papers and metadata
make pubmed-data-collector-run

# Override defaults (optional)
make pubmed-data-collector-run QUERY="cancer immunotherapy" MAX_RESULTS=200
```

```bash
# Collect gene information related to the pubmed dataset
make gene-data-collector-run
```

```bash
# Enrich PubMed dataset with related papers
make enrich-pubmed-dataset

# Skip the first N papers as sources (optional)
make enrich-pubmed-dataset START_INDEX=1000
```

### Infrastructure Setup

#### Neo4j Graph Database

```bash
# Create the knowledge graph from datasets
make create-graph

# Delete all graph data (clean slate)
make delete-graph
```

#### Qdrant Vector Search Engine

```bash
# Create vector collection for embeddings
make create-qdrant-collection

# Ingest embeddings into Qdrant
make ingest-qdrant-data

# Delete vector collection
make delete-qdrant-collection
```

Notes:

- Embeddings are built from **PubMed paper abstracts**.
- This project uses OpenAI embeddings **Matryoshka Representation Learning (MRL)** feature:
  - `QDRANT__EMBEDDING_DIMENSION` is the prefix dimension used for **retrieval** (stored in Qdrant as the `Dense` vector).
  - `QDRANT__RERANKER_EMBEDDING_DIMENSION` is the (larger) prefix dimension used for **reranking** (stored in Qdrant as the `Reranker` vector).
- `make ingest-qdrant-data` currently recreates the collection each run (see `qdrant_ingestion.py`).
  If you don't want that, change `recreate=True` to `False`. There's also an `only_new` parameter which defaults to `True`, so we ingest only papers whose PMID is not already
  present in the collection. Set `only_new=False` if you'd prefer to overwrite existing points or on a clean ingestion (then it should be `False`!)
- The collection is configured by default with **scalar quantization** (compressed dense vectors).
- `QDRANT__CLOUD_INFERENCE=true` enables **Qdrant Cloud Inference** when embeddings are computed by Qdrant Cloud.
- `QDRANT__ESTIMATE_BM25_AVG_LEN_ON_X_DOCS` controls how many documents are sampled to estimate the average abstract length used by **BM25**. This helps calibrate BM25-based scoring when using dense+BM25 hybrid retrieval.

### Query Commands

#### Hybrid Neo4j + Qdrant Queries

```bash
# Run a custom natural language query
make custom-graph-query QUESTION="What are the latest research trends in cancer immunotherapy?"

# Or run directly with the CLI (positional args)
uv run src/biomedical_graphrag/application/cli/fusion_query.py "What are the latest research trends in cancer immunotherapy?"
```

The hybrid query system combines vector search engine (Qdrant) with graph enrichment (Neo4j):
- Author collaboration networks
- Citation analysis and paper relationships
- Gene-paper associations
- MeSH term relationships
- Institution affiliations

LLM-powered tool selection & fusion:
- Runs one Qdrant tool: hybrid retrieval (BM25 + dense & reranking) or recommendations with constraints — to fetch relevant papers.
- Calls Neo4j enrichment tools for graph evidence.
- Produces one fused answer from both sources.

Output:

- The hybrid CLI prints a final section titled "=== Unified Biomedical Answer ===".

#### Sample Queries

- Who collaborates with Jennifer Doudna on CRISPR research?

- Which genes are frequently co-mentioned with TP53?

### API Server

The project includes a FastAPI server (PubMed Navigator) that exposes the context engineering pipeline via HTTP endpoints:

```bash
# Start the API server (runs on port 8765)
make run-api
```

**Available Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/neo4j/stats` | Neo4j graph statistics (node/relationship counts) |
| POST | `/api/graphrag-query` | Context engineering search (Qdrant + Neo4j) |

**Search Request Example:**

```bash
curl -X POST http://localhost:8765/api/graphrag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "What genes are associated with breast cancer?", "limit": 5}'
```

### Frontend

The frontend is maintained in a separate repository: **[biomedical-graphrag-frontend](https://github.com/thierrypdamiba/biomedical-graphrag-frontend)**

The quickest way to run it locally:

```bash
# Auto-clones the frontend repo and starts it (requires pnpm)
make run-frontend
```

Or manually:

```bash
git clone https://github.com/thierrypdamiba/biomedical-graphrag-frontend.git
cd biomedical-graphrag-frontend
pnpm install
pnpm dev
```

The frontend connects to the hosted backend at `https://biomedical-graphrag-9qqm.onrender.com` by default, or you can point it to a local backend via `GRAPHRAG_API_URL=http://localhost:8765`.

### Docker

Build and run the API server with Docker:

```bash
# Build the image
docker build -t biomedical-graphrag:latest .

# Run the container
docker run --rm -p 8765:8765 --env-file .env biomedical-graphrag:latest

# Test health endpoint
curl http://localhost:8765/health
```

### Troubleshooting

- **Make fails immediately with ".env file is missing"**
  - Create it with `cp .env.example .env` and fill in required values.

- **Qdrant ingestion/query fails**
  - Confirm Qdrant is running and `QDRANT__URL` points to it.
  - If you changed embedding model, recreate the collection.

- **Hybrid queries return errors about Neo4j**
  - Make sure Neo4j is running and credentials in `NEO4J__*` are correct.
  - Ensure the graph exists (`make create-graph`) before asking graph enrichment questions.

### Testing

Run all tests:

```bash
make tests
```

### Quality Checks

Run all quality checks (lint, format, type check, clean):

```bash
make all-check
make all-fix
```

Individual Commands:

- Display all available commands:

  ```bash
  make help
  ```

- Check code static typing

  ```bash
  make mypy
  ```

- Clean cache and build files:

  ```bash
  make clean
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
