# Biomedical GraphRAG Frontend

Next.js 16 dashboard for the Biomedical GraphRAG system.

## Prerequisites

- Node.js 18+
- pnpm (`npm install -g pnpm`)
- Running backend services:
  - Qdrant (vector database)
  - Neo4j (graph database)
  - GraphRAG API server (`make run-api` from project root)

## Setup

1. Install dependencies:
```bash
pnpm install
```

2. Create `.env.local` from the example:
```bash
cp .env.example .env.local
```

3. Edit `.env.local` with your credentials:
```
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=biomedical_papers
OPENAI_API_KEY=your-openai-api-key
GRAPHRAG_API_URL=http://localhost:8765
```

4. Start the development server:
```bash
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

## Running with the Backend

From the project root, run in two terminals:

```bash
# Terminal 1: Start the GraphRAG API
make run-api

# Terminal 2: Start the frontend
make run-frontend
```

## Features

- **Collections**: Browse Qdrant vector collections
- **Graph**: View Neo4j graph statistics
- **Console**: Run GraphRAG searches
- **Assistant**: Code examples and API documentation
