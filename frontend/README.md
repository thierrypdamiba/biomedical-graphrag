# Biomedical GraphRAG Frontend

Next.js 16 dashboard for the Biomedical GraphRAG system - a hybrid search interface combining vector search (Qdrant) with knowledge graphs (Neo4j) for biomedical research papers.

## Quick Start (2 minutes)

The fastest way to run the frontend using the **hosted backend** (no API keys needed):

```bash
# 1. Clone the repository
git clone https://github.com/thierrypdamiba/biomedical-graphrag-frontend.git
cd biomedical-graphrag-frontend

# 2. Install dependencies
pnpm install

# 3. Copy environment config (uses hosted backend by default)
cp .env.example .env.local

# 4. Start the development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

The hosted backend at `https://ihdx3ugyrv.us-east-1.awsapprunner.com` provides:
- 26,788 biomedical papers indexed
- 606,396 knowledge graph nodes
- Full hybrid GraphRAG search

## Features

| Page | Description |
|------|-------------|
| **Assistant** | Chat interface for searching papers with hybrid vector + graph search |
| **Data** | Browse the Qdrant collection, view papers, run vector searches |
| **Graph** | View Neo4j knowledge graph statistics (Papers, Authors, Genes, etc.) |
| **Collections** | Manage and inspect Qdrant collections |
| **Console** | API request/response testing interface |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend API   │────▶│   Databases     │
│   (Next.js)     │     │   (FastAPI)     │     │                 │
│                 │     │                 │     │   - Qdrant      │
│   localhost:3000│     │   AWS App Runner│     │   - Neo4j       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Configuration Options

### Option 1: Use Hosted Backend (Recommended)

The default `.env.example` points to the hosted backend. No additional configuration needed.

```bash
cp .env.example .env.local
pnpm dev
```

### Option 2: Run Everything Locally

If you want to run your own backend with custom data:

1. **Clone the backend repository:**
   ```bash
   git clone https://github.com/thierrypdamiba/biomedical-graphrag.git
   cd biomedical-graphrag
   ```

2. **Set up backend environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Qdrant, Neo4j, and OpenAI credentials
   ```

3. **Install backend dependencies:**
   ```bash
   uv sync
   ```

4. **Update frontend to use local backend:**
   ```bash
   # In frontend/.env.local
   GRAPHRAG_API_URL=http://localhost:8765
   ```

5. **Start both services:**
   ```bash
   # Terminal 1: Start the backend API
   make run-api

   # Terminal 2: Start the frontend
   make run-frontend
   ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GRAPHRAG_API_URL` | Yes | `https://ihdx3ugyrv.us-east-1.awsapprunner.com` | Backend API URL |
| `QDRANT_URL` | No | - | Direct Qdrant access (fallback mode) |
| `QDRANT_API_KEY` | No | - | Qdrant API key (fallback mode) |
| `OPENAI_API_KEY` | No | - | OpenAI key for embeddings (fallback mode) |

## Tech Stack

- **Framework:** Next.js 16 with React 19
- **Language:** TypeScript 5
- **Styling:** Tailwind CSS v4
- **UI Components:** Radix UI
- **State Management:** Zustand
- **Package Manager:** pnpm

## Development

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev

# Build for production
pnpm build

# Run production server
pnpm start

# Lint code
pnpm lint
```

## API Endpoints

The frontend proxies requests to the backend through Next.js API routes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | POST | Hybrid GraphRAG search |
| `/api/neo4j/stats` | GET | Knowledge graph statistics |
| `/api/qdrant/collections` | GET | List Qdrant collections |
| `/api/qdrant/search` | POST | Direct vector search |
| `/api/config` | GET | Frontend configuration |

## Troubleshooting

**"Failed to fetch Neo4j stats"**
- Check that `GRAPHRAG_API_URL` in `.env.local` is correct
- Verify the backend is running: `curl https://ihdx3ugyrv.us-east-1.awsapprunner.com/health`

**Search returns no results**
- The hosted backend has biomedical papers about CRISPR, gene editing, cancer research
- Try queries like "CRISPR gene editing" or "breast cancer genes"

**CORS errors**
- Ensure you're accessing via `http://localhost:3000`, not `127.0.0.1`

## Related Repositories

- **Backend:** [biomedical-graphrag](https://github.com/thierrypdamiba/biomedical-graphrag) - FastAPI server, data pipelines, and CLI tools

## License

MIT
