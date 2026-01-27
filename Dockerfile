FROM python:3.13-slim

WORKDIR /app

# Install build dependencies for biopython
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files and README (needed for hatchling build)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies only first (faster layer caching)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Now install the project itself
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8765

# Run directly with venv python (no uv run overhead)
CMD [".venv/bin/python", "-m", "biomedical_graphrag.api.server"]
