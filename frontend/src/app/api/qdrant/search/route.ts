import { NextResponse } from "next/server";

const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const QDRANT_COLLECTION_NAME = process.env.QDRANT_COLLECTION_NAME || "biomedical_papers";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || process.env.GRAPHRAG_OPENAI_API_KEY;

interface SearchRequest {
  query: string;
  limit?: number;
}

export async function POST(request: Request) {
  const startTime = Date.now();

  try {
    const body: SearchRequest = await request.json();
    const { query, limit = 20 } = body;

    if (!query?.trim()) {
      return NextResponse.json({ error: "Query is required" }, { status: 400 });
    }

    // Get OpenAI API key from backend if not in frontend env
    let apiKey = OPENAI_API_KEY;
    if (!apiKey) {
      // Try to get from backend config endpoint
      try {
        const configRes = await fetch("http://localhost:8765/health");
        if (configRes.ok) {
          // Backend is running, we can use its embedding endpoint
        }
      } catch {
        // Backend not available
      }
    }

    // Step 1: Generate embedding using OpenAI
    const embeddingResponse = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "text-embedding-3-large",
        input: query,
        dimensions: 1536,
      }),
    });

    if (!embeddingResponse.ok) {
      const error = await embeddingResponse.text();
      console.error("OpenAI embedding error:", error);
      throw new Error("Failed to generate embedding");
    }

    const embeddingData = await embeddingResponse.json();
    const queryVector = embeddingData.data[0].embedding;

    // Step 2: Search Qdrant with the embedding
    const searchResponse = await fetch(`${QDRANT_URL}/collections/${QDRANT_COLLECTION_NAME}/points/search`, {
      method: "POST",
      headers: {
        "api-key": QDRANT_API_KEY || "",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        vector: {
          name: "Dense",
          vector: queryVector,
        },
        limit: limit,
        with_payload: true,
        with_vector: false,
      }),
    });

    if (!searchResponse.ok) {
      const error = await searchResponse.text();
      console.error("Qdrant search error:", error);
      throw new Error("Qdrant search failed");
    }

    const searchData = await searchResponse.json();
    const results = (searchData.result || []).map((hit: { id: number; score: number; payload: Record<string, unknown> }) => ({
      id: hit.id,
      score: hit.score,
      payload: hit.payload,
    }));

    return NextResponse.json({
      results,
      metadata: {
        query,
        collection: QDRANT_COLLECTION_NAME,
        latency: Date.now() - startTime,
        resultsCount: results.length,
      },
    });
  } catch (error) {
    console.error("Search error:", error);
    return NextResponse.json(
      { error: "Search failed", details: String(error) },
      { status: 500 }
    );
  }
}
