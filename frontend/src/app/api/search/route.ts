import { NextResponse } from "next/server";

// GraphRAG API endpoint (Python FastAPI server)
const GRAPHRAG_API_URL = process.env.GRAPHRAG_API_URL || "http://localhost:8765";

// Fallback to direct Qdrant if GraphRAG is not available
const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const QDRANT_COLLECTION_NAME = process.env.QDRANT_COLLECTION_NAME || "biomedical_papers";

interface SearchRequest {
  query: string;
  limit?: number;
  mode?: "dense" | "sparse" | "hybrid" | "graphrag";
}

interface TraceStep {
  name: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  details?: Record<string, unknown>;
}

export async function POST(request: Request) {
  const startTime = Date.now();

  try {
    const body: SearchRequest = await request.json();
    const { query, limit = 10, mode = "graphrag" } = body;

    // Try GraphRAG API first
    if (mode === "graphrag" || mode === "hybrid") {
      try {
        const graphragResponse = await fetch(`${GRAPHRAG_API_URL}/api/search`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query,
            limit,
            mode: "graphrag",
          }),
        });

        if (graphragResponse.ok) {
          const data = await graphragResponse.json();
          return NextResponse.json({
            summary: data.summary,
            results: data.results || [],
            trace: data.trace || [],
            metadata: {
              ...data.metadata,
              totalLatency: Date.now() - startTime,
            },
          });
        }

        // If GraphRAG fails, fall through to fallback
        console.warn("GraphRAG API not available, falling back to text search");
      } catch (graphragError) {
        console.warn("GraphRAG API error, falling back to text search:", graphragError);
      }
    }

    // Fallback: Direct Qdrant text search
    const trace: TraceStep[] = [];

    // Step 1: Query normalization
    const normalizeStart = Date.now();
    const normalizedQuery = query.trim().toLowerCase();
    trace.push({
      name: "Query normalization",
      startTime: normalizeStart - startTime,
      duration: Date.now() - normalizeStart,
      details: { original: query, normalized: normalizedQuery },
    });

    // Step 2: Fetch points from Qdrant
    const fetchStart = Date.now();
    const response = await fetch(`${QDRANT_URL}/collections/${QDRANT_COLLECTION_NAME}/points/scroll`, {
      method: "POST",
      headers: {
        "api-key": QDRANT_API_KEY || "",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        limit: 500,
        with_payload: true,
        with_vector: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Qdrant API error: ${response.status}`);
    }

    const data = await response.json();
    trace.push({
      name: "Qdrant fetch",
      startTime: fetchStart - startTime,
      duration: Date.now() - fetchStart,
      details: {
        collection: QDRANT_COLLECTION_NAME,
        pointsFetched: data.result?.points?.length || 0,
      },
    });

    // Step 3: Text-based filtering
    const filterStart = Date.now();
    const queryTerms = normalizedQuery.split(/\s+/).filter(t => t.length >= 2);

    const scoredResults = (data.result?.points || [])
      .map((point: { id: string | number; payload: Record<string, unknown> }) => {
        const paper = (point.payload?.paper as Record<string, unknown>) || point.payload;
        const title = String(paper?.title || "").toLowerCase();
        const abstract = String(paper?.abstract || "").toLowerCase();
        const meshTerms = (paper?.mesh_terms as Array<{ term?: string }> || [])
          .map(m => (m.term || "").toLowerCase())
          .join(" ");
        const allText = `${title} ${abstract} ${meshTerms}`;

        let score = 0;
        let matchedTerms = 0;

        for (const term of queryTerms) {
          const wordBoundaryRegex = new RegExp(`\\b${term}`, 'i');
          if (wordBoundaryRegex.test(title)) {
            score += 0.5;
            matchedTerms++;
          } else if (wordBoundaryRegex.test(abstract)) {
            score += 0.3;
            matchedTerms++;
          } else if (wordBoundaryRegex.test(meshTerms)) {
            score += 0.4;
            matchedTerms++;
          } else if (allText.includes(term)) {
            score += 0.15;
            matchedTerms++;
          }
        }

        const normalizedScore = queryTerms.length > 0
          ? Math.min((score / queryTerms.length) * (matchedTerms / queryTerms.length + 0.5), 1)
          : 0;

        return { ...point, score: normalizedScore };
      })
      .filter((p: { score: number }) => p.score > 0.05)
      .sort((a: { score: number }, b: { score: number }) => b.score - a.score)
      .slice(0, limit);

    trace.push({
      name: "Text filtering & scoring",
      startTime: filterStart - startTime,
      duration: Date.now() - filterStart,
      details: {
        queryTerms,
        totalPoints: data.result?.points?.length || 0,
        matchingResults: scoredResults.length,
        mode: "text-fallback",
      },
    });

    const totalDuration = Date.now() - startTime;

    return NextResponse.json({
      results: scoredResults,
      trace,
      metadata: {
        query,
        collection: QDRANT_COLLECTION_NAME,
        mode: "text-fallback",
        limit,
        totalLatency: totalDuration,
        resultsCount: scoredResults.length,
      },
    });
  } catch (error) {
    console.error("Search error:", error);
    return NextResponse.json(
      {
        error: "Search failed",
        trace: [],
        metadata: {
          totalLatency: Date.now() - startTime,
        },
      },
      { status: 500 }
    );
  }
}
