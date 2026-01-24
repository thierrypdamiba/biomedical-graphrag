import { NextResponse } from "next/server";

const GRAPHRAG_API_URL = process.env.GRAPHRAG_API_URL || "http://localhost:8765";

export async function GET() {
  try {
    // Call the backend GraphRAG API for Neo4j stats
    const response = await fetch(`${GRAPHRAG_API_URL}/api/neo4j/stats`, {
      headers: {
        "Content-Type": "application/json",
      },
      // Add timeout
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error(`Backend API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching Neo4j stats:", error);
    return NextResponse.json(
      { error: "Failed to fetch Neo4j stats. Make sure the GraphRAG API server is running." },
      { status: 500 }
    );
  }
}
