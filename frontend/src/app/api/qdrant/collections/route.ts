import { NextResponse } from "next/server";

const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

export async function GET() {
  try {
    const response = await fetch(`${QDRANT_URL}/collections`, {
      headers: {
        "api-key": QDRANT_API_KEY || "",
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      throw new Error(`Qdrant API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching collections:", error);
    return NextResponse.json(
      { error: "Failed to fetch collections" },
      { status: 500 }
    );
  }
}
