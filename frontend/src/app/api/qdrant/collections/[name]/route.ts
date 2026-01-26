import { NextResponse } from "next/server";

const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

export async function GET(
  request: Request,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params;

    const response = await fetch(`${QDRANT_URL}/collections/${name}`, {
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
    console.error("Error fetching collection:", error);
    return NextResponse.json(
      { error: "Failed to fetch collection" },
      { status: 500 }
    );
  }
}
