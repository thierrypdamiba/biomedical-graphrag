import { NextResponse } from "next/server";

const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;

export async function POST(
  request: Request,
  { params }: { params: Promise<{ name: string }> }
) {
  try {
    const { name } = await params;
    const body = await request.json();

    const response = await fetch(`${QDRANT_URL}/collections/${name}/points/scroll`, {
      method: "POST",
      headers: {
        "api-key": QDRANT_API_KEY || "",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        limit: body.limit || 20,
        offset: body.offset || null,
        with_payload: true,
        with_vector: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Qdrant API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error fetching points:", error);
    return NextResponse.json(
      { error: "Failed to fetch points" },
      { status: 500 }
    );
  }
}
