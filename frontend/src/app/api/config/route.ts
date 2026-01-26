import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    qdrantCollectionName: process.env.QDRANT_COLLECTION_NAME || "biomedical_papers",
  });
}
