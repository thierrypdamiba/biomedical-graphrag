import { NextRequest } from "next/server";

const QDRANT_URL = process.env.QDRANT_URL;
const QDRANT_API_KEY = process.env.QDRANT_API_KEY;
const QDRANT_COLLECTION_NAME = process.env.QDRANT_COLLECTION_NAME || "biomedical_papers";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

interface TraceStep {
  id: string;
  name: string;
  status: "pending" | "running" | "complete" | "error";
  startTime?: number;
  endTime?: number;
  duration?: number;
  details?: Record<string, unknown>;
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const { query, limit = 20 } = body;

  const encoder = new TextEncoder();
  const startTime = Date.now();

  const stream = new ReadableStream({
    async start(controller) {
      const sendEvent = (event: string, data: unknown) => {
        controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
      };

      const steps: TraceStep[] = [
        { id: "embedding", name: "Generating embedding", status: "pending" },
        { id: "qdrant", name: "Searching Qdrant", status: "pending" },
        { id: "format", name: "Formatting results", status: "pending" },
      ];

      // Send initial state
      sendEvent("trace", { steps, phase: "start" });

      try {
        // Step 1: Generate embedding
        steps[0].status = "running";
        steps[0].startTime = Date.now() - startTime;
        sendEvent("trace", { steps, phase: "embedding" });

        const embeddingResponse = await fetch("https://api.openai.com/v1/embeddings", {
          method: "POST",
          headers: {
            "Authorization": `Bearer ${OPENAI_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: "text-embedding-3-large",
            input: query,
            dimensions: 1536,
          }),
        });

        if (!embeddingResponse.ok) {
          throw new Error("Failed to generate embedding");
        }

        const embeddingData = await embeddingResponse.json();
        const queryVector = embeddingData.data[0].embedding;

        steps[0].status = "complete";
        steps[0].endTime = Date.now() - startTime;
        steps[0].duration = steps[0].endTime - (steps[0].startTime || 0);
        steps[0].details = { model: "text-embedding-3-large", dimensions: 1536 };
        sendEvent("trace", { steps, phase: "embedding-complete" });

        // Step 2: Search Qdrant
        steps[1].status = "running";
        steps[1].startTime = Date.now() - startTime;
        sendEvent("trace", { steps, phase: "qdrant" });

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
          throw new Error("Qdrant search failed");
        }

        const searchData = await searchResponse.json();
        
        steps[1].status = "complete";
        steps[1].endTime = Date.now() - startTime;
        steps[1].duration = steps[1].endTime - (steps[1].startTime || 0);
        steps[1].details = { 
          collection: QDRANT_COLLECTION_NAME, 
          resultsFound: searchData.result?.length || 0 
        };
        sendEvent("trace", { steps, phase: "qdrant-complete" });

        // Step 3: Format results
        steps[2].status = "running";
        steps[2].startTime = Date.now() - startTime;
        sendEvent("trace", { steps, phase: "format" });

        const results = (searchData.result || []).map((hit: { id: number; score: number; payload: Record<string, unknown> }) => ({
          id: hit.id,
          score: hit.score,
          payload: hit.payload,
        }));

        steps[2].status = "complete";
        steps[2].endTime = Date.now() - startTime;
        steps[2].duration = steps[2].endTime - (steps[2].startTime || 0);
        steps[2].details = { resultsFormatted: results.length };
        sendEvent("trace", { steps, phase: "complete" });

        // Send final results
        sendEvent("results", {
          results,
          metadata: {
            query,
            collection: QDRANT_COLLECTION_NAME,
            totalLatency: Date.now() - startTime,
            resultsCount: results.length,
          },
        });

        sendEvent("done", { success: true });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : "Unknown error";
        
        // Mark current running step as error
        const runningStep = steps.find(s => s.status === "running");
        if (runningStep) {
          runningStep.status = "error";
          runningStep.details = { error: errorMessage };
        }
        
        sendEvent("trace", { steps, phase: "error" });
        sendEvent("error", { message: errorMessage });
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
