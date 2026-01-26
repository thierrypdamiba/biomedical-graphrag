"use client";

import * as React from "react";
import {
  Play,
  Copy,
  Save,
  Trash2,
  Clock,
  CheckCircle,
  XCircle,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Chip } from "@/components/ui/chip";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn, formatLatency } from "@/lib/utils";

const sampleHistory = [
  {
    id: "req-001",
    method: "POST",
    path: "/collections/products/points/query",
    status: 200,
    latency: 42,
    timestamp: new Date(Date.now() - 120000),
    body: '{"query": "machine learning", "limit": 10}',
    response: '{"result": [{"id": "prod-001", "score": 0.92}]}',
  },
  {
    id: "req-002",
    method: "GET",
    path: "/collections",
    status: 200,
    latency: 15,
    timestamp: new Date(Date.now() - 300000),
    response:
      '{"result": {"collections": [{"name": "products"}, {"name": "documents"}]}}',
  },
  {
    id: "req-003",
    method: "POST",
    path: "/collections/products/points/search",
    status: 400,
    latency: 8,
    timestamp: new Date(Date.now() - 600000),
    body: '{"vector": [0.1, 0.2]}',
    response: '{"status": {"error": "Vector dimension mismatch"}}',
  },
];

const defaultRequest = `{
  "query": "machine learning products",
  "limit": 10,
  "with_payload": true,
  "filter": {
    "must": [
      {
        "key": "category",
        "match": {
          "value": "software"
        }
      }
    ]
  }
}`;

export default function ConsolePage() {
  const [selectedRequest, setSelectedRequest] = React.useState(sampleHistory[0]);
  const [method, setMethod] = React.useState("POST");
  const [path, setPath] = React.useState("/collections/products/points/query");
  const [body, setBody] = React.useState(defaultRequest);
  const [response, setResponse] = React.useState<string | null>(null);
  const [responseTab, setResponseTab] = React.useState<"json" | "raw">("json");
  const [isRunning, setIsRunning] = React.useState(false);

  const handleRun = async () => {
    setIsRunning(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 500));
    setResponse(
      JSON.stringify(
        {
          result: [
            { id: "prod-001", score: 0.92, payload: { name: "TensorFlow Pro" } },
            { id: "prod-002", score: 0.89, payload: { name: "AI Vision SDK" } },
          ],
          status: "ok",
          time: 0.042,
        },
        null,
        2
      )
    );
    setIsRunning(false);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="flex h-[calc(100vh-56px-48px)] gap-4">
      {/* Left Sidebar - History */}
      <div className="w-72 shrink-0 overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)]">
        <div className="flex items-center justify-between border-b border-[var(--stroke-1)] p-3">
          <h3 className="font-semibold">History</h3>
          <Button variant="ghost" size="sm" className="text-[var(--text-tertiary)]">
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
        <div className="overflow-y-auto p-2">
          {sampleHistory.map((req) => (
            <button
              key={req.id}
              onClick={() => {
                setSelectedRequest(req);
                setMethod(req.method);
                setPath(req.path);
                setBody(req.body || "");
                setResponse(req.response || null);
              }}
              className={cn(
                "flex w-full flex-col gap-1 rounded-lg px-3 py-2 text-left transition-colors",
                selectedRequest?.id === req.id
                  ? "bg-[var(--bg-2)]"
                  : "hover:bg-[var(--bg-2)]"
              )}
            >
              <div className="flex items-center gap-2">
                <Chip
                  variant={
                    req.method === "GET"
                      ? "blue"
                      : req.method === "POST"
                      ? "teal"
                      : "default"
                  }
                  className="text-[10px] px-1.5 py-0"
                >
                  {req.method}
                </Chip>
                {req.status >= 200 && req.status < 300 ? (
                  <CheckCircle className="h-3 w-3 text-[var(--teal)]" />
                ) : (
                  <XCircle className="h-3 w-3 text-[var(--amaranth)]" />
                )}
                <span className="text-xs text-[var(--text-tertiary)]">
                  {req.status}
                </span>
              </div>
              <div className="truncate text-sm">{req.path}</div>
              <div className="flex items-center gap-2 text-xs text-[var(--text-tertiary)]">
                <Clock className="h-3 w-3" />
                {formatLatency(req.latency)}
                <span>•</span>
                {req.timestamp.toLocaleTimeString()}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Editor */}
      <div className="flex flex-1 flex-col gap-4">
        {/* Request Section */}
        <div className="flex-1 overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)]">
          <div className="flex items-center gap-2 border-b border-[var(--stroke-1)] p-3">
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className="rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-2)] px-2 py-1.5 text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--blue)]"
            >
              <option value="GET">GET</option>
              <option value="POST">POST</option>
              <option value="PUT">PUT</option>
              <option value="DELETE">DELETE</option>
              <option value="PATCH">PATCH</option>
            </select>
            <input
              type="text"
              value={path}
              onChange={(e) => setPath(e.target.value)}
              className="flex-1 rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)] px-3 py-1.5 text-sm font-mono focus:border-[var(--stroke-2)] focus:outline-none focus:ring-2 focus:ring-[var(--blue)]"
            />
            <Button
              variant="primary"
              onClick={handleRun}
              disabled={isRunning}
              className="gap-2"
            >
              {isRunning ? (
                <>
                  <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Run
                </>
              )}
            </Button>
            <Button variant="secondary" onClick={() => copyToClipboard(body)}>
              <Copy className="h-4 w-4" />
            </Button>
            <Button variant="secondary">
              <Save className="h-4 w-4" />
            </Button>
          </div>
          <div className="h-[calc(100%-52px)] overflow-auto p-0">
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              className="h-full w-full resize-none bg-[#0B1220] p-4 font-mono text-sm text-[var(--text-primary)] focus:outline-none"
              placeholder="Request body (JSON)..."
              spellCheck={false}
            />
          </div>
        </div>

        {/* Response Section */}
        <div className="flex-1 overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)]">
          <div className="flex items-center justify-between border-b border-[var(--stroke-1)] p-3">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold">Response</h3>
              {response && (
                <Chip variant="teal">200 OK</Chip>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Tabs
                value={responseTab}
                onValueChange={(v) => setResponseTab(v as "json" | "raw")}
              >
                <TabsList className="h-8">
                  <TabsTrigger value="json" className="text-xs px-2 py-1">
                    JSON
                  </TabsTrigger>
                  <TabsTrigger value="raw" className="text-xs px-2 py-1">
                    Raw
                  </TabsTrigger>
                </TabsList>
              </Tabs>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => response && copyToClipboard(response)}
                disabled={!response}
              >
                <Copy className="h-4 w-4" />
              </Button>
            </div>
          </div>
          <div className="h-[calc(100%-52px)] overflow-auto">
            {response ? (
              <pre className="p-4 font-mono text-sm text-[var(--text-primary)]">
                {response}
              </pre>
            ) : (
              <div className="flex h-full items-center justify-center text-[var(--text-tertiary)]">
                Run a request to see the response
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
