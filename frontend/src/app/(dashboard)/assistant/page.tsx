"use client";

import * as React from "react";
import {
  Send,
  Paperclip,
  FileText,
  Clock,
  Database,
  Zap,
  Code,
  Pin,
  ChevronDown,
  ChevronUp,
  Copy,
  ExternalLink,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Chip } from "@/components/ui/chip";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DetailedTracePanel } from "@/components/ui/detailed-trace-panel";
import { useAppStore } from "@/store/app-store";
import { cn, formatLatency } from "@/lib/utils";

interface TraceStep {
  name: string;
  startTime: number;
  duration?: number;
  details?: Record<string, unknown>;
}

interface SearchResult {
  id: string | number;
  score: number;
  payload: {
    paper?: {
      pmid?: string;
      title?: string;
      abstract?: string;
      authors?: Array<{ name?: string }>;
      journal?: string;
      publication_date?: string;
      mesh_terms?: Array<{ term?: string }>;
    };
  };
}

interface MessageMetadata {
  collection: string;
  filter?: string;
  vectorMode: "dense" | "sparse" | "hybrid";
  latency: number;
  results: SearchResult[];
  trace?: TraceStep[];
  query?: string;
  qdrant_results?: Record<string, unknown[]>;
  neo4j_results?: Record<string, unknown>;
  toolsUsed?: {
    qdrant?: string;
    neo4j?: string[];
  };
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  metadata?: MessageMetadata;
}

export default function AssistantPage() {
  const {
    vectorMode,
    setVectorMode,
    topK,
    setTopK,
    activeCollection,
    artifactsPaneOpen,
    toggleArtifactsPane,
    activeArtifactTab,
    setActiveArtifactTab,
    traceDrawerOpen,
    toggleTraceDrawer,
    selectedPoint,
    setSelectedPoint,
    sidebarCollapsed,
    pendingSearch,
    setPendingSearch,
  } = useAppStore();

  const [messages, setMessages] = React.useState<Message[]>([]);
  const [input, setInput] = React.useState("");
  const [isLoading, setIsLoading] = React.useState(false);
  const [selectedTraceStep, setSelectedTraceStep] = React.useState<TraceStep | null>(null);
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle search from top bar
  React.useEffect(() => {
    if (pendingSearch && !isLoading) {
      executeSearch(pendingSearch);
      setPendingSearch(null);
    }
  }, [pendingSearch]);

  const executeSearch = async (query: string) => {
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          limit: topK,
          mode: vectorMode,
        }),
      });

      const data = await response.json();

      let content = "";

      // If GraphRAG returned a summary, use it
      if (data.summary) {
        content = data.summary;
      } else if (data.results && data.results.length > 0) {
        // Fallback: build content from results
        content = `Found ${data.results.length} relevant papers for "${query}":\n\n`;
        data.results.slice(0, 5).forEach((result: SearchResult, i: number) => {
          const paper = result.payload?.paper;
          const pmid = paper?.pmid || String(result.id);
          const title = paper?.title || "Untitled";
          const authors = paper?.authors?.slice(0, 2).map((a: { name?: string }) => a.name).join(", ");
          const journal = paper?.journal || "";
          content += `${i + 1}. **PMID:${pmid}** - ${title}${authors ? ` (${authors})` : ""}${journal ? ` - ${journal}` : ""}\n\n`;
        });
      } else {
        content = `No results found for "${query}". Try different search terms.`;
      }

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content,
        timestamp: new Date(),
        metadata: {
          collection: data.metadata?.collection || activeCollection || "biomedical_papers",
          vectorMode: data.metadata?.mode || vectorMode,
          latency: data.metadata?.totalLatency || 0,
          results: data.results || [],
          trace: data.trace || [],
          query,
          qdrant_results: data.metadata?.qdrant_results,
          neo4j_results: data.metadata?.neo4j_results,
          toolsUsed: data.metadata?.toolsUsed,
        },
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Search error:", error);
      const errorMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Sorry, there was an error processing your search. Please try again.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const query = input;
    setInput("");
    await executeSearch(query);
  };

  const lastAssistantMessage = [...messages]
    .reverse()
    .find((m) => m.role === "assistant" && m.metadata);

  return (
    <div className="flex flex-col lg:flex-row h-[calc(100vh-56px-32px-56px)] md:h-[calc(100vh-56px-48px-56px)] gap-4">
      {/* Chat Pane */}
      <div className="flex flex-1 flex-col min-w-0">
        {/* Header */}
        <div className="mb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="min-w-0">
            <h1 className="text-lg md:text-xl font-semibold">Assistant</h1>
            <p className="text-xs md:text-sm text-[var(--text-secondary)] truncate">
              Collection: {activeCollection || "pubmed_papers"} | Mode: {vectorMode} |
              top_k: {topK}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleArtifactsPane}
              className="gap-2 text-xs md:text-sm"
            >
              <span className="hidden sm:inline">{artifactsPaneOpen ? "Hide" : "Show"} Artifacts</span>
              <span className="sm:hidden">{artifactsPaneOpen ? "Hide" : "Show"}</span>
              {artifactsPaneOpen ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto space-y-4 pb-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "animate-fade-in",
                message.role === "user"
                  ? "border-l-2 border-[var(--stroke-2)] pl-4"
                  : "rounded-lg bg-[var(--bg-1)] p-4"
              )}
            >
              <div className="mb-1 flex items-center gap-2 text-xs text-[var(--text-tertiary)]">
                <span className="font-medium">
                  {message.role === "user" ? "You" : "Assistant"}
                </span>
                <span>•</span>
                <span>{message.timestamp.toLocaleTimeString()}</span>
              </div>
              <div className="whitespace-pre-wrap text-[var(--text-primary)]">
                {message.content}
              </div>
              {message.role === "assistant" && message.metadata && (
                <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-[var(--stroke-1)] pt-3">
                  <Chip variant="default">
                    <Database className="h-3 w-3" />
                    {message.metadata.collection}
                  </Chip>
                  <Chip variant="violet">
                    <Zap className="h-3 w-3" />
                    {message.metadata.vectorMode}
                  </Chip>
                  <Chip variant="teal">
                    <Clock className="h-3 w-3" />
                    {formatLatency(message.metadata.latency || 0)}
                  </Chip>
                  <div className="flex-1" />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setActiveArtifactTab("code");
                      if (!artifactsPaneOpen) toggleArtifactsPane();
                    }}
                  >
                    <Code className="h-4 w-4 mr-1" />
                    Show code
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setActiveArtifactTab("results");
                      if (!artifactsPaneOpen) toggleArtifactsPane();
                    }}
                  >
                    <ExternalLink className="h-4 w-4 mr-1" />
                    Open results
                  </Button>
                  <Button variant="ghost" size="sm">
                    <Pin className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="rounded-lg bg-[var(--bg-1)] p-4">
              <div className="flex items-center gap-2 text-[var(--text-secondary)]">
                <div className="h-2 w-2 animate-pulse rounded-full bg-[var(--violet)]" />
                Running...
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Composer */}
        <div className="border-t border-[var(--stroke-1)] pt-3 md:pt-4">
          <div className="flex items-end gap-2">
            <div className="flex-1">
              <div className="flex flex-wrap md:flex-nowrap items-center gap-2 rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)] p-2">
                <div className="hidden md:flex gap-1">
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <Paperclip className="h-4 w-4" />
                  </Button>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <FileText className="h-4 w-4" />
                  </Button>
                </div>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask about genes, diseases..."
                  className="flex-1 min-w-0 bg-transparent text-sm md:text-base text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] focus:outline-none"
                />
                <div className="flex items-center gap-1 md:gap-2">
                  <Select
                    value={vectorMode}
                    onValueChange={(v) => setVectorMode(v as typeof vectorMode)}
                  >
                    <SelectTrigger className="h-8 w-20 md:w-28 border-0 bg-[var(--bg-2)] text-xs md:text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="graphrag">GraphRAG</SelectItem>
                      <SelectItem value="dense">Dense</SelectItem>
                      <SelectItem value="sparse">Sparse</SelectItem>
                      <SelectItem value="hybrid">Hybrid</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select
                    value={topK.toString()}
                    onValueChange={(v) => setTopK(parseInt(v))}
                  >
                    <SelectTrigger className="h-8 w-14 md:w-16 border-0 bg-[var(--bg-2)] text-xs md:text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="10">10</SelectItem>
                      <SelectItem value="20">20</SelectItem>
                      <SelectItem value="50">50</SelectItem>
                    </SelectContent>
                  </Select>
                  <Button
                    variant="primary"
                    size="icon"
                    onClick={handleSend}
                    disabled={!input.trim() || isLoading}
                    className="h-8 w-8"
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
          <div className="mt-2 hidden md:flex items-center gap-4 text-xs text-[var(--text-tertiary)]">
            <span>Enter to send, Shift+Enter for new line</span>
            <span>⌘/ to toggle trace</span>
          </div>
        </div>
      </div>

      {/* Artifacts Pane */}
      {artifactsPaneOpen && (
        <div className="w-full lg:w-[420px] xl:w-[520px] shrink-0 overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)] max-h-[50vh] lg:max-h-none">
          <Tabs
            value={activeArtifactTab}
            onValueChange={(v) => setActiveArtifactTab(v as typeof activeArtifactTab)}
            className="h-full flex flex-col"
          >
            <TabsList className="px-4 pt-4">
              <TabsTrigger value="results">Results</TabsTrigger>
              <TabsTrigger value="point">Point</TabsTrigger>
              <TabsTrigger value="code">Code</TabsTrigger>
              <TabsTrigger value="trace">Trace</TabsTrigger>
            </TabsList>

            <TabsContent value="results" className="flex-1 overflow-y-auto p-4">
              {lastAssistantMessage?.metadata?.results && lastAssistantMessage.metadata.results.length > 0 ? (
                <div className="space-y-2">
                  <div className="mb-4 flex items-center justify-between">
                    <span className="text-sm text-[var(--text-secondary)]">
                      {lastAssistantMessage.metadata.results.length} results
                    </span>
                    <div className="flex gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          navigator.clipboard.writeText(
                            JSON.stringify(lastAssistantMessage.metadata?.results, null, 2)
                          );
                        }}
                      >
                        <Copy className="h-4 w-4 mr-1" />
                        Export JSON
                      </Button>
                    </div>
                  </div>
                  {lastAssistantMessage.metadata.results.map((result) => {
                    const paper = result.payload?.paper;
                    const pmid = paper?.pmid || String(result.id);
                    const title = paper?.title || "Untitled";
                    const authors = paper?.authors?.slice(0, 2).map((a: { name?: string }) => a.name).join(", ");
                    const journal = paper?.journal;

                    return (
                      <div
                        key={String(result.id)}
                        className={cn(
                          "cursor-pointer rounded-lg border border-[var(--stroke-1)] p-3 transition-colors hover:bg-[var(--bg-2)]",
                          selectedPoint?.id === result.id &&
                            "bg-[var(--selection)] border-[var(--blue)]"
                        )}
                        onClick={() => {
                          setSelectedPoint(result);
                          setActiveArtifactTab("point");
                        }}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <code className="text-xs text-[var(--text-tertiary)]">
                            PMID:{pmid}
                          </code>
                          <Chip variant="blue">{result.score.toFixed(3)}</Chip>
                        </div>
                        <p className="text-sm font-medium line-clamp-2">{title}</p>
                        {authors && (
                          <p className="text-xs text-[var(--text-tertiary)] mt-1">
                            {authors}
                          </p>
                        )}
                        {journal && (
                          <p className="text-xs text-[var(--text-tertiary)]">
                            {journal}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-[var(--text-tertiary)]">
                  No results yet. Ask a question to see results.
                </div>
              )}
            </TabsContent>

            <TabsContent value="point" className="flex-1 overflow-y-auto p-4">
              {selectedPoint ? (
                <div className="space-y-4">
                  {(() => {
                    const paper = (selectedPoint.payload as { paper?: Record<string, unknown> })?.paper || selectedPoint.payload;
                    const pmid = paper?.pmid || selectedPoint.id;
                    const title = paper?.title as string;
                    const abstract = paper?.abstract as string;
                    const authors = paper?.authors as Array<{ name?: string }>;
                    const journal = paper?.journal as string;
                    const pubDate = paper?.publication_date as string;
                    const meshTerms = paper?.mesh_terms as Array<{ term?: string }>;

                    return (
                      <>
                        <div className="flex items-center justify-between">
                          <code className="text-sm text-[var(--violet)]">
                            PMID:{pmid}
                          </code>
                          <div className="flex gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => navigator.clipboard.writeText(JSON.stringify(selectedPoint, null, 2))}
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                            {pmid && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => window.open(`https://pubmed.ncbi.nlm.nih.gov/${pmid}`, "_blank")}
                              >
                                <ExternalLink className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </div>

                        <div>
                          <h3 className="mb-1 text-sm font-semibold">Score</h3>
                          <Chip variant="blue">{selectedPoint.score.toFixed(4)}</Chip>
                        </div>

                        {title && (
                          <div>
                            <h3 className="mb-1 text-sm font-semibold">Title</h3>
                            <p className="text-sm">{title}</p>
                          </div>
                        )}

                        {authors && authors.length > 0 && (
                          <div>
                            <h3 className="mb-1 text-sm font-semibold">Authors</h3>
                            <p className="text-sm text-[var(--text-secondary)]">
                              {authors.map(a => a.name).join(", ")}
                            </p>
                          </div>
                        )}

                        {journal && (
                          <div>
                            <h3 className="mb-1 text-sm font-semibold">Journal</h3>
                            <p className="text-sm text-[var(--text-secondary)]">{journal}</p>
                          </div>
                        )}

                        {pubDate && (
                          <div>
                            <h3 className="mb-1 text-sm font-semibold">Publication Date</h3>
                            <p className="text-sm text-[var(--text-secondary)]">{pubDate}</p>
                          </div>
                        )}

                        {abstract && (
                          <div>
                            <h3 className="mb-1 text-sm font-semibold">Abstract</h3>
                            <p className="text-sm text-[var(--text-secondary)] max-h-40 overflow-y-auto">
                              {abstract}
                            </p>
                          </div>
                        )}

                        {meshTerms && meshTerms.length > 0 && (
                          <div>
                            <h3 className="mb-2 text-sm font-semibold">MeSH Terms</h3>
                            <div className="flex flex-wrap gap-1">
                              {meshTerms.map((m, i) => (
                                <Chip key={i} variant="default" className="text-[10px]">
                                  {m.term}
                                </Chip>
                              ))}
                            </div>
                          </div>
                        )}

                        <div>
                          <h3 className="mb-2 text-sm font-semibold">Raw Payload</h3>
                          <pre className="rounded-lg bg-[#0B1220] p-4 text-xs overflow-x-auto max-h-48">
                            {JSON.stringify(selectedPoint.payload, null, 2)}
                          </pre>
                        </div>
                      </>
                    );
                  })()}
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-[var(--text-tertiary)]">
                  Select a result to view details
                </div>
              )}
            </TabsContent>

            <TabsContent value="code" className="flex-1 overflow-y-auto p-4">
              {lastAssistantMessage?.metadata?.query ? (
                <div className="space-y-4">
                  {/* Python */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Chip variant="blue">Python</Chip>
                      </h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const code = `from qdrant_client import QdrantClient

client = QdrantClient(url="YOUR_QDRANT_URL", api_key="YOUR_API_KEY")

results = client.query_points(
    collection_name="${lastAssistantMessage.metadata?.collection || "biomedical_papers"}",
    query="${lastAssistantMessage.metadata.query}",
    limit=${topK},
    with_payload=True
)

for point in results.points:
    print(f"ID: {point.id}, Score: {point.score}")
    print(f"Title: {point.payload.get('paper', {}).get('title')}")`;
                          navigator.clipboard.writeText(code);
                        }}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                    <pre className="rounded-lg bg-[#0B1220] p-4 text-xs overflow-x-auto">
{`from qdrant_client import QdrantClient

client = QdrantClient(url="YOUR_QDRANT_URL", api_key="YOUR_API_KEY")

results = client.query_points(
    collection_name="${lastAssistantMessage.metadata?.collection || "biomedical_papers"}",
    query="${lastAssistantMessage.metadata.query}",
    limit=${topK},
    with_payload=True
)

for point in results.points:
    print(f"ID: {point.id}, Score: {point.score}")`}
                    </pre>
                  </div>

                  {/* curl */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Chip variant="teal">curl</Chip>
                      </h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const code = `curl -X POST 'YOUR_QDRANT_URL/collections/${lastAssistantMessage.metadata?.collection || "biomedical_papers"}/points/query' \\
  -H 'api-key: YOUR_API_KEY' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "query": "${lastAssistantMessage.metadata.query}",
    "limit": ${topK},
    "with_payload": true
  }'`;
                          navigator.clipboard.writeText(code);
                        }}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                    <pre className="rounded-lg bg-[#0B1220] p-4 text-xs overflow-x-auto">
{`curl -X POST 'YOUR_QDRANT_URL/collections/${lastAssistantMessage.metadata?.collection || "biomedical_papers"}/points/query' \\
  -H 'api-key: YOUR_API_KEY' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "query": "${lastAssistantMessage.metadata.query}",
    "limit": ${topK},
    "with_payload": true
  }'`}
                    </pre>
                  </div>

                  {/* JavaScript */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-sm font-semibold flex items-center gap-2">
                        <Chip variant="violet">JavaScript</Chip>
                      </h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          const code = `import { QdrantClient } from '@qdrant/js-client-rest';

const client = new QdrantClient({
  url: 'YOUR_QDRANT_URL',
  apiKey: 'YOUR_API_KEY',
});

const results = await client.query('${lastAssistantMessage.metadata?.collection || "biomedical_papers"}', {
  query: '${lastAssistantMessage.metadata.query}',
  limit: ${topK},
  with_payload: true,
});

console.log(results);`;
                          navigator.clipboard.writeText(code);
                        }}
                      >
                        <Copy className="h-4 w-4" />
                      </Button>
                    </div>
                    <pre className="rounded-lg bg-[#0B1220] p-4 text-xs overflow-x-auto">
{`import { QdrantClient } from '@qdrant/js-client-rest';

const client = new QdrantClient({
  url: 'YOUR_QDRANT_URL',
  apiKey: 'YOUR_API_KEY',
});

const results = await client.query('${lastAssistantMessage.metadata?.collection || "biomedical_papers"}', {
  query: '${lastAssistantMessage.metadata.query}',
  limit: ${topK},
  with_payload: true,
});`}
                    </pre>
                  </div>
                </div>
              ) : (
                <div className="flex h-full items-center justify-center text-[var(--text-tertiary)]">
                  Run a search to see code examples
                </div>
              )}
            </TabsContent>

            <TabsContent value="trace" className="flex-1 overflow-y-auto p-4">
              <DetailedTracePanel
                trace={lastAssistantMessage?.metadata?.trace || []}
                qdrantResults={lastAssistantMessage?.metadata?.qdrant_results as Record<string, Array<{
                  id: number | string;
                  score?: number;
                  payload?: {
                    paper?: {
                      pmid?: string;
                      title?: string;
                      abstract?: string;
                      authors?: Array<{ name?: string }>;
                      journal?: string;
                      publication_date?: string;
                      mesh_terms?: Array<{ term?: string }>;
                    };
                  };
                }>>}
                neo4jResults={lastAssistantMessage?.metadata?.neo4j_results as Record<string, unknown>}
                totalLatency={lastAssistantMessage?.metadata?.latency}
                toolsUsed={lastAssistantMessage?.metadata?.toolsUsed}
              />
            </TabsContent>
          </Tabs>
        </div>
      )}

      {/* Bottom Trace Drawer */}
      <div
        className={cn(
          "fixed bottom-0 right-0 border-t border-[var(--stroke-1)] bg-[var(--bg-1)] transition-all duration-300",
          traceDrawerOpen ? "h-[280px] md:h-[360px]" : "h-10 md:h-11",
          // Mobile: full width
          "left-0",
          // Desktop: account for sidebar
          sidebarCollapsed ? "lg:left-[72px]" : "lg:left-[260px]"
        )}
      >
        <button
          onClick={toggleTraceDrawer}
          className="flex h-10 md:h-11 w-full items-center justify-between px-3 md:px-4 hover:bg-[var(--bg-2)]"
        >
          <div className="flex items-center gap-2 md:gap-4 text-xs md:text-sm overflow-x-auto">
            <span className="text-[var(--text-secondary)] hidden sm:inline">Total latency:</span>
            <Chip variant="teal" className="text-xs">
              {lastAssistantMessage?.metadata?.latency || 0}ms
            </Chip>
            <span className="text-[var(--text-secondary)] hidden sm:inline">Steps:</span>
            <span className="text-[var(--text-secondary)] sm:hidden">S:</span>
            <Chip variant="default" className="text-xs">
              {lastAssistantMessage?.metadata?.trace?.length || 0}
            </Chip>
            <span className="text-[var(--text-secondary)] hidden sm:inline">Results:</span>
            <span className="text-[var(--text-secondary)] sm:hidden">R:</span>
            <Chip variant="default" className="text-xs">
              {lastAssistantMessage?.metadata?.results?.length || 0}
            </Chip>
          </div>
          {traceDrawerOpen ? (
            <ChevronDown className="h-4 w-4 shrink-0" />
          ) : (
            <ChevronUp className="h-4 w-4 shrink-0" />
          )}
        </button>
        {traceDrawerOpen && (
          <div className="h-[calc(280px-40px)] md:h-[calc(360px-44px)] overflow-y-auto p-3 md:p-4">
            {lastAssistantMessage?.metadata?.trace && lastAssistantMessage.metadata.trace.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-2">
                  <h3 className="mb-2 md:mb-3 text-sm font-semibold">Timeline</h3>
                  <div className="space-y-2">
                    {lastAssistantMessage.metadata.trace.map((step, i) => {
                      const totalLatency = lastAssistantMessage.metadata?.latency || 1;
                      const widthPercent = Math.max(5, ((step.duration || 0) / totalLatency) * 100);
                      const colors = [
                        "bg-[var(--blue)]",
                        "bg-[var(--violet)]",
                        "bg-[var(--teal)]",
                        "bg-[var(--amaranth)]",
                      ];
                      return (
                        <div
                          key={i}
                          className={cn(
                            "flex items-center gap-2 cursor-pointer rounded p-1 -m-1 transition-colors hover:bg-[var(--bg-2)]",
                            selectedTraceStep?.name === step.name && "bg-[var(--selection)]"
                          )}
                          onClick={() => setSelectedTraceStep(selectedTraceStep?.name === step.name ? null : step)}
                        >
                          <span className="w-36 text-xs text-[var(--text-secondary)] truncate">
                            {step.name}
                          </span>
                          <div className="flex-1 h-5 rounded bg-[var(--bg-2)] relative">
                            <div
                              className={cn("h-full rounded flex items-center justify-end pr-2", colors[i % colors.length])}
                              style={{ width: `${widthPercent}%` }}
                            >
                              <span className="text-[10px] text-white font-medium">
                                {step.duration}ms
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div>
                  <h3 className="mb-3 text-sm font-semibold">
                    {selectedTraceStep ? "Step Details" : "Raw Events"}
                  </h3>
                  {selectedTraceStep ? (
                    <div className="space-y-3">
                      <div>
                        <span className="text-xs text-[var(--text-tertiary)]">Name</span>
                        <p className="text-sm font-medium">{selectedTraceStep.name}</p>
                      </div>
                      <div>
                        <span className="text-xs text-[var(--text-tertiary)]">Duration</span>
                        <p className="text-sm">{selectedTraceStep.duration}ms</p>
                      </div>
                      <div>
                        <span className="text-xs text-[var(--text-tertiary)]">Start Time</span>
                        <p className="text-sm">{selectedTraceStep.startTime}ms</p>
                      </div>
                      {selectedTraceStep.details && (
                        <div>
                          <span className="text-xs text-[var(--text-tertiary)]">Details</span>
                          <pre className="mt-1 rounded bg-[#0B1220] p-2 text-[10px] overflow-x-auto max-h-32">
                            {JSON.stringify(selectedTraceStep.details, null, 2)}
                          </pre>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-1 text-xs font-mono text-[var(--text-secondary)]">
                      {lastAssistantMessage.metadata.trace.map((step, i) => (
                        <div key={i}>
                          {String(step.startTime).padStart(5, "0")}ms {step.name.toLowerCase().replace(/\s+/g, "_")}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex h-full items-center justify-center text-[var(--text-tertiary)]">
                No trace data. Run a search to see the execution timeline.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
