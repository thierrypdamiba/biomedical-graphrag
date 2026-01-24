"use client";

import * as React from "react";
import {
  Database,
  RefreshCw,
  FileText,
  AlertCircle,
  Search,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Chip } from "@/components/ui/chip";
import { SearchTimeline } from "@/components/ui/search-timeline";
import { cn, formatNumber } from "@/lib/utils";

interface TraceStep {
  id: string;
  name: string;
  status: "pending" | "running" | "complete" | "error";
  startTime?: number;
  endTime?: number;
  duration?: number;
  details?: Record<string, unknown>;
}

interface CollectionInfo {
  status: string;
  points_count: number;
  indexed_vectors_count: number;
  config: {
    params: {
      vectors: Record<string, { size: number; distance: string }>;
      sparse_vectors?: Record<string, unknown>;
    };
  };
  payload_schema?: Record<string, { data_type: string }>;
}

interface Point {
  id: string | number;
  payload: Record<string, unknown>;
  score?: number;
}

export default function DataPage() {
  const [collectionName, setCollectionName] = React.useState<string | null>(null);
  const [collectionInfo, setCollectionInfo] = React.useState<CollectionInfo | null>(null);
  const [points, setPoints] = React.useState<Point[]>([]);
  const [searchResults, setSearchResults] = React.useState<Point[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [loadingPoints, setLoadingPoints] = React.useState(false);
  const [searching, setSearching] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [selectedPoint, setSelectedPoint] = React.useState<Point | null>(null);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [isSearchMode, setIsSearchMode] = React.useState(false);
  const [traceSteps, setTraceSteps] = React.useState<TraceStep[]>([]);
  const [searchTotalTime, setSearchTotalTime] = React.useState<number | undefined>();
  const [showTimeline, setShowTimeline] = React.useState(false);

  // Fetch collection name from config
  React.useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await fetch("/api/config");
        const data = await response.json();
        setCollectionName(data.qdrantCollectionName);
      } catch {
        setCollectionName("biomedical_papers"); // fallback
      }
    };
    fetchConfig();
  }, []);

  // Fetch collection data when name is available
  React.useEffect(() => {
    if (collectionName) {
      fetchCollectionInfo();
      fetchPoints();
    }
  }, [collectionName]);

  const fetchCollectionInfo = async () => {
    if (!collectionName) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/qdrant/collections/${collectionName}`);
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setCollectionInfo(data.result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch collection info");
    } finally {
      setLoading(false);
    }
  };

  const fetchPoints = async () => {
    if (!collectionName) return;
    setLoadingPoints(true);
    try {
      const response = await fetch(`/api/qdrant/collections/${collectionName}/points`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 50 }),
      });
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setPoints(data.result?.points || []);
    } catch (err) {
      console.error("Error fetching points:", err);
    } finally {
      setLoadingPoints(false);
    }
  };

  const refresh = () => {
    fetchCollectionInfo();
    fetchPoints();
    setIsSearchMode(false);
    setSearchQuery("");
    setSearchResults([]);
    setShowTimeline(false);
    setTraceSteps([]);
    setSearchTotalTime(undefined);
  };

  // Vector search function with streaming timeline
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      setIsSearchMode(false);
      setSearchResults([]);
      setShowTimeline(false);
      return;
    }

    setSearching(true);
    setIsSearchMode(true);
    setShowTimeline(true);
    setTraceSteps([]);
    setSearchTotalTime(undefined);
    setSearchResults([]);

    try {
      // Use streaming search with real-time timeline updates
      const response = await fetch("/api/qdrant/search-stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          query: searchQuery, 
          limit: 20,
        }),
      });

      if (!response.ok) {
        throw new Error("Search request failed");
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            const eventType = line.slice(7);
            continue;
          }
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              
              // Handle trace updates
              if (data.steps) {
                setTraceSteps([...data.steps]);
              }
              
              // Handle final results
              if (data.results) {
                const results: Point[] = data.results.map((r: { id: number; score: number; payload: Record<string, unknown> }) => ({
                  id: r.id,
                  payload: r.payload,
                  score: r.score,
                }));
                setSearchResults(results);
                if (data.metadata?.totalLatency) {
                  setSearchTotalTime(data.metadata.totalLatency);
                }
              }
            } catch {
              // Ignore parse errors for incomplete chunks
            }
          }
        }
      }
    } catch (err) {
      console.error("Search error:", err);
      setSearchResults([]);
    } finally {
      setSearching(false);
    }
  };

  // Helper to get paper data from payload (nested under payload.paper)
  const getPaper = (point: Point) => {
    const paper = point.payload?.paper as Record<string, unknown> | undefined;
    return paper || point.payload; // fallback to direct payload if not nested
  };

  // Use search results if in search mode, otherwise use loaded points
  const displayPoints = isSearchMode ? searchResults : points;

  // Get vector configs
  const vectorConfigs = collectionInfo?.config?.params?.vectors || {};
  const sparseVectors = collectionInfo?.config?.params?.sparse_vectors || {};

  if (loading) {
    return (
      <div className="flex h-[calc(100vh-56px-48px)] items-center justify-center">
        <RefreshCw className="h-8 w-8 animate-spin text-[var(--text-tertiary)]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-[calc(100vh-56px-48px)] items-center justify-center">
        <Card className="max-w-md">
          <CardContent className="flex flex-col items-center gap-4 py-8">
            <AlertCircle className="h-12 w-12 text-[var(--amaranth)]" />
            <p className="text-center text-[var(--text-secondary)]">{error}</p>
            <Button variant="primary" onClick={refresh}>
              Try Again
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-[var(--violet)]">
            <Database className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-semibold">{collectionName}</h1>
            <p className="text-[var(--text-secondary)]">
              {collectionInfo && formatNumber(collectionInfo.points_count)} papers indexed
            </p>
          </div>
        </div>
        <Button variant="secondary" onClick={refresh} disabled={loading}>
          <RefreshCw className={cn("h-4 w-4 mr-2", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      {/* Stats Cards */}
      {collectionInfo && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                Total Papers
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold">
                {formatNumber(collectionInfo.points_count)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                Total Vectors
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold">
                {formatNumber(collectionInfo.indexed_vectors_count)}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                Status
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Chip variant="teal" className="text-base">
                {collectionInfo.status}
              </Chip>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                Vector Types
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-semibold">
                {Object.keys(vectorConfigs).length + Object.keys(sparseVectors).length}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Vector Configurations */}
      <Card>
        <CardHeader>
          <CardTitle>Vector Configurations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(vectorConfigs).map(([name, config]) => (
              <div
                key={name}
                className="rounded-lg border border-[var(--stroke-1)] p-4"
              >
                <div className="flex items-center gap-2 mb-3">
                  <Chip variant="violet">{name}</Chip>
                  <span className="text-sm text-[var(--text-secondary)]">Dense</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-[var(--text-tertiary)]">Dimensions</span>
                    <p className="font-semibold">{config.size}</p>
                  </div>
                  <div>
                    <span className="text-[var(--text-tertiary)]">Distance</span>
                    <p className="font-semibold">{config.distance}</p>
                  </div>
                </div>
              </div>
            ))}
            {Object.keys(sparseVectors).map((name) => (
              <div
                key={name}
                className="rounded-lg border border-[var(--stroke-1)] p-4"
              >
                <div className="flex items-center gap-2 mb-3">
                  <Chip variant="blue">{name}</Chip>
                  <span className="text-sm text-[var(--text-secondary)]">Sparse</span>
                </div>
                <p className="text-sm text-[var(--text-tertiary)]">
                  BM25 / SPLADE compatible
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Papers Browser */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>
              {isSearchMode ? `Search Results (${searchResults.length})` : "Papers"}
            </CardTitle>
            <form onSubmit={handleSearch} className="flex items-center gap-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-tertiary)]" />
                <Input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Vector search papers..."
                  className="pl-9 w-64"
                />
              </div>
              <Button type="submit" variant="primary" disabled={searching || !searchQuery.trim()}>
                {searching ? <RefreshCw className="h-4 w-4 animate-spin" /> : "Search"}
              </Button>
              {isSearchMode && (
                <Button type="button" variant="secondary" onClick={refresh}>
                  Clear
                </Button>
              )}
            </form>
          </div>
        </CardHeader>
        <CardContent>
          {/* Search Timeline */}
          <SearchTimeline 
            steps={traceSteps} 
            totalTime={searchTotalTime} 
            isVisible={showTimeline && (searching || traceSteps.length > 0)} 
          />

          {loadingPoints ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-5 w-5 animate-spin text-[var(--text-tertiary)]" />
            </div>
          ) : searching && traceSteps.length === 0 ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-5 w-5 animate-spin text-[var(--text-tertiary)]" />
            </div>
          ) : (
            <div className="grid gap-3 lg:grid-cols-2">
              {displayPoints.slice(0, 20).map((point) => {
                const paper = getPaper(point);
                const authors = paper?.authors as Array<{ name?: string }> | undefined;
                const meshTerms = paper?.mesh_terms as Array<{ term?: string }> | undefined;
                const pubDate = paper?.publication_date as string | undefined;
                const year = pubDate ? pubDate.split("-")[0] : null;

                return (
                  <div
                    key={String(point.id)}
                    className={cn(
                      "cursor-pointer rounded-lg border border-[var(--stroke-1)] p-4 transition-colors hover:bg-[var(--bg-2)]",
                      selectedPoint?.id === point.id && "bg-[var(--selection)] border-[var(--blue)]"
                    )}
                    onClick={() => setSelectedPoint(selectedPoint?.id === point.id ? null : point)}
                  >
                    <div className="flex items-start gap-3">
                      <FileText className="h-5 w-5 mt-0.5 text-[var(--violet)] shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <code className="text-xs text-[var(--text-tertiary)]">
                            PMID:{String(paper?.pmid || point.id)}
                          </code>
                          {point.score !== undefined && (
                            <Chip variant="violet" className="text-[10px]">
                              Score: {point.score.toFixed(3)}
                            </Chip>
                          )}
                          {year && (
                            <Chip variant="default" className="text-[10px]">
                              {year}
                            </Chip>
                          )}
                          {typeof paper?.journal === "string" && paper.journal && (
                            <span className="text-[10px] text-[var(--text-tertiary)] truncate max-w-[120px]">
                              {paper.journal}
                            </span>
                          )}
                        </div>
                        <p className="font-medium text-sm line-clamp-2">
                          {String(paper?.title || "Untitled")}
                        </p>
                        {authors && authors.length > 0 && (
                          <p className="text-xs text-[var(--text-tertiary)] mt-1 line-clamp-1">
                            {authors.slice(0, 3).map(a => a.name).join(", ")}
                            {authors.length > 3 && ` +${authors.length - 3} more`}
                          </p>
                        )}
                        {selectedPoint?.id === point.id && typeof paper?.abstract === "string" && paper.abstract && (
                          <p className="text-sm text-[var(--text-secondary)] mt-2 line-clamp-4">
                            {paper.abstract}
                          </p>
                        )}
                        {meshTerms && meshTerms.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {meshTerms.slice(0, 4).map((mesh, i) => (
                              <Chip key={i} variant="default" className="text-[10px]">
                                {mesh.term || String(mesh)}
                              </Chip>
                            ))}
                            {meshTerms.length > 4 && (
                              <Chip variant="default" className="text-[10px]">
                                +{meshTerms.length - 4}
                              </Chip>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {displayPoints.length > 20 && (
            <p className="text-center text-sm text-[var(--text-tertiary)] mt-4">
              Showing 20 of {displayPoints.length} papers
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
