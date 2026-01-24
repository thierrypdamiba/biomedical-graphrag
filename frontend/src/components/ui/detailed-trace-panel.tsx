"use client";

import * as React from "react";
import {
  ChevronDown,
  ChevronRight,
  Database,
  Network,
  FileText,
  Cpu,
  Clock,
  Dna,
  Users,
  BookOpen,
  Link2,
  Copy,
  Check,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Chip } from "@/components/ui/chip";
import { Button } from "@/components/ui/button";

interface TraceStep {
  name: string;
  startTime: number;
  duration?: number;
  details?: Record<string, unknown>;
}

interface QdrantResult {
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
}

interface Neo4jResult {
  [key: string]: Array<{
    name?: string;
    symbol?: string;
    description?: string;
    type?: string;
    [key: string]: unknown;
  }> | unknown;
}

interface DetailedTracePanelProps {
  trace: TraceStep[];
  qdrantResults?: Record<string, QdrantResult[]>;
  neo4jResults?: Neo4jResult;
  totalLatency?: number;
  toolsUsed?: {
    qdrant?: string;
    neo4j?: string[];
  };
}

export function DetailedTracePanel({
  trace,
  qdrantResults,
  neo4jResults,
  totalLatency,
  toolsUsed,
}: DetailedTracePanelProps) {
  const [expandedSections, setExpandedSections] = React.useState<Set<string>>(
    new Set(["timeline"])
  );
  const [copiedId, setCopiedId] = React.useState<string | null>(null);

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Extract papers from Qdrant results
  const papers: QdrantResult[] = React.useMemo(() => {
    if (!qdrantResults) return [];
    const allPapers: QdrantResult[] = [];
    Object.values(qdrantResults).forEach((results) => {
      if (Array.isArray(results)) {
        allPapers.push(...results);
      }
    });
    return allPapers;
  }, [qdrantResults]);

  // Extract entities from Neo4j results
  const neo4jEntities = React.useMemo(() => {
    if (!neo4jResults) return { genes: [], authors: [], papers: [], others: [] };
    
    const genes: Array<Record<string, unknown>> = [];
    const authors: Array<Record<string, unknown>> = [];
    const papers: Array<Record<string, unknown>> = [];
    const others: Array<{ key: string; data: unknown }> = [];

    Object.entries(neo4jResults).forEach(([key, value]) => {
      if (Array.isArray(value)) {
        value.forEach((item) => {
          if (item.symbol || item.gene_id || key.toLowerCase().includes("gene")) {
            genes.push(item);
          } else if (item.name && (key.toLowerCase().includes("author") || item.affiliation)) {
            authors.push(item);
          } else if (item.pmid || item.title) {
            papers.push(item);
          } else {
            others.push({ key, data: item });
          }
        });
      } else if (typeof value === "object" && value !== null) {
        others.push({ key, data: value });
      }
    });

    return { genes, authors, papers, others };
  }, [neo4jResults]);

  const SectionHeader = ({
    title,
    icon: Icon,
    section,
    count,
    color = "violet",
  }: {
    title: string;
    icon: React.ElementType;
    section: string;
    count?: number;
    color?: string;
  }) => (
    <button
      onClick={() => toggleSection(section)}
      className="flex items-center gap-2 w-full p-3 hover:bg-[var(--bg-2)] rounded-lg transition-colors"
    >
      {expandedSections.has(section) ? (
        <ChevronDown className="h-4 w-4 text-[var(--text-tertiary)]" />
      ) : (
        <ChevronRight className="h-4 w-4 text-[var(--text-tertiary)]" />
      )}
      <Icon className={cn("h-4 w-4", `text-[var(--${color})]`)} />
      <span className="font-medium text-sm">{title}</span>
      {count !== undefined && (
        <Chip variant="default" className="ml-auto text-xs">
          {count}
        </Chip>
      )}
    </button>
  );

  if (trace.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-[var(--text-tertiary)]">
        No trace data. Run a search to see execution details.
      </div>
    );
  }

  return (
    <div className="space-y-2 overflow-y-auto">
      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-2 p-3 bg-[var(--bg-2)] rounded-lg">
        <div className="text-center">
          <div className="text-xs text-[var(--text-tertiary)]">Total Time</div>
          <div className="text-lg font-semibold text-[var(--teal)]">
            {totalLatency || 0}ms
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-[var(--text-tertiary)]">Documents</div>
          <div className="text-lg font-semibold text-[var(--violet)]">
            {papers.length}
          </div>
        </div>
        <div className="text-center">
          <div className="text-xs text-[var(--text-tertiary)]">Entities</div>
          <div className="text-lg font-semibold text-[var(--blue)]">
            {neo4jEntities.genes.length + neo4jEntities.authors.length}
          </div>
        </div>
      </div>

      {/* Timeline Section */}
      <div className="border border-[var(--stroke-1)] rounded-lg overflow-hidden">
        <SectionHeader
          title="Execution Timeline"
          icon={Clock}
          section="timeline"
          color="teal"
        />
        {expandedSections.has("timeline") && (
          <div className="p-3 space-y-2 border-t border-[var(--stroke-1)]">
            {trace.map((step, i) => {
              const widthPercent = totalLatency
                ? Math.max(8, ((step.duration || 0) / totalLatency) * 100)
                : 50;
              const colors = ["bg-[var(--blue)]", "bg-[var(--violet)]", "bg-[var(--teal)]"];
              
              return (
                <div key={i} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-[var(--text-secondary)]">{step.name}</span>
                    <span className="text-[var(--text-tertiary)]">{step.duration}ms</span>
                  </div>
                  <div className="h-6 bg-[var(--bg-2)] rounded relative">
                    <div
                      className={cn(
                        "h-full rounded flex items-center justify-end pr-2",
                        colors[i % colors.length]
                      )}
                      style={{ width: `${widthPercent}%` }}
                    >
                      <span className="text-[10px] text-white font-medium">
                        {step.duration}ms
                      </span>
                    </div>
                  </div>
                  {step.details && (
                    <div className="text-[10px] text-[var(--text-tertiary)] flex flex-wrap gap-2">
                      {Object.entries(step.details).map(([key, value]) => (
                        <span key={key}>
                          {key}: {JSON.stringify(value)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Qdrant Documents Section */}
      {papers.length > 0 && (
        <div className="border border-[var(--stroke-1)] rounded-lg overflow-hidden">
          <SectionHeader
            title="Retrieved Documents"
            icon={Database}
            section="qdrant"
            count={papers.length}
            color="violet"
          />
          {expandedSections.has("qdrant") && (
            <div className="p-3 space-y-2 border-t border-[var(--stroke-1)] max-h-[300px] overflow-y-auto">
              {papers.slice(0, 10).map((paper, i) => {
                const p = paper.payload?.paper || paper.payload;
                const title = p?.title || "Untitled";
                const pmid = p?.pmid || paper.id;
                const authors = p?.authors?.slice(0, 2).map((a) => a.name).join(", ");
                const abstractPreview = p?.abstract?.slice(0, 150);

                return (
                  <div
                    key={i}
                    className="p-2 rounded border border-[var(--stroke-1)] hover:bg-[var(--bg-2)] transition-colors"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <FileText className="h-3 w-3 text-[var(--violet)] shrink-0" />
                          <code className="text-[10px] text-[var(--text-tertiary)]">
                            PMID:{pmid}
                          </code>
                          {paper.score !== undefined && (
                            <Chip variant="blue" className="text-[9px]">
                              {paper.score.toFixed(3)}
                            </Chip>
                          )}
                        </div>
                        <p className="text-xs font-medium line-clamp-2">{title}</p>
                        {authors && (
                          <p className="text-[10px] text-[var(--text-tertiary)] mt-1">
                            {authors}
                          </p>
                        )}
                        {abstractPreview && (
                          <p className="text-[10px] text-[var(--text-tertiary)] mt-1 line-clamp-2">
                            {abstractPreview}...
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
              {papers.length > 10 && (
                <p className="text-xs text-center text-[var(--text-tertiary)]">
                  +{papers.length - 10} more documents
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* Neo4j Genes Section */}
      {neo4jEntities.genes.length > 0 && (
        <div className="border border-[var(--stroke-1)] rounded-lg overflow-hidden">
          <SectionHeader
            title="Genes Found"
            icon={Dna}
            section="genes"
            count={neo4jEntities.genes.length}
            color="teal"
          />
          {expandedSections.has("genes") && (
            <div className="p-3 border-t border-[var(--stroke-1)]">
              <div className="flex flex-wrap gap-1">
                {neo4jEntities.genes.map((gene, i) => (
                  <Chip key={i} variant="teal" className="text-xs">
                    {(gene.symbol as string) || (gene.name as string) || `Gene ${i + 1}`}
                  </Chip>
                ))}
              </div>
              {neo4jEntities.genes.length > 0 && neo4jEntities.genes[0].description && (
                <div className="mt-3 space-y-2">
                  {neo4jEntities.genes.slice(0, 3).map((gene, i) => (
                    <div key={i} className="text-xs p-2 bg-[var(--bg-2)] rounded">
                      <div className="font-medium text-[var(--teal)]">
                        {(gene.symbol as string) || (gene.name as string)}
                      </div>
                      <p className="text-[var(--text-secondary)] mt-1 line-clamp-2">
                        {gene.description as string}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Neo4j Authors Section */}
      {neo4jEntities.authors.length > 0 && (
        <div className="border border-[var(--stroke-1)] rounded-lg overflow-hidden">
          <SectionHeader
            title="Authors Found"
            icon={Users}
            section="authors"
            count={neo4jEntities.authors.length}
            color="blue"
          />
          {expandedSections.has("authors") && (
            <div className="p-3 border-t border-[var(--stroke-1)]">
              <div className="flex flex-wrap gap-1">
                {neo4jEntities.authors.slice(0, 20).map((author, i) => (
                  <Chip key={i} variant="default" className="text-xs">
                    {author.name as string}
                  </Chip>
                ))}
                {neo4jEntities.authors.length > 20 && (
                  <Chip variant="default" className="text-xs">
                    +{neo4jEntities.authors.length - 20} more
                  </Chip>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tools Used Section */}
      {toolsUsed && (
        <div className="border border-[var(--stroke-1)] rounded-lg overflow-hidden">
          <SectionHeader
            title="Tools Used"
            icon={Cpu}
            section="tools"
            color="amaranth"
          />
          {expandedSections.has("tools") && (
            <div className="p-3 border-t border-[var(--stroke-1)] space-y-2">
              {toolsUsed.qdrant && (
                <div className="flex items-center gap-2">
                  <Database className="h-3 w-3 text-[var(--violet)]" />
                  <span className="text-xs text-[var(--text-secondary)]">Qdrant:</span>
                  <Chip variant="violet" className="text-xs">
                    {toolsUsed.qdrant}
                  </Chip>
                </div>
              )}
              {toolsUsed.neo4j && toolsUsed.neo4j.length > 0 && (
                <div className="flex items-center gap-2 flex-wrap">
                  <Network className="h-3 w-3 text-[var(--teal)]" />
                  <span className="text-xs text-[var(--text-secondary)]">Neo4j:</span>
                  {toolsUsed.neo4j.map((tool, i) => (
                    <Chip key={i} variant="teal" className="text-xs">
                      {tool}
                    </Chip>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Raw Data Section */}
      <div className="border border-[var(--stroke-1)] rounded-lg overflow-hidden">
        <SectionHeader
          title="Raw Data"
          icon={FileText}
          section="raw"
          color="default"
        />
        {expandedSections.has("raw") && (
          <div className="p-3 border-t border-[var(--stroke-1)] space-y-3">
            {qdrantResults && Object.keys(qdrantResults).length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-[var(--text-secondary)]">
                    Qdrant Results
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 text-xs"
                    onClick={() =>
                      copyToClipboard(JSON.stringify(qdrantResults, null, 2), "qdrant")
                    }
                  >
                    {copiedId === "qdrant" ? (
                      <Check className="h-3 w-3" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
                <pre className="rounded bg-[#0B1220] p-2 text-[10px] overflow-x-auto max-h-32 overflow-y-auto">
                  {JSON.stringify(qdrantResults, null, 2).slice(0, 2000)}
                  {JSON.stringify(qdrantResults, null, 2).length > 2000 && "..."}
                </pre>
              </div>
            )}
            {neo4jResults && Object.keys(neo4jResults).length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-[var(--text-secondary)]">
                    Neo4j Results
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-6 text-xs"
                    onClick={() =>
                      copyToClipboard(JSON.stringify(neo4jResults, null, 2), "neo4j")
                    }
                  >
                    {copiedId === "neo4j" ? (
                      <Check className="h-3 w-3" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
                <pre className="rounded bg-[#0B1220] p-2 text-[10px] overflow-x-auto max-h-32 overflow-y-auto">
                  {JSON.stringify(neo4jResults, null, 2).slice(0, 2000)}
                  {JSON.stringify(neo4jResults, null, 2).length > 2000 && "..."}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
