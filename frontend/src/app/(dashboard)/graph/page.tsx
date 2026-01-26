"use client";

import * as React from "react";
import {
  Network,
  RefreshCw,
  AlertCircle,
  Circle,
  ArrowRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Chip } from "@/components/ui/chip";
import { formatNumber } from "@/lib/utils";

interface NodeLabel {
  label: string;
  count: number;
}

interface RelationshipType {
  type: string;
  count: number;
}

interface GraphStats {
  nodeLabels: NodeLabel[];
  relationshipTypes: RelationshipType[];
  totalNodes: number;
  totalRelationships: number;
}

const nodeColors: Record<string, string> = {
  Paper: "var(--violet)",
  Author: "var(--blue)",
  Gene: "var(--teal)",
  Disease: "var(--amaranth)",
  Drug: "#10B981",
  Pathway: "#F59E0B",
  default: "var(--text-tertiary)",
};

export default function GraphPage() {
  const [stats, setStats] = React.useState<GraphStats | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/neo4j/stats");
      const data = await response.json();

      if (data.error) {
        throw new Error(data.error);
      }

      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch graph stats");
    } finally {
      setLoading(false);
    }
  };

  const getNodeColor = (label: string): string => {
    return nodeColors[label] || nodeColors.default;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Knowledge Graph</h1>
          <p className="mt-1 text-[var(--text-secondary)]">
            Explore the biomedical knowledge graph powered by Neo4j
          </p>
        </div>
        <Button
          variant="secondary"
          onClick={fetchStats}
          disabled={loading}
        >
          <RefreshCw className={loading ? "h-4 w-4 mr-2 animate-spin" : "h-4 w-4 mr-2"} />
          Refresh
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-16">
          <RefreshCw className="h-8 w-8 animate-spin text-[var(--text-tertiary)]" />
        </div>
      ) : error ? (
        <Card>
          <CardContent className="flex flex-col items-center gap-4 py-16">
            <AlertCircle className="h-12 w-12 text-[var(--amaranth)]" />
            <p className="text-[var(--text-secondary)]">{error}</p>
            <Button variant="primary" onClick={fetchStats}>
              Try Again
            </Button>
          </CardContent>
        </Card>
      ) : stats ? (
        <>
          {/* Summary Cards */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                  Total Nodes
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold">
                  {formatNumber(stats.totalNodes)}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                  Total Relationships
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold">
                  {formatNumber(stats.totalRelationships)}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                  Node Types
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold">
                  {stats.nodeLabels.length}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                  Relationship Types
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-semibold">
                  {stats.relationshipTypes.length}
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* Node Labels */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Circle className="h-5 w-5" />
                  Node Types
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {stats.nodeLabels.map((node) => (
                    <div key={node.label} className="flex items-center gap-3">
                      <div
                        className="h-4 w-4 rounded-full shrink-0"
                        style={{ backgroundColor: getNodeColor(node.label) }}
                      />
                      <span className="flex-1 font-medium">{node.label}</span>
                      <Chip variant="default">
                        {formatNumber(node.count)}
                      </Chip>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Relationship Types */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ArrowRight className="h-5 w-5" />
                  Relationship Types
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {stats.relationshipTypes.map((rel) => (
                    <div key={rel.type} className="flex items-center gap-3">
                      <div className="h-0.5 w-4 bg-[var(--text-tertiary)] shrink-0" />
                      <span className="flex-1 font-mono text-sm">{rel.type}</span>
                      <Chip variant="default">
                        {formatNumber(rel.count)}
                      </Chip>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Graph Schema Visualization */}
          <Card>
            <CardHeader>
              <CardTitle>Graph Schema</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap items-center justify-center gap-4 py-8">
                {stats.nodeLabels.slice(0, 5).map((node, i) => (
                  <React.Fragment key={node.label}>
                    <div className="flex flex-col items-center gap-2">
                      <div
                        className="h-16 w-16 rounded-full flex items-center justify-center text-white font-semibold"
                        style={{ backgroundColor: getNodeColor(node.label) }}
                      >
                        {node.label.slice(0, 2)}
                      </div>
                      <span className="text-sm font-medium">{node.label}</span>
                      <span className="text-xs text-[var(--text-tertiary)]">
                        {formatNumber(node.count)}
                      </span>
                    </div>
                    {i < Math.min(stats.nodeLabels.length - 1, 4) && (
                      <div className="flex items-center gap-1 text-[var(--text-tertiary)]">
                        <div className="h-0.5 w-8 bg-current" />
                        <ArrowRight className="h-4 w-4" />
                      </div>
                    )}
                  </React.Fragment>
                ))}
              </div>
              <p className="text-center text-sm text-[var(--text-tertiary)]">
                Simplified schema visualization. Use Neo4j Browser for full graph exploration.
              </p>
            </CardContent>
          </Card>
        </>
      ) : null}
    </div>
  );
}
