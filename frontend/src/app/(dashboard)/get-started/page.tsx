"use client";

import * as React from "react";
import {
  BookOpen,
  ArrowRight,
  Eye,
  Package,
  FileJson,
  Brain,
  Sparkles,
  MessageSquare,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Chip } from "@/components/ui/chip";

const sampleDatasets = [
  {
    id: "ecommerce",
    name: "E-commerce Products",
    description: "10K product listings with descriptions and metadata",
    config: {
      dense: 768,
      sparse: "bm25",
      payloadFields: 14,
    },
    icon: Package,
  },
  {
    id: "scientific",
    name: "Scientific Papers",
    description: "ArXiv abstracts with citations and categories",
    config: {
      dense: 1024,
      sparse: "splade",
      payloadFields: 8,
    },
    icon: Brain,
  },
  {
    id: "movies",
    name: "Movie Database",
    description: "50K movies with plots, genres, and ratings",
    config: {
      dense: 384,
      sparse: "bm25",
      payloadFields: 12,
    },
    icon: FileJson,
  },
];

const tutorials = [
  {
    title: "Hybrid Search Basics",
    description: "Learn to combine dense and sparse vectors for better results",
    time: "5 min",
  },
  {
    title: "Filtering & Facets",
    description: "Build complex queries with payload filters",
    time: "8 min",
  },
  {
    title: "Performance Tuning",
    description: "Optimize your collections for production workloads",
    time: "12 min",
  },
];

const quickActions = [
  {
    title: "Try the Assistant",
    description: "Ask natural language questions about your data",
    icon: MessageSquare,
    href: "/assistant",
    color: "var(--violet)",
  },
  {
    title: "Explore Collections",
    description: "Browse and search your vector collections",
    icon: Package,
    href: "/collections",
    color: "var(--blue)",
  },
  {
    title: "Open Console",
    description: "Run raw REST queries against the API",
    icon: Sparkles,
    href: "/console",
    color: "var(--teal)",
  },
];

export default function GetStartedPage() {
  const [loading, setLoading] = React.useState<string | null>(null);

  const handleLoadSample = async (datasetId: string) => {
    setLoading(datasetId);
    // Simulate loading
    await new Promise((resolve) => setTimeout(resolve, 2000));
    setLoading(null);
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Get Started</h1>
        <p className="mt-1 text-[var(--text-secondary)]">
          Explore sample data or jump right into the demo
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-3">
        {/* Quick Actions */}
        <div className="lg:col-span-1 space-y-3">
          <h2 className="text-lg font-semibold mb-4">Quick Actions</h2>
          {quickActions.map((action) => (
            <Card
              key={action.title}
              className="cursor-pointer hover:bg-[var(--bg-2)] transition-colors"
            >
              <CardContent className="flex items-center gap-4 p-4">
                <div
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg"
                  style={{ backgroundColor: `${action.color}20` }}
                >
                  <action.icon
                    className="h-5 w-5"
                    style={{ color: action.color }}
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold">{action.title}</h3>
                  <p className="text-sm text-[var(--text-secondary)]">
                    {action.description}
                  </p>
                </div>
                <ArrowRight className="h-4 w-4 shrink-0 text-[var(--text-tertiary)]" />
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Sample Datasets */}
        <div className="lg:col-span-2 space-y-6">
          <div>
            <h2 className="text-lg font-semibold mb-4">Start with Samples</h2>
            <div className="space-y-3">
              {sampleDatasets.map((dataset) => (
                <Card key={dataset.id} className="overflow-hidden">
                  <CardContent className="p-4">
                    <div className="flex items-start gap-4">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--bg-2)]">
                        <dataset.icon className="h-5 w-5 text-[var(--violet)]" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold">{dataset.name}</h3>
                        <p className="mt-0.5 text-sm text-[var(--text-secondary)]">
                          {dataset.description}
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Chip variant="violet">
                            dense: {dataset.config.dense}
                          </Chip>
                          <Chip variant="blue">
                            sparse: {dataset.config.sparse}
                          </Chip>
                          <Chip variant="default">
                            {dataset.config.payloadFields} fields
                          </Chip>
                        </div>
                      </div>
                      <div className="flex flex-col gap-2">
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => handleLoadSample(dataset.id)}
                          disabled={loading !== null}
                        >
                          {loading === dataset.id ? (
                            <>
                              <div className="h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent" />
                              Loading...
                            </>
                          ) : (
                            "Load sample"
                          )}
                        </Button>
                        <Button variant="ghost" size="sm">
                          <Eye className="h-4 w-4 mr-1" />
                          View schema
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>

          {/* Tutorials */}
          <div>
            <h2 className="text-lg font-semibold mb-4">Recommended Tutorials</h2>
            <div className="grid gap-3 sm:grid-cols-3">
              {tutorials.map((tutorial, i) => (
                <Card
                  key={i}
                  className="cursor-pointer hover:bg-[var(--bg-2)] transition-colors"
                >
                  <CardContent className="p-4">
                    <div className="flex items-center gap-3 mb-2">
                      <BookOpen className="h-5 w-5 text-[var(--teal)]" />
                      <span className="text-xs text-[var(--text-tertiary)]">
                        {tutorial.time}
                      </span>
                    </div>
                    <h3 className="font-medium">{tutorial.title}</h3>
                    <p className="mt-1 text-sm text-[var(--text-secondary)]">
                      {tutorial.description}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
