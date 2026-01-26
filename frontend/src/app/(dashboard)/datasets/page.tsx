"use client";

import * as React from "react";
import { Plus, Upload, Database, Trash2, Download, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Chip } from "@/components/ui/chip";
import { formatNumber } from "@/lib/utils";

const sampleDatasets = [
  {
    id: "ds-001",
    name: "PubMed Abstracts",
    description: "Biomedical research paper abstracts with embeddings",
    records: 125000,
    size: "2.4 GB",
    format: "JSONL",
    created: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    status: "ready" as const,
  },
  {
    id: "ds-002",
    name: "Product Catalog",
    description: "E-commerce products with descriptions and metadata",
    records: 45000,
    size: "890 MB",
    format: "Parquet",
    created: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
    status: "ready" as const,
  },
  {
    id: "ds-003",
    name: "News Articles",
    description: "News articles from various sources",
    records: 0,
    size: "0 MB",
    format: "CSV",
    created: new Date(),
    status: "processing" as const,
  },
];

export default function DatasetsPage() {
  const [showUploadModal, setShowUploadModal] = React.useState(false);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Datasets</h1>
          <p className="mt-1 text-[var(--text-secondary)]">
            Manage your data sources and uploads
          </p>
        </div>
        <Button variant="primary" onClick={() => setShowUploadModal(true)}>
          <Upload className="h-4 w-4 mr-2" />
          Upload Dataset
        </Button>
      </div>

      {/* Upload Drop Zone */}
      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-[var(--stroke-2)] bg-[var(--bg-0)] p-8 text-center">
            <Upload className="h-10 w-10 text-[var(--text-tertiary)]" />
            <p className="mt-3 font-medium">Drop files here to upload</p>
            <p className="mt-1 text-sm text-[var(--text-tertiary)]">
              Supports JSONL, CSV, and Parquet formats up to 10GB
            </p>
            <Button variant="secondary" size="sm" className="mt-4">
              Browse files
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Datasets Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {sampleDatasets.map((dataset) => (
          <Card key={dataset.id} className="overflow-hidden">
            <CardHeader className="pb-2">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[var(--bg-2)]">
                    <Database className="h-5 w-5 text-[var(--violet)]" />
                  </div>
                  <div>
                    <CardTitle className="text-base">{dataset.name}</CardTitle>
                    <Chip
                      variant={
                        dataset.status === "ready" ? "teal" : "default"
                      }
                      className="mt-1"
                    >
                      {dataset.status === "processing" ? (
                        <>
                          <div className="h-2 w-2 animate-pulse rounded-full bg-current" />
                          Processing...
                        </>
                      ) : (
                        "Ready"
                      )}
                    </Chip>
                  </div>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-[var(--text-secondary)] line-clamp-2">
                {dataset.description}
              </p>
              <div className="mt-4 grid grid-cols-3 gap-2 text-sm">
                <div>
                  <p className="text-[var(--text-tertiary)]">Records</p>
                  <p className="font-medium">
                    {formatNumber(dataset.records)}
                  </p>
                </div>
                <div>
                  <p className="text-[var(--text-tertiary)]">Size</p>
                  <p className="font-medium">{dataset.size}</p>
                </div>
                <div>
                  <p className="text-[var(--text-tertiary)]">Format</p>
                  <p className="font-medium">{dataset.format}</p>
                </div>
              </div>
              <div className="mt-4 flex gap-2">
                <Button variant="secondary" size="sm" className="flex-1">
                  <Eye className="h-4 w-4 mr-1" />
                  Preview
                </Button>
                <Button variant="ghost" size="sm">
                  <Download className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-[var(--amaranth)]"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}

        {/* Add Dataset Card */}
        <Card className="flex min-h-[200px] cursor-pointer items-center justify-center border-dashed hover:bg-[var(--bg-2)] transition-colors">
          <div className="text-center">
            <Plus className="mx-auto h-8 w-8 text-[var(--text-tertiary)]" />
            <p className="mt-2 text-sm text-[var(--text-secondary)]">
              Add Dataset
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
