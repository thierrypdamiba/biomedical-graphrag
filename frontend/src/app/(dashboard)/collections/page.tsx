"use client";

import * as React from "react";
import {
  Database,
  Search,
  Plus,
  MoreHorizontal,
  Trash2,
  Download,
  Upload,
  Settings,
  ChevronRight,
  Filter,
  Play,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Chip } from "@/components/ui/chip";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn, formatNumber } from "@/lib/utils";

const sampleCollections = [
  {
    name: "products",
    pointsCount: 125420,
    vectorsConfig: { size: 768, distance: "Cosine" as const },
    payloadFields: ["name", "description", "category", "price", "tags"],
  },
  {
    name: "documents",
    pointsCount: 45230,
    vectorsConfig: { size: 1024, distance: "Dot" as const },
    payloadFields: ["title", "content", "author", "date"],
  },
  {
    name: "images",
    pointsCount: 89100,
    vectorsConfig: { size: 512, distance: "Euclid" as const },
    payloadFields: ["url", "caption", "tags", "dimensions"],
  },
];

const samplePoints = [
  {
    id: "point-001",
    payload: {
      name: "TensorFlow Pro",
      category: "software",
      price: 2999,
      tags: ["ml", "ai", "enterprise"],
    },
  },
  {
    id: "point-002",
    payload: {
      name: "AI Vision SDK",
      category: "software",
      price: 1499,
      tags: ["vision", "ml"],
    },
  },
  {
    id: "point-003",
    payload: {
      name: "Data Pipeline Tool",
      category: "data",
      price: 799,
      tags: ["etl", "pipeline"],
    },
  },
  {
    id: "point-004",
    payload: {
      name: "Neural Network Kit",
      category: "software",
      price: 1299,
      tags: ["nn", "deep-learning"],
    },
  },
  {
    id: "point-005",
    payload: {
      name: "Cloud ML Platform",
      category: "cloud",
      price: 4999,
      tags: ["cloud", "ml", "scalable"],
    },
  },
];

export default function CollectionsPage() {
  const [selectedCollection, setSelectedCollection] = React.useState(
    sampleCollections[0]
  );
  const [searchQuery, setSearchQuery] = React.useState("");
  const [collectionFilter, setCollectionFilter] = React.useState("");

  const filteredCollections = sampleCollections.filter((c) =>
    c.name.toLowerCase().includes(collectionFilter.toLowerCase())
  );

  return (
    <div className="flex h-[calc(100vh-56px-48px)] gap-4">
      {/* Left Sidebar - Collections List */}
      <div className="w-64 shrink-0 overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)]">
        <div className="border-b border-[var(--stroke-1)] p-3">
          <div className="flex items-center gap-2">
            <Input
              value={collectionFilter}
              onChange={(e) => setCollectionFilter(e.target.value)}
              placeholder="Search collections..."
              className="h-8"
            />
            <Button variant="primary" size="icon" className="h-8 w-8 shrink-0">
              <Plus className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="overflow-y-auto p-2">
          {filteredCollections.map((collection) => (
            <button
              key={collection.name}
              onClick={() => setSelectedCollection(collection)}
              className={cn(
                "flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors",
                selectedCollection?.name === collection.name
                  ? "bg-[var(--bg-2)] text-[var(--text-primary)]"
                  : "text-[var(--text-secondary)] hover:bg-[var(--bg-2)]"
              )}
            >
              <Database className="h-4 w-4 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="truncate font-medium">{collection.name}</div>
                <div className="text-xs text-[var(--text-tertiary)]">
                  {formatNumber(collection.pointsCount)} points
                </div>
              </div>
              <ChevronRight className="h-4 w-4 shrink-0 text-[var(--text-tertiary)]" />
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)]">
        {selectedCollection ? (
          <Tabs defaultValue="overview" className="h-full flex flex-col">
            <div className="flex items-center justify-between border-b border-[var(--stroke-1)] px-4 pt-4">
              <div>
                <h2 className="text-lg font-semibold">{selectedCollection.name}</h2>
                <p className="text-sm text-[var(--text-secondary)]">
                  {formatNumber(selectedCollection.pointsCount)} points •{" "}
                  {selectedCollection.vectorsConfig.size} dims •{" "}
                  {selectedCollection.vectorsConfig.distance}
                </p>
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="icon">
                    <MoreHorizontal className="h-5 w-5" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem>
                    <Download className="h-4 w-4 mr-2" />
                    Create snapshot
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Upload className="h-4 w-4 mr-2" />
                    Upload points
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className="text-[var(--amaranth)]">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete collection
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            <TabsList className="px-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="data">Data</TabsTrigger>
              <TabsTrigger value="search">Search</TabsTrigger>
              <TabsTrigger value="snapshots">Snapshots</TabsTrigger>
              <TabsTrigger value="config">Config</TabsTrigger>
            </TabsList>

            {/* Overview Tab */}
            <TabsContent value="overview" className="flex-1 overflow-y-auto p-4">
              <div className="grid gap-4 sm:grid-cols-3">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                      Points
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-semibold">
                      {formatNumber(selectedCollection.pointsCount)}
                    </div>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                      Vector Config
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-semibold">
                      {selectedCollection.vectorsConfig.size}
                    </div>
                    <p className="text-sm text-[var(--text-secondary)]">
                      {selectedCollection.vectorsConfig.distance}
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-[var(--text-secondary)]">
                      Payload Fields
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-semibold">
                      {selectedCollection.payloadFields.length}
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="mt-6">
                <h3 className="mb-3 font-semibold">Payload Schema</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedCollection.payloadFields.map((field) => (
                    <Chip key={field} variant="default">
                      {field}
                    </Chip>
                  ))}
                </div>
              </div>

              <div className="mt-6 flex gap-3">
                <Button variant="secondary">
                  <Download className="h-4 w-4 mr-2" />
                  Create Snapshot
                </Button>
                <Button variant="secondary">
                  <Upload className="h-4 w-4 mr-2" />
                  Upload Points
                </Button>
              </div>
            </TabsContent>

            {/* Data Tab */}
            <TabsContent value="data" className="flex-1 overflow-y-auto p-4">
              <div className="mb-4 flex items-center gap-2">
                <Input
                  placeholder="Filter by ID..."
                  className="max-w-xs"
                />
                <Button variant="secondary" size="sm">
                  <Filter className="h-4 w-4 mr-1" />
                  Filters
                </Button>
              </div>
              <div className="rounded-lg border border-[var(--stroke-1)] overflow-hidden">
                <table className="w-full">
                  <thead className="bg-[var(--bg-2)]">
                    <tr className="text-left text-sm">
                      <th className="px-4 py-3 font-medium">ID</th>
                      <th className="px-4 py-3 font-medium">Payload</th>
                      <th className="px-4 py-3 font-medium w-20"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--stroke-1)]">
                    {samplePoints.map((point) => (
                      <tr
                        key={point.id}
                        className="hover:bg-[var(--bg-2)] cursor-pointer"
                      >
                        <td className="px-4 py-3">
                          <code className="text-sm text-[var(--text-secondary)]">
                            {point.id}
                          </code>
                        </td>
                        <td className="px-4 py-3">
                          <span className="text-sm line-clamp-1">
                            {JSON.stringify(point.payload)}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <Button variant="ghost" size="sm">
                            View
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 flex items-center justify-between">
                <span className="text-sm text-[var(--text-secondary)]">
                  Showing 1-5 of {formatNumber(selectedCollection.pointsCount)}
                </span>
                <div className="flex gap-2">
                  <Button variant="secondary" size="sm" disabled>
                    Previous
                  </Button>
                  <Button variant="secondary" size="sm">
                    Next
                  </Button>
                </div>
              </div>
            </TabsContent>

            {/* Search Tab */}
            <TabsContent value="search" className="flex-1 overflow-y-auto p-4">
              <div className="space-y-4">
                <div>
                  <label className="mb-1.5 block text-sm font-medium">Query</label>
                  <Input
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Enter search query..."
                  />
                </div>
                <div className="grid gap-4 sm:grid-cols-3">
                  <div>
                    <label className="mb-1.5 block text-sm font-medium">Mode</label>
                    <Select defaultValue="hybrid">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="dense">Dense</SelectItem>
                        <SelectItem value="sparse">Sparse</SelectItem>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-sm font-medium">top_k</label>
                    <Select defaultValue="10">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="10">10</SelectItem>
                        <SelectItem value="20">20</SelectItem>
                        <SelectItem value="50">50</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="mb-1.5 block text-sm font-medium">Score threshold</label>
                    <Input placeholder="0.0" />
                  </div>
                </div>
                <div>
                  <label className="mb-1.5 block text-sm font-medium">Filters (JSON)</label>
                  <textarea
                    className="w-full rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)] p-3 text-sm font-mono placeholder:text-[var(--text-tertiary)] focus:border-[var(--stroke-2)] focus:outline-none focus:ring-2 focus:ring-[var(--blue)]"
                    rows={3}
                    placeholder='{"must": [{"key": "category", "match": {"value": "software"}}]}'
                  />
                </div>
                <div className="flex gap-3">
                  <Button variant="primary">
                    <Play className="h-4 w-4 mr-2" />
                    Run Search
                  </Button>
                  <Button variant="secondary">Open in Assistant</Button>
                </div>
              </div>
            </TabsContent>

            {/* Snapshots Tab */}
            <TabsContent value="snapshots" className="flex-1 overflow-y-auto p-4">
              <div className="mb-4">
                <Button variant="primary" size="sm">
                  <Plus className="h-4 w-4 mr-2" />
                  Create Snapshot
                </Button>
              </div>
              <div className="rounded-lg border border-[var(--stroke-1)] overflow-hidden">
                <table className="w-full">
                  <thead className="bg-[var(--bg-2)]">
                    <tr className="text-left text-sm">
                      <th className="px-4 py-3 font-medium">Name</th>
                      <th className="px-4 py-3 font-medium">Created</th>
                      <th className="px-4 py-3 font-medium">Size</th>
                      <th className="px-4 py-3 font-medium w-32"></th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--stroke-1)]">
                    <tr className="hover:bg-[var(--bg-2)]">
                      <td className="px-4 py-3 font-mono text-sm">
                        snapshot-2025-01-20
                      </td>
                      <td className="px-4 py-3 text-sm text-[var(--text-secondary)]">
                        Jan 20, 2025
                      </td>
                      <td className="px-4 py-3 text-sm text-[var(--text-secondary)]">
                        1.2 GB
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex gap-2">
                          <Button variant="ghost" size="sm">
                            Restore
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-[var(--amaranth)]"
                          >
                            Delete
                          </Button>
                        </div>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </TabsContent>

            {/* Config Tab */}
            <TabsContent value="config" className="flex-1 overflow-y-auto p-4">
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Vector Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-[var(--text-secondary)]">Size</span>
                      <span>{selectedCollection.vectorsConfig.size}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--text-secondary)]">Distance</span>
                      <span>{selectedCollection.vectorsConfig.distance}</span>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Indexing</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-[var(--text-secondary)]">HNSW m</span>
                      <span>16</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--text-secondary)]">HNSW ef_construct</span>
                      <span>100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[var(--text-secondary)]">
                        On-disk payload
                      </span>
                      <span>false</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        ) : (
          <div className="flex h-full items-center justify-center text-[var(--text-tertiary)]">
            Select a collection to view details
          </div>
        )}
      </div>
    </div>
  );
}
