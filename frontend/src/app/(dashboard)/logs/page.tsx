"use client";

import * as React from "react";
import { Pause, Play, Filter, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Chip } from "@/components/ui/chip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

const sampleLogs = [
  {
    id: "log-001",
    timestamp: new Date(),
    level: "info" as const,
    service: "qdrant",
    message: "Collection 'products' indexed successfully",
  },
  {
    id: "log-002",
    timestamp: new Date(Date.now() - 5000),
    level: "debug" as const,
    service: "qdrant",
    message: "Query executed in 42ms",
  },
  {
    id: "log-003",
    timestamp: new Date(Date.now() - 12000),
    level: "warn" as const,
    service: "api",
    message: "Rate limit approaching for client 192.168.1.100",
  },
  {
    id: "log-004",
    timestamp: new Date(Date.now() - 30000),
    level: "error" as const,
    service: "qdrant",
    message: "Failed to connect to replica node-2",
  },
  {
    id: "log-005",
    timestamp: new Date(Date.now() - 45000),
    level: "info" as const,
    service: "api",
    message: "New API key created: demo-key-xxx",
  },
  {
    id: "log-006",
    timestamp: new Date(Date.now() - 60000),
    level: "info" as const,
    service: "qdrant",
    message: "Snapshot created for collection 'products'",
  },
  {
    id: "log-007",
    timestamp: new Date(Date.now() - 90000),
    level: "debug" as const,
    service: "qdrant",
    message: "HNSW index optimization started",
  },
  {
    id: "log-008",
    timestamp: new Date(Date.now() - 120000),
    level: "info" as const,
    service: "api",
    message: "Cluster health check passed",
  },
];

const levelColors = {
  info: "text-[var(--blue)]",
  debug: "text-[var(--text-tertiary)]",
  warn: "text-yellow-500",
  error: "text-[var(--amaranth)]",
};

const levelBgColors = {
  info: "bg-[rgba(47,111,240,0.18)]",
  debug: "bg-[var(--bg-2)]",
  warn: "bg-[rgba(234,179,8,0.18)]",
  error: "bg-[rgba(220,36,76,0.18)]",
};

export default function LogsPage() {
  const [paused, setPaused] = React.useState(false);
  const [filterLevel, setFilterLevel] = React.useState<string>("all");
  const [filterService, setFilterService] = React.useState<string>("all");
  const [searchText, setSearchText] = React.useState("");
  const [selectedLog, setSelectedLog] = React.useState<typeof sampleLogs[0] | null>(
    null
  );

  const filteredLogs = sampleLogs.filter((log) => {
    if (filterLevel !== "all" && log.level !== filterLevel) return false;
    if (filterService !== "all" && log.service !== filterService) return false;
    if (searchText && !log.message.toLowerCase().includes(searchText.toLowerCase()))
      return false;
    return true;
  });

  return (
    <div className="flex h-[calc(100vh-56px-48px)] gap-4">
      {/* Main Logs */}
      <div className="flex flex-1 flex-col overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)]">
        {/* Filters */}
        <div className="flex items-center gap-2 border-b border-[var(--stroke-1)] p-3">
          <Button
            variant={paused ? "primary" : "secondary"}
            size="sm"
            onClick={() => setPaused(!paused)}
          >
            {paused ? (
              <>
                <Play className="h-4 w-4 mr-1" />
                Resume
              </>
            ) : (
              <>
                <Pause className="h-4 w-4 mr-1" />
                Pause
              </>
            )}
          </Button>
          <div className="w-px h-6 bg-[var(--stroke-1)]" />
          <Select value={filterLevel} onValueChange={setFilterLevel}>
            <SelectTrigger className="w-28 h-8">
              <SelectValue placeholder="Level" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All levels</SelectItem>
              <SelectItem value="info">Info</SelectItem>
              <SelectItem value="debug">Debug</SelectItem>
              <SelectItem value="warn">Warning</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>
          <Select value={filterService} onValueChange={setFilterService}>
            <SelectTrigger className="w-28 h-8">
              <SelectValue placeholder="Service" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All services</SelectItem>
              <SelectItem value="qdrant">qdrant</SelectItem>
              <SelectItem value="api">api</SelectItem>
            </SelectContent>
          </Select>
          <Input
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Search logs..."
            className="max-w-xs h-8"
          />
          {(filterLevel !== "all" || filterService !== "all" || searchText) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setFilterLevel("all");
                setFilterService("all");
                setSearchText("");
              }}
            >
              Clear filters
            </Button>
          )}
        </div>

        {/* Logs List */}
        <div className="flex-1 overflow-y-auto font-mono text-sm">
          {filteredLogs.map((log) => (
            <div
              key={log.id}
              onClick={() => setSelectedLog(log)}
              className={cn(
                "flex cursor-pointer items-start gap-4 border-b border-[var(--stroke-1)] px-4 py-2 hover:bg-[var(--bg-2)]",
                selectedLog?.id === log.id && "bg-[var(--selection)]"
              )}
            >
              <span className="shrink-0 text-[var(--text-tertiary)]">
                {log.timestamp.toLocaleTimeString()}
              </span>
              <span
                className={cn(
                  "shrink-0 w-12 text-center rounded px-1",
                  levelColors[log.level],
                  levelBgColors[log.level]
                )}
              >
                {log.level.toUpperCase()}
              </span>
              <span className="shrink-0 w-16 text-[var(--text-secondary)]">
                {log.service}
              </span>
              <span className="flex-1 text-[var(--text-primary)]">
                {log.message}
              </span>
            </div>
          ))}
          {filteredLogs.length === 0 && (
            <div className="flex h-32 items-center justify-center text-[var(--text-tertiary)]">
              No logs match the current filters
            </div>
          )}
        </div>

        {/* Status Bar */}
        <div className="flex items-center justify-between border-t border-[var(--stroke-1)] px-4 py-2 text-xs text-[var(--text-tertiary)]">
          <span>
            {filteredLogs.length} logs
            {paused && " (paused)"}
          </span>
          <span>Auto-refresh: 1s</span>
        </div>
      </div>

      {/* Log Detail Drawer */}
      {selectedLog && (
        <div className="w-96 shrink-0 overflow-hidden rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)]">
          <div className="flex items-center justify-between border-b border-[var(--stroke-1)] p-3">
            <h3 className="font-semibold">Log Details</h3>
            <Button variant="ghost" size="icon" onClick={() => setSelectedLog(null)}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="space-y-4 p-4">
            <div>
              <p className="text-sm text-[var(--text-secondary)]">Timestamp</p>
              <p className="font-mono">{selectedLog.timestamp.toISOString()}</p>
            </div>
            <div>
              <p className="text-sm text-[var(--text-secondary)]">Level</p>
              <Chip
                variant={
                  selectedLog.level === "error"
                    ? "amaranth"
                    : selectedLog.level === "warn"
                    ? "default"
                    : selectedLog.level === "info"
                    ? "blue"
                    : "default"
                }
              >
                {selectedLog.level.toUpperCase()}
              </Chip>
            </div>
            <div>
              <p className="text-sm text-[var(--text-secondary)]">Service</p>
              <p>{selectedLog.service}</p>
            </div>
            <div>
              <p className="text-sm text-[var(--text-secondary)]">Message</p>
              <p className="rounded-lg bg-[var(--bg-2)] p-3 font-mono text-sm">
                {selectedLog.message}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
