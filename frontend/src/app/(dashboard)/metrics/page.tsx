"use client";

import * as React from "react";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Simple chart component (placeholder for actual charting library)
function SimpleChart({
  data,
  color,
  label,
}: {
  data: number[];
  color: string;
  label: string;
}) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;

  return (
    <div className="h-32">
      <div className="flex h-full items-end gap-1">
        {data.map((value, i) => (
          <div
            key={i}
            className="flex-1 rounded-t transition-all hover:opacity-80"
            style={{
              height: `${((value - min) / range) * 100}%`,
              minHeight: "4px",
              backgroundColor: color,
            }}
            title={`${label}: ${value}`}
          />
        ))}
      </div>
    </div>
  );
}

const generateData = (points: number, base: number, variance: number) =>
  Array.from({ length: points }, () =>
    Math.round(base + (Math.random() - 0.5) * variance)
  );

export default function MetricsPage() {
  const [timeRange, setTimeRange] = React.useState("1h");

  const cpuData = generateData(24, 35, 30);
  const ramData = generateData(24, 60, 20);
  const diskData = generateData(24, 45, 10);
  const requestsData = generateData(24, 1200, 800);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Metrics</h1>
          <p className="mt-1 text-[var(--text-secondary)]">
            Monitor your cluster performance
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="15m">Last 15m</SelectItem>
              <SelectItem value="1h">Last 1h</SelectItem>
              <SelectItem value="24h">Last 24h</SelectItem>
              <SelectItem value="7d">Last 7d</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="secondary">
            <Download className="h-4 w-4 mr-2" />
            Export CSV
          </Button>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">CPU Usage</CardTitle>
            <span className="text-2xl font-semibold text-[var(--blue)]">
              {cpuData[cpuData.length - 1]}%
            </span>
          </CardHeader>
          <CardContent>
            <SimpleChart data={cpuData} color="var(--blue)" label="CPU" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Memory Usage</CardTitle>
            <span className="text-2xl font-semibold text-[var(--violet)]">
              {ramData[ramData.length - 1]}%
            </span>
          </CardHeader>
          <CardContent>
            <SimpleChart data={ramData} color="var(--violet)" label="RAM" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Disk Usage</CardTitle>
            <span className="text-2xl font-semibold text-[var(--teal)]">
              {diskData[diskData.length - 1]}%
            </span>
          </CardHeader>
          <CardContent>
            <SimpleChart data={diskData} color="var(--teal)" label="Disk" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Requests/min</CardTitle>
            <span className="text-2xl font-semibold text-[var(--amaranth)]">
              {requestsData[requestsData.length - 1]}
            </span>
          </CardHeader>
          <CardContent>
            <SimpleChart
              data={requestsData}
              color="var(--amaranth)"
              label="Requests"
            />
          </CardContent>
        </Card>
      </div>

      {/* Stats Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Summary Statistics</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-4">
            <div className="rounded-lg border border-[var(--stroke-1)] p-4">
              <p className="text-sm text-[var(--text-secondary)]">Avg CPU</p>
              <p className="text-xl font-semibold">
                {Math.round(cpuData.reduce((a, b) => a + b, 0) / cpuData.length)}%
              </p>
            </div>
            <div className="rounded-lg border border-[var(--stroke-1)] p-4">
              <p className="text-sm text-[var(--text-secondary)]">Avg Memory</p>
              <p className="text-xl font-semibold">
                {Math.round(ramData.reduce((a, b) => a + b, 0) / ramData.length)}%
              </p>
            </div>
            <div className="rounded-lg border border-[var(--stroke-1)] p-4">
              <p className="text-sm text-[var(--text-secondary)]">Total Requests</p>
              <p className="text-xl font-semibold">
                {requestsData.reduce((a, b) => a + b, 0).toLocaleString()}
              </p>
            </div>
            <div className="rounded-lg border border-[var(--stroke-1)] p-4">
              <p className="text-sm text-[var(--text-secondary)]">Uptime</p>
              <p className="text-xl font-semibold">99.99%</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
