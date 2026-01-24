"use client";

import * as React from "react";
import { Sun, Moon, Monitor } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useAppStore, type Theme, type VectorMode } from "@/store/app-store";
import { cn } from "@/lib/utils";

export default function SettingsPage() {
  const { theme, setTheme, vectorMode, setVectorMode, topK, setTopK, traceDrawerOpen, toggleTraceDrawer } =
    useAppStore();

  const themes: { value: Theme; label: string; icon: React.ReactNode }[] = [
    { value: "light", label: "Light", icon: <Sun className="h-4 w-4" /> },
    { value: "dark", label: "Dark", icon: <Moon className="h-4 w-4" /> },
    { value: "system", label: "System", icon: <Monitor className="h-4 w-4" /> },
  ];

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Settings</h1>
        <p className="mt-1 text-[var(--text-secondary)]">
          Customize your demo console experience
        </p>
      </div>

      {/* Appearance */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Appearance</CardTitle>
          <CardDescription>Choose your preferred theme</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-3">
            {themes.map((t) => (
              <button
                key={t.value}
                onClick={() => setTheme(t.value)}
                className={cn(
                  "flex flex-1 flex-col items-center gap-2 rounded-lg border p-4 transition-colors",
                  theme === t.value
                    ? "border-[var(--violet)] bg-[rgba(133,71,255,0.1)]"
                    : "border-[var(--stroke-1)] hover:bg-[var(--bg-2)]"
                )}
              >
                {t.icon}
                <span className="text-sm font-medium">{t.label}</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Default Search Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Search Defaults</CardTitle>
          <CardDescription>
            Configure default settings for search queries
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="mb-1.5 block text-sm font-medium">
              Default Vector Mode
            </label>
            <Select
              value={vectorMode}
              onValueChange={(v) => setVectorMode(v as VectorMode)}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="dense">Dense</SelectItem>
                <SelectItem value="sparse">Sparse</SelectItem>
                <SelectItem value="hybrid">Hybrid (Recommended)</SelectItem>
              </SelectContent>
            </Select>
            <p className="mt-1.5 text-sm text-[var(--text-tertiary)]">
              Hybrid search combines dense and sparse vectors for best results
            </p>
          </div>

          <div>
            <label className="mb-1.5 block text-sm font-medium">
              Default top_k
            </label>
            <Select
              value={topK.toString()}
              onValueChange={(v) => setTopK(parseInt(v))}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="10">10 results</SelectItem>
                <SelectItem value="20">20 results</SelectItem>
                <SelectItem value="50">50 results</SelectItem>
                <SelectItem value="100">100 results</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center justify-between rounded-lg border border-[var(--stroke-1)] p-4">
            <div>
              <p className="font-medium">Show trace by default</p>
              <p className="text-sm text-[var(--text-secondary)]">
                Automatically open the trace drawer for new queries
              </p>
            </div>
            <button
              onClick={toggleTraceDrawer}
              className={cn(
                "relative h-6 w-11 rounded-full transition-colors",
                traceDrawerOpen ? "bg-[var(--violet)]" : "bg-[var(--bg-2)]"
              )}
            >
              <span
                className={cn(
                  "absolute top-1 h-4 w-4 rounded-full bg-white transition-transform",
                  traceDrawerOpen ? "left-6" : "left-1"
                )}
              />
            </button>
          </div>
        </CardContent>
      </Card>

      {/* Keyboard Shortcuts */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Keyboard Shortcuts</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            {[
              { keys: "⌘K", action: "Open command palette" },
              { keys: "⌘/", action: "Toggle trace drawer" },
              { keys: "⌘Enter", action: "Run search / Execute request" },
              { keys: "Esc", action: "Close drawer / modal" },
            ].map((shortcut, i) => (
              <div key={i} className="flex items-center justify-between">
                <span className="text-[var(--text-secondary)]">
                  {shortcut.action}
                </span>
                <kbd className="rounded bg-[var(--bg-2)] px-2 py-1 font-mono text-xs">
                  {shortcut.keys}
                </kbd>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
