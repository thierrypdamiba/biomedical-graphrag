"use client";

import * as React from "react";
import { Plus, Copy, Eye, EyeOff, Trash2, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Chip } from "@/components/ui/chip";
import { cn } from "@/lib/utils";

const sampleKeys = [
  {
    id: "key-001",
    name: "Production API Key",
    prefix: "qdrant_prod_",
    created: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
    lastUsed: new Date(Date.now() - 3600000),
    scopes: ["read", "write"],
  },
  {
    id: "key-002",
    name: "Development Key",
    prefix: "qdrant_dev_",
    created: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    lastUsed: new Date(Date.now() - 86400000),
    scopes: ["read", "write", "admin"],
  },
  {
    id: "key-003",
    name: "Read-only Analytics",
    prefix: "qdrant_ro_",
    created: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000),
    lastUsed: null,
    scopes: ["read"],
  },
];

export default function ApiKeysPage() {
  const [showCreateModal, setShowCreateModal] = React.useState(false);
  const [newKeyName, setNewKeyName] = React.useState("");
  const [newKeyScopes, setNewKeyScopes] = React.useState<string[]>(["read"]);
  const [createdKey, setCreatedKey] = React.useState<string | null>(null);
  const [copiedKeyId, setCopiedKeyId] = React.useState<string | null>(null);

  const handleCreateKey = () => {
    // Simulate key creation
    setCreatedKey(
      `qdrant_${newKeyName.toLowerCase().replace(/\s+/g, "_")}_${crypto
        .randomUUID()
        .slice(0, 16)}`
    );
  };

  const handleCopy = (text: string, keyId: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKeyId(keyId);
    setTimeout(() => setCopiedKeyId(null), 2000);
  };

  const toggleScope = (scope: string) => {
    setNewKeyScopes((prev) =>
      prev.includes(scope) ? prev.filter((s) => s !== scope) : [...prev, scope]
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">API Keys</h1>
          <p className="mt-1 text-[var(--text-secondary)]">
            Manage access keys for your cluster
          </p>
        </div>
        <Button variant="primary" onClick={() => setShowCreateModal(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Create Key
        </Button>
      </div>

      {/* Keys Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="border-b border-[var(--stroke-1)] bg-[var(--bg-2)]">
                <tr className="text-left text-sm">
                  <th className="px-4 py-3 font-medium">Name</th>
                  <th className="px-4 py-3 font-medium">Key Prefix</th>
                  <th className="px-4 py-3 font-medium">Created</th>
                  <th className="px-4 py-3 font-medium">Last Used</th>
                  <th className="px-4 py-3 font-medium">Scopes</th>
                  <th className="px-4 py-3 font-medium w-32"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--stroke-1)]">
                {sampleKeys.map((key) => (
                  <tr key={key.id} className="hover:bg-[var(--bg-2)]">
                    <td className="px-4 py-3 font-medium">{key.name}</td>
                    <td className="px-4 py-3">
                      <code className="text-sm text-[var(--text-secondary)]">
                        {key.prefix}***
                      </code>
                    </td>
                    <td className="px-4 py-3 text-sm text-[var(--text-secondary)]">
                      {key.created.toLocaleDateString()}
                    </td>
                    <td className="px-4 py-3 text-sm text-[var(--text-secondary)]">
                      {key.lastUsed ? key.lastUsed.toLocaleDateString() : "Never"}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex gap-1">
                        {key.scopes.map((scope) => (
                          <Chip
                            key={scope}
                            variant={
                              scope === "admin"
                                ? "amaranth"
                                : scope === "write"
                                ? "violet"
                                : "default"
                            }
                          >
                            {scope}
                          </Chip>
                        ))}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleCopy(key.prefix + "***", key.id)}
                        >
                          {copiedKeyId === key.id ? (
                            <Check className="h-4 w-4 text-[var(--teal)]" />
                          ) : (
                            <Copy className="h-4 w-4" />
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-[var(--amaranth)]"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Create Key Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <Card className="w-full max-w-md shadow-[var(--modal-shadow)]">
            <CardHeader>
              <CardTitle>Create API Key</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {!createdKey ? (
                <>
                  <div>
                    <label className="mb-1.5 block text-sm font-medium">
                      Key Name
                    </label>
                    <Input
                      value={newKeyName}
                      onChange={(e) => setNewKeyName(e.target.value)}
                      placeholder="e.g., Production API Key"
                    />
                  </div>
                  <div>
                    <label className="mb-2 block text-sm font-medium">Scopes</label>
                    <div className="flex flex-wrap gap-2">
                      {["read", "write", "admin"].map((scope) => (
                        <button
                          key={scope}
                          onClick={() => toggleScope(scope)}
                          className={cn(
                            "rounded-lg border px-3 py-1.5 text-sm transition-colors",
                            newKeyScopes.includes(scope)
                              ? "border-[var(--violet)] bg-[rgba(133,71,255,0.18)] text-[var(--violet)]"
                              : "border-[var(--stroke-1)] hover:bg-[var(--bg-2)]"
                          )}
                        >
                          {scope}
                        </button>
                      ))}
                    </div>
                    {newKeyScopes.includes("admin") && (
                      <p className="mt-2 text-sm text-[var(--amaranth)]">
                        Warning: Admin scope grants full cluster access
                      </p>
                    )}
                  </div>
                  <div className="flex gap-3 pt-2">
                    <Button
                      variant="primary"
                      className="flex-1"
                      onClick={handleCreateKey}
                      disabled={!newKeyName.trim()}
                    >
                      Create Key
                    </Button>
                    <Button
                      variant="secondary"
                      onClick={() => {
                        setShowCreateModal(false);
                        setNewKeyName("");
                        setNewKeyScopes(["read"]);
                      }}
                    >
                      Cancel
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="rounded-lg border border-[var(--teal)] bg-[rgba(3,133,133,0.1)] p-4">
                    <p className="mb-2 text-sm font-medium text-[var(--teal)]">
                      Key created successfully!
                    </p>
                    <p className="text-sm text-[var(--text-secondary)]">
                      Copy this key now. You won&apos;t be able to see it again.
                    </p>
                  </div>
                  <div className="flex items-center gap-2 rounded-lg bg-[var(--bg-2)] p-3">
                    <code className="flex-1 break-all text-sm">{createdKey}</code>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleCopy(createdKey, "new")}
                    >
                      {copiedKeyId === "new" ? (
                        <Check className="h-4 w-4 text-[var(--teal)]" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  <Button
                    variant="primary"
                    className="w-full"
                    onClick={() => {
                      setShowCreateModal(false);
                      setNewKeyName("");
                      setNewKeyScopes(["read"]);
                      setCreatedKey(null);
                    }}
                  >
                    Done
                  </Button>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
