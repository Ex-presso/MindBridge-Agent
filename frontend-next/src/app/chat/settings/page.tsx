"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Brain, Check, Eye, EyeOff, Key, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { useApiKeys } from "@/hooks/useApiKeys";
import { useAuth } from "@/hooks/useAuth";
import { useMemory } from "@/hooks/useMemory";
import { PROVIDER_INFO } from "@/types";

const PROVIDERS = ["openai", "anthropic", "google_genai", "openai_compatible", "anthropic_compatible"];
const MEMORY_ERROR_TOAST_ID = "memory-settings-error";

function ProviderCard({
  provider,
  existingKey,
  onSave,
  onDelete,
}: {
  provider: string;
  existingKey: { api_key_masked: string; base_url: string | null; model_id: string | null; display_name: string | null } | null;
  onSave: (data: { provider: string; api_key: string; base_url?: string; model_id?: string; display_name?: string }) => Promise<unknown>;
  onDelete: (provider: string) => Promise<void>;
}) {
  const info = PROVIDER_INFO[provider];
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState(existingKey?.base_url ?? "");
  const [modelId, setModelId] = useState(existingKey?.model_id ?? "");
  const [displayName, setDisplayName] = useState(existingKey?.display_name ?? "");
  const [showKey, setShowKey] = useState(false);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (!apiKey.trim()) {
      toast.error("API key is required");
      return;
    }
    setSaving(true);
    try {
      await onSave({
        provider,
        api_key: apiKey,
        base_url: info.needsBaseUrl ? baseUrl || undefined : undefined,
        model_id: info.needsModelId ? modelId || undefined : undefined,
        display_name: displayName || undefined,
      });
      setApiKey("");
      toast.success(`${info.label} key saved`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    try {
      await onDelete(provider);
      setBaseUrl("");
      setModelId("");
      setDisplayName("");
      toast.success(`${info.label} key removed`);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove");
    }
  };

  return (
    <div className="rounded-xl border border-[--border] bg-[--background] p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-medium text-sm">{info.label}</h3>
          <p className="text-xs text-[--muted-foreground]">{info.description}</p>
        </div>
        {existingKey ? (
          <span className="flex items-center gap-1 text-xs text-green-600 bg-green-50 dark:bg-green-900/20 px-2 py-0.5 rounded-full">
            <Check className="h-3 w-3" /> Configured
          </span>
        ) : (
          <span className="text-xs text-[--muted-foreground] bg-[--muted] px-2 py-0.5 rounded-full">Not configured</span>
        )}
      </div>

      {existingKey && (
        <p className="text-xs text-[--muted-foreground] mb-3 font-mono">{existingKey.api_key_masked}</p>
      )}

      <div className="flex flex-col gap-3">
        <div>
          <Label className="text-xs">API Key {existingKey ? "(enter new to replace)" : ""}</Label>
          <div className="relative mt-1">
            <Input
              type={showKey ? "text" : "password"}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={existingKey ? "Enter new key to replace…" : "Enter API key…"}
              className="pr-8 font-mono text-xs"
            />
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-[--muted-foreground] hover:text-[--foreground]"
            >
              {showKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
            </button>
          </div>
        </div>

        {info.needsBaseUrl && (
          <div>
            <Label className="text-xs">Base URL</Label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.example.com/v1"
              className="mt-1 text-xs"
            />
          </div>
        )}

        {info.needsModelId && (
          <div>
            <Label className="text-xs">Model Name</Label>
            <Input
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="e.g., llama3, mistral, etc."
              className="mt-1 text-xs"
            />
          </div>
        )}

        {info.needsBaseUrl && (
          <div>
            <Label className="text-xs">Display Name (optional)</Label>
            <Input
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="e.g., My Local Ollama"
              className="mt-1 text-xs"
            />
          </div>
        )}

        <div className="flex gap-2 mt-1">
          <Button size="sm" onClick={handleSave} disabled={saving || !apiKey.trim()}>
            <Key className="h-3.5 w-3.5 mr-1" />
            {saving ? "Saving..." : existingKey ? "Update Key" : "Save Key"}
          </Button>
          {existingKey && (
            <Button size="sm" variant="outline" onClick={handleDelete} className="text-[--destructive]">
              <Trash2 className="h-3.5 w-3.5 mr-1" /> Remove
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const router = useRouter();
  const { deleteAccount } = useAuth();
  const { keys, saveKey, deleteKey, keysLoading } = useApiKeys();
  const {
    memory,
    memoryLoading,
    memoryError,
    setMemoryEnabled,
    clearMemory,
  } = useMemory();
  const [memoryBusy, setMemoryBusy] = useState<"toggle" | "clear" | null>(null);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);
  const [accountDialogOpen, setAccountDialogOpen] = useState(false);
  const [accountPassword, setAccountPassword] = useState("");
  const [accountBusy, setAccountBusy] = useState(false);

  useEffect(() => {
    if (memoryError) {
      toast.error(
        memoryError instanceof Error ? memoryError.message : "Failed to load memory settings",
        { id: MEMORY_ERROR_TOAST_ID },
      );
    }
  }, [memoryError]);

  const memoryUnknown = !memoryLoading && (!memory || Boolean(memoryError));
  const memoryEnabled = memoryUnknown ? null : (memory?.memory_enabled ?? null);
  const systemEnabled = memoryUnknown ? null : (memory?.system_enabled ?? null);
  const storedMemoryCount = memory && !memoryUnknown
    ? `${memory.items.length}${memory.has_more ? "+" : ""}`
    : "—";

  const handleMemoryToggle = async () => {
    // When inspection is unavailable, only offer the privacy-safe direction.
    const nextEnabled = memoryEnabled === null ? false : !memoryEnabled;
    setMemoryBusy("toggle");
    try {
      const status = await setMemoryEnabled(nextEnabled);
      if (nextEnabled && !status.system_enabled) {
        toast.error("Memory is disabled at the system level");
      } else {
        toast.success(status.memory_enabled ? "Memory enabled" : "Memory disabled");
      }
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to update memory settings",
        { id: MEMORY_ERROR_TOAST_ID },
      );
    } finally {
      setMemoryBusy(null);
    }
  };

  const handleClearMemory = async () => {
    setMemoryBusy("clear");
    try {
      const result = await clearMemory();
      setClearDialogOpen(false);
      toast.success(
        result.deleted_items === 1
          ? "1 stored memory cleared"
          : `${result.deleted_items} stored memories cleared`,
      );
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to clear memory",
        { id: MEMORY_ERROR_TOAST_ID },
      );
    } finally {
      setMemoryBusy(null);
    }
  };

  const handleDeleteAccount = async () => {
    if (!accountPassword) return;
    setAccountBusy(true);
    try {
      await deleteAccount(accountPassword);
      toast.success("Account deleted");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete account");
      setAccountBusy(false);
    }
  };

  const keyMap: Record<string, typeof keys[number] | null> = {};
  for (const provider of PROVIDERS) {
    keyMap[provider] = keys.find((k) => k.provider === provider) ?? null;
  }

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="max-w-2xl mx-auto w-full px-4 py-6">
        <div className="flex items-center gap-3 mb-6">
          <Button variant="ghost" size="icon" onClick={() => router.push("/chat")} className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-lg font-semibold">API Keys</h1>
            <p className="text-xs text-[--muted-foreground]">Configure your LLM provider keys. Keys are encrypted and stored securely.</p>
          </div>
        </div>

        {keysLoading ? (
          <div className="flex justify-center py-12">
            <div className="h-6 w-6 rounded-full border-2 border-[--primary] border-t-transparent animate-spin" />
          </div>
        ) : (
          <div className="flex flex-col gap-4">
            {PROVIDERS.map((provider) => (
              <ProviderCard
                key={provider}
                provider={provider}
                existingKey={keyMap[provider]}
                onSave={saveKey}
                onDelete={deleteKey}
              />
            ))}
          </div>
        )}

        <Separator className="my-6" />
        <p className="text-xs text-[--muted-foreground] text-center">
          Your keys are encrypted at rest and never shared. They are used only to make LLM API calls on your behalf.
        </p>

        <Separator className="my-8" />

        <section aria-labelledby="memory-settings-title">
          <div className="mb-4 flex items-start gap-3">
            <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[--muted] text-[--muted-foreground]">
              <Brain className="h-4 w-4" aria-hidden="true" />
            </div>
            <div>
              <h2 id="memory-settings-title" className="text-base font-semibold">Memory</h2>
              <p className="text-xs leading-5 text-[--muted-foreground]">
                Control whether MindBridge can retain useful context between conversations.
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-[--border] bg-[--background] p-5" aria-busy={memoryLoading || memoryBusy !== null}>
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-medium">Long-term memory</p>
                  {memoryLoading ? (
                    <span className="text-xs text-[--muted-foreground]">Loading…</span>
                  ) : memoryUnknown ? (
                    <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-900/20 dark:text-amber-400">Unable to load</span>
                  ) : !systemEnabled ? (
                    <span className="rounded-full bg-[--muted] px-2 py-0.5 text-xs text-[--muted-foreground]">Unavailable</span>
                  ) : memoryEnabled ? (
                    <span className="flex items-center gap-1 rounded-full bg-green-50 px-2 py-0.5 text-xs text-green-700 dark:bg-green-900/20 dark:text-green-400">
                      <Check className="h-3 w-3" aria-hidden="true" /> Enabled
                    </span>
                  ) : (
                    <span className="rounded-full bg-[--muted] px-2 py-0.5 text-xs text-[--muted-foreground]">Disabled</span>
                  )}
                </div>
                <p className="mt-1 max-w-lg text-xs leading-5 text-[--muted-foreground]">
                  {memoryUnknown
                    ? "We could not verify your current memory status or stored item count. You can still turn memory off or clear it below."
                    : systemEnabled
                    ? memoryEnabled
                      ? "MindBridge can save and use context to make future conversations more consistent. Disabling stops use and new storage without deleting existing items."
                      : "Memory is off by default. Enabling allows MindBridge to save and use context between conversations. Existing items remain until you clear them."
                    : "Long-term memory has been disabled at the system level. You can still clear anything already stored."}
                </p>
              </div>

              <div className="shrink-0 text-left sm:text-right">
                <p className="text-lg font-semibold tabular-nums" aria-label={`${storedMemoryCount} stored memories`}>
                  {storedMemoryCount}
                </p>
                <p className="text-xs text-[--muted-foreground]">stored memories</p>
              </div>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-[--border] pt-4">
              <Button
                size="sm"
                variant={memoryEnabled ? "outline" : "default"}
                onClick={handleMemoryToggle}
                disabled={memoryLoading || memoryBusy !== null || (systemEnabled === false && memoryEnabled === false)}
              >
                {memoryBusy === "toggle"
                  ? "Updating…"
                  : memoryEnabled === null
                    ? "Turn memory off"
                  : memoryEnabled
                    ? "Disable memory"
                    : "Enable memory"}
              </Button>

              <Dialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
                <DialogTrigger asChild>
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-[--destructive]"
                    disabled={memoryLoading || memoryBusy !== null}
                  >
                    <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                    Clear and disable
                  </Button>
                </DialogTrigger>
                <DialogContent aria-describedby="clear-memory-description">
                  <DialogHeader>
                    <DialogTitle>Clear all stored memory?</DialogTitle>
                    <p id="clear-memory-description" className="text-sm leading-6 text-[--muted-foreground]">
                      This permanently deletes the context MindBridge has saved about you and turns memory off. This action cannot be undone.
                    </p>
                  </DialogHeader>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      onClick={() => setClearDialogOpen(false)}
                      disabled={memoryBusy === "clear"}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={handleClearMemory}
                      disabled={memoryBusy === "clear"}
                    >
                      {memoryBusy === "clear" ? "Clearing…" : "Clear and disable"}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </section>

        <Separator className="my-8" />

        <section aria-labelledby="account-settings-title">
          <div className="mb-4">
            <h2 id="account-settings-title" className="text-base font-semibold text-[--destructive]">Danger zone</h2>
            <p className="mt-1 text-xs leading-5 text-[--muted-foreground]">
              Permanently remove your conversations, checkpoints, API keys, and all stored memory.
            </p>
          </div>

          <div className="rounded-xl border border-[--destructive]/40 bg-[--background] p-5">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-sm font-medium">Delete account</p>
                <p className="mt-1 text-xs leading-5 text-[--muted-foreground]">
                  This action cannot be undone. External memory and conversation checkpoints are cleared before your account row is removed.
                </p>
              </div>

              <Dialog open={accountDialogOpen} onOpenChange={(open) => {
                setAccountDialogOpen(open);
                if (!open && !accountBusy) setAccountPassword("");
              }}>
                <DialogTrigger asChild>
                  <Button size="sm" variant="destructive" className="shrink-0">
                    <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                    Delete account
                  </Button>
                </DialogTrigger>
                <DialogContent aria-describedby="delete-account-description">
                  <DialogHeader>
                    <DialogTitle>Delete your account permanently?</DialogTitle>
                    <p id="delete-account-description" className="text-sm leading-6 text-[--muted-foreground]">
                      Enter your password to confirm. If external cleanup is temporarily unavailable, your account stays disabled and the deletion can be retried.
                    </p>
                  </DialogHeader>
                  <div>
                    <Label htmlFor="delete-account-password">Password</Label>
                    <Input
                      id="delete-account-password"
                      type="password"
                      autoComplete="current-password"
                      value={accountPassword}
                      onChange={(event) => setAccountPassword(event.target.value)}
                      disabled={accountBusy}
                      className="mt-1"
                    />
                  </div>
                  <DialogFooter>
                    <Button
                      variant="outline"
                      onClick={() => setAccountDialogOpen(false)}
                      disabled={accountBusy}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={handleDeleteAccount}
                      disabled={accountBusy || !accountPassword}
                    >
                      {accountBusy ? "Deleting…" : "Delete permanently"}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
