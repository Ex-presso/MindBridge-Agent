"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Check, Eye, EyeOff, Key, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { useApiKeys } from "@/hooks/useApiKeys";
import { PROVIDER_INFO } from "@/types";

const PROVIDERS = ["openai", "anthropic", "google_genai", "openai_compatible", "anthropic_compatible"];

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
  const { keys, saveKey, deleteKey, keysLoading } = useApiKeys();

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
      </div>
    </div>
  );
}
