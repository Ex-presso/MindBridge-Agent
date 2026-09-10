"use client";
import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Brain, MessageSquare, Sparkles, Trash2 } from "lucide-react";
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
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/authStore";
import { useMemory } from "@/hooks/useMemory";
import type { MemoryItem } from "@/types";

const PAGE_SIZE = 20;

type Category = "all" | "semantic" | "episodes";

const CATEGORY_TABS: { value: Category; label: string }[] = [
  { value: "all", label: "All" },
  { value: "semantic", label: "Facts" },
  { value: "episodes", label: "Episodes" },
];

const KIND_LABELS: Record<string, string> = {
  preference: "Preference",
  ongoing_concern: "Ongoing concern",
  trigger: "Trigger",
  helpful_strategy: "Helpful strategy",
  important_person: "Important person",
  goal: "Goal",
};

function itemCategory(item: MemoryItem): "semantic" | "episodes" | null {
  const ns = item.namespace[item.namespace.length - 1];
  return ns === "semantic" || ns === "episodes" ? ns : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}

function formatTimestamp(raw: string): string {
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function SemanticCard({ item }: { item: MemoryItem }) {
  const kind = asString(item.value.kind);
  const content = asString(item.value.content) ?? "(unreadable item)";
  return (
    <div className="rounded-xl border border-(--border) bg-(--background) p-4">
      <div className="flex items-center gap-2">
        <Sparkles className="h-3.5 w-3.5 shrink-0 text-(--primary)" aria-hidden="true" />
        <span className="rounded-full bg-(--muted) px-2 py-0.5 text-xs text-(--muted-foreground)">
          {(kind && KIND_LABELS[kind]) ?? "Fact"}
        </span>
        <span className="ml-auto text-xs text-(--muted-foreground)">{formatTimestamp(item.updated_at)}</span>
      </div>
      <p className="mt-2 text-sm leading-6">{content}</p>
    </div>
  );
}

function EpisodeCard({ item }: { item: MemoryItem }) {
  const summary = asString(item.value.summary) ?? "(unreadable episode)";
  const topics = Array.isArray(item.value.topics)
    ? item.value.topics.filter((t): t is string => typeof t === "string")
    : [];
  const messageCount = typeof item.value.message_count === "number" ? item.value.message_count : null;
  const crisis = item.value.crisis === true;
  return (
    <div className="rounded-xl border border-(--border) bg-(--background) p-4">
      <div className="flex items-center gap-2">
        <MessageSquare className="h-3.5 w-3.5 shrink-0 text-(--primary)" aria-hidden="true" />
        <span className="rounded-full bg-(--muted) px-2 py-0.5 text-xs text-(--muted-foreground)">Episode</span>
        {crisis && (
          <span className="flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-900/20 dark:text-amber-400">
            <AlertTriangle className="h-3 w-3" aria-hidden="true" /> Sensitive
          </span>
        )}
        <span className="ml-auto text-xs text-(--muted-foreground)">{formatTimestamp(item.updated_at)}</span>
      </div>
      <p className="mt-2 text-sm leading-6">{summary}</p>
      {(topics.length > 0 || messageCount !== null) && (
        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          {topics.map((topic) => (
            <span key={topic} className="rounded-full border border-(--border) px-2 py-0.5 text-xs text-(--muted-foreground)">
              {topic}
            </span>
          ))}
          {messageCount !== null && (
            <span className="ml-auto text-xs text-(--muted-foreground)">{messageCount} messages</span>
          )}
        </div>
      )}
    </div>
  );
}

export function MemorySection() {
  const userId = useAuthStore((state) => state.user?.id);
  const { memory, memoryLoading, setMemoryEnabled, clearMemory } = useMemory();

  const [category, setCategory] = useState<Category>("all");
  const [items, setItems] = useState<MemoryItem[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"toggle" | "clear" | null>(null);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);

  const loadPage = useCallback(
    async (offset: number, activeCategory: Category) => {
      if (!userId) return;
      setListLoading(true);
      setListError(null);
      try {
        const page = await api.memory.get(
          userId,
          PAGE_SIZE,
          offset,
          activeCategory === "all" ? undefined : activeCategory,
        );
        setItems((current) => (offset === 0 ? page.items : [...current, ...page.items]));
        setHasMore(page.has_more);
      } catch (err) {
        setListError(err instanceof Error ? err.message : "Failed to load memory");
      } finally {
        setListLoading(false);
      }
    },
    [userId],
  );

  useEffect(() => {
    setItems([]);
    void loadPage(0, category);
  }, [category, loadPage]);

  const handleToggle = async () => {
    setBusy("toggle");
    try {
      const status = await setMemoryEnabled(!(memory?.memory_enabled ?? false));
      if (status.memory_enabled && !status.system_enabled) {
        toast.error("Memory is disabled at the system level");
      } else {
        toast.success(status.memory_enabled ? "Memory enabled" : "Memory disabled");
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to update memory");
    } finally {
      setBusy(null);
    }
  };

  const handleClear = async () => {
    setBusy("clear");
    try {
      const result = await clearMemory();
      setClearDialogOpen(false);
      setItems([]);
      setHasMore(false);
      toast.success(
        result.deleted_items === 1
          ? "1 stored memory cleared"
          : `${result.deleted_items} stored memories cleared`,
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to clear memory");
    } finally {
      setBusy(null);
    }
  };

  const memoryEnabled = memory?.memory_enabled ?? false;
  const systemEnabled = memory?.system_enabled ?? true;

  return (
    <section aria-labelledby="memory-settings-title">
      <div className="mb-4">
        <h2 id="memory-settings-title" className="text-base font-semibold">Memory</h2>
        <p className="mt-1 text-xs leading-5 text-(--muted-foreground)">
          What MindBridge has saved to keep future conversations consistent. Everything here is inspectable and deletable.
        </p>
      </div>

      <div className="mb-6 rounded-xl border border-(--border) bg-(--background) p-4">
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex cursor-pointer items-center gap-2">
            <Switch
              checked={memoryEnabled}
              onCheckedChange={handleToggle}
              disabled={memoryLoading || busy !== null || !systemEnabled}
              aria-label="Long-term memory"
            />
            <span className="text-sm font-medium">Long-term memory</span>
          </label>
          <Dialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" variant="outline" className="ml-auto text-(--destructive)" disabled={busy !== null}>
                <Trash2 className="h-3.5 w-3.5" aria-hidden="true" /> Clear all
              </Button>
            </DialogTrigger>
            <DialogContent aria-describedby="clear-memory-description">
              <DialogHeader>
                <DialogTitle>Clear all stored memory?</DialogTitle>
                <p id="clear-memory-description" className="text-sm leading-6 text-(--muted-foreground)">
                  This permanently deletes everything MindBridge has saved about you and turns memory off. This action cannot be undone.
                </p>
              </DialogHeader>
              <DialogFooter>
                <Button variant="outline" onClick={() => setClearDialogOpen(false)} disabled={busy === "clear"}>
                  Cancel
                </Button>
                <Button variant="destructive" onClick={handleClear} disabled={busy === "clear"}>
                  {busy === "clear" ? "Clearing…" : "Clear and disable"}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
        <p className="mt-2 text-xs leading-5 text-(--muted-foreground)">
          {!systemEnabled
            ? "Memory is disabled at the system level. Anything already stored can still be cleared."
            : memoryEnabled
            ? "Saving is on. Facts are only stored when you state them explicitly, and nothing is saved from conversations flagged as sensitive."
            : "Saving is off. Existing items remain visible here until you clear them, but they are not used in conversations."}
        </p>
      </div>

      <div className="mb-4 flex items-center gap-1 rounded-lg bg-(--muted) p-1" role="tablist" aria-label="Memory category">
        {CATEGORY_TABS.map((tab) => (
          <button
            key={tab.value}
            role="tab"
            aria-selected={category === tab.value}
            onClick={() => setCategory(tab.value)}
            className={cn(
              "flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              category === tab.value
                ? "bg-(--background) shadow-sm"
                : "text-(--muted-foreground) hover:text-(--foreground)",
            )}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {listError ? (
        <div className="rounded-xl border border-(--border) p-8 text-center text-sm text-(--muted-foreground)">
          <p>{listError}</p>
          <Button size="sm" variant="outline" className="mt-3" onClick={() => loadPage(0, category)}>
            Retry
          </Button>
        </div>
      ) : items.length === 0 && !listLoading ? (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-(--border) p-10 text-center">
          <Brain className="h-8 w-8 text-(--muted-foreground) opacity-30" aria-hidden="true" />
          <p className="text-sm font-medium">Nothing stored yet</p>
          <p className="max-w-sm text-xs leading-5 text-(--muted-foreground)">
            {memoryEnabled
              ? "As you chat, MindBridge saves explicitly stated facts and short conversation synopses. They will appear here."
              : "Turn on long-term memory and MindBridge will start saving context between conversations."}
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {items.map((item) => {
            const key = `${item.namespace.join("/")}/${item.key}`;
            return itemCategory(item) === "episodes"
              ? <EpisodeCard key={key} item={item} />
              : <SemanticCard key={key} item={item} />;
          })}
          {listLoading && (
            <div className="flex justify-center py-4">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-(--primary) border-t-transparent" />
            </div>
          )}
          {hasMore && !listLoading && (
            <Button variant="outline" size="sm" className="mx-auto" onClick={() => loadPage(items.length, category)}>
              Load more
            </Button>
          )}
        </div>
      )}

      {items.length === 0 && listLoading && (
        <div className="flex justify-center py-10">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-(--primary) border-t-transparent" />
        </div>
      )}
    </section>
  );
}
