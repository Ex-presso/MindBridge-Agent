"use client";
import { useCallback, useEffect, useState } from "react";
import { Brain, Sparkles, Trash2 } from "lucide-react";
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
import { useAuthStore } from "@/stores/authStore";
import { useMemory } from "@/hooks/useMemory";
import type { MemoryItem } from "@/types";

const PAGE_SIZE = 20;

function formatTimestamp(raw: string | null): string {
  if (!raw) return "";
  const date = new Date(raw);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function MemoryCard({
  item,
  deleting,
  onDelete,
}: {
  item: MemoryItem;
  deleting: boolean;
  onDelete: () => void;
}) {
  return (
    <div className="rounded-xl border border-(--border) bg-(--background) p-4">
      <div className="flex items-center gap-2">
        <Sparkles className="h-3.5 w-3.5 shrink-0 text-(--primary)" aria-hidden="true" />
        <span className="text-xs text-(--muted-foreground)">{formatTimestamp(item.created_at)}</span>
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto h-7 px-2 text-(--muted-foreground) hover:text-(--destructive)"
          onClick={onDelete}
          disabled={deleting}
          aria-label="Delete this memory"
        >
          <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      </div>
      <p className="mt-2 text-sm leading-6">{item.text}</p>
    </div>
  );
}

export function MemorySection() {
  const userId = useAuthStore((state) => state.user?.id);
  const { memory, memoryLoading, setMemoryEnabled, clearMemory } = useMemory();

  const [items, setItems] = useState<MemoryItem[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);
  const [busy, setBusy] = useState<"toggle" | "clear" | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [clearDialogOpen, setClearDialogOpen] = useState(false);

  const loadPage = useCallback(
    async (offset: number) => {
      if (!userId) return;
      setListLoading(true);
      setListError(null);
      try {
        const page = await api.memory.get(userId, PAGE_SIZE, offset);
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
    void loadPage(0);
  }, [loadPage]);

  const handleDelete = async (id: string) => {
    if (!userId) return;
    setDeletingId(id);
    try {
      await api.memory.remove(userId, id);
      setItems((current) => current.filter((item) => item.id !== id));
      toast.success("Memory deleted");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete memory");
    } finally {
      setDeletingId(null);
    }
  };

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
            ? "Saving is on. Memories are drawn from what you say, never from the assistant's replies, and nothing is saved from conversations flagged as sensitive."
            : "Saving is off. Existing items remain visible here until you clear them, but they are not used in conversations."}
        </p>
      </div>

      {listError ? (
        <div className="rounded-xl border border-(--border) p-8 text-center text-sm text-(--muted-foreground)">
          <p>{listError}</p>
          <Button size="sm" variant="outline" className="mt-3" onClick={() => loadPage(0)}>
            Retry
          </Button>
        </div>
      ) : items.length === 0 && !listLoading ? (
        <div className="flex flex-col items-center gap-2 rounded-xl border border-(--border) p-10 text-center">
          <Brain className="h-8 w-8 text-(--muted-foreground) opacity-30" aria-hidden="true" />
          <p className="text-sm font-medium">Nothing stored yet</p>
          <p className="max-w-sm text-xs leading-5 text-(--muted-foreground)">
            {memoryEnabled
              ? "As you chat, MindBridge saves short facts about what you share. They will appear here."
              : "Turn on long-term memory and MindBridge will start saving context between conversations."}
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          {items.map((item) => (
            <MemoryCard
              key={item.id}
              item={item}
              deleting={deletingId === item.id}
              onDelete={() => handleDelete(item.id)}
            />
          ))}
          {listLoading && (
            <div className="flex justify-center py-4">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-(--primary) border-t-transparent" />
            </div>
          )}
          {hasMore && !listLoading && (
            <Button variant="outline" size="sm" className="mx-auto" onClick={() => loadPage(items.length)}>
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
