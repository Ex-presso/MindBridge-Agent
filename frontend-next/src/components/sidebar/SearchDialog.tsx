"use client";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { Clock, MessageSquare, Search, X } from "lucide-react";
import { useConversations } from "@/hooks/useConversations";

const HISTORY_KEY = "mb-search-history";
const HISTORY_MAX = 8;

function loadHistory(): string[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((q): q is string => typeof q === "string") : [];
  } catch {
    return [];
  }
}

function saveHistory(history: string[]) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // Private mode or blocked storage; search still works without history.
  }
}

export function SearchDialog({ open, onOpenChange }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const { conversations } = useConversations();
  const [query, setQuery] = useState("");
  const [history, setHistory] = useState<string[]>([]);
  const [prevOpen, setPrevOpen] = useState(open);

  // Reset the query and refresh history each time the dialog opens.
  if (open !== prevOpen) {
    setPrevOpen(open);
    if (open) {
      setQuery("");
      setHistory(loadHistory());
    }
  }

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return conversations.filter((c) => c.title.toLowerCase().includes(q)).slice(0, 10);
  }, [conversations, query]);

  const rememberQuery = (q: string) => {
    const trimmed = q.trim();
    if (!trimmed) return;
    const next = [trimmed, ...history.filter((item) => item !== trimmed)].slice(0, HISTORY_MAX);
    setHistory(next);
    saveHistory(next);
  };

  const removeQuery = (q: string) => {
    const next = history.filter((item) => item !== q);
    setHistory(next);
    saveHistory(next);
  };

  const openConversation = (id: string) => {
    rememberQuery(query);
    onOpenChange(false);
    router.push(`/chat/${id}`);
  };

  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/40" />
        <DialogPrimitive.Content
          className="fixed left-1/2 top-[18%] z-50 w-full max-w-lg -translate-x-1/2 rounded-xl border border-(--border) bg-(--background) shadow-lg outline-none"
          aria-describedby={undefined}
        >
          <DialogPrimitive.Title className="sr-only">Search conversations</DialogPrimitive.Title>

          <div className="flex items-center gap-2 border-b border-(--border) px-3">
            <Search className="h-4 w-4 shrink-0 text-(--muted-foreground)" aria-hidden="true" />
            <input
              autoFocus
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") rememberQuery(query);
              }}
              placeholder="Search conversations..."
              className="h-11 w-full bg-transparent text-sm outline-none placeholder:text-(--muted-foreground)"
            />
            <DialogPrimitive.Close className="rounded-md p-1 text-(--muted-foreground) transition-colors hover:bg-(--muted) hover:text-(--foreground)">
              <X className="h-4 w-4" aria-hidden="true" />
              <span className="sr-only">Close search</span>
            </DialogPrimitive.Close>
          </div>

          <div className="max-h-72 overflow-y-auto p-2">
            {query.trim() ? (
              results.length > 0 ? (
                results.map((conv) => (
                  <button
                    key={conv.id}
                    onClick={() => openConversation(conv.id)}
                    className="flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left text-sm transition-colors hover:bg-(--muted)"
                  >
                    <MessageSquare className="h-3.5 w-3.5 shrink-0 text-(--muted-foreground)" aria-hidden="true" />
                    <span className="truncate">{conv.title}</span>
                  </button>
                ))
              ) : (
                <p className="px-2 py-6 text-center text-xs text-(--muted-foreground)">
                  No conversation titles match. Full-text message search is coming soon.
                </p>
              )
            ) : history.length > 0 ? (
              <>
                <p className="px-2 py-1 text-xs font-medium text-(--muted-foreground)">Recent searches</p>
                {history.map((item) => (
                  <div
                    key={item}
                    className="group flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors hover:bg-(--muted)"
                  >
                    <Clock className="h-3.5 w-3.5 shrink-0 text-(--muted-foreground)" aria-hidden="true" />
                    <button className="flex-1 truncate text-left" onClick={() => setQuery(item)}>
                      {item}
                    </button>
                    <button
                      onClick={() => removeQuery(item)}
                      className="rounded p-0.5 text-(--muted-foreground) opacity-0 transition-opacity hover:text-(--foreground) group-hover:opacity-100"
                      aria-label={`Remove ${item} from history`}
                    >
                      <X className="h-3.5 w-3.5" aria-hidden="true" />
                    </button>
                  </div>
                ))}
              </>
            ) : (
              <p className="px-2 py-6 text-center text-xs text-(--muted-foreground)">
                Search your conversations by title.
              </p>
            )}
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
