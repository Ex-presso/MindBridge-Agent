"use client";
import { usePathname } from "next/navigation";
import { useRouter } from "next/navigation";
import { Plus, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ConversationItem } from "./ConversationItem";
import { useConversations } from "@/hooks/useConversations";
import { formatDate } from "@/lib/utils";
import type { Conversation } from "@/types";

function groupByDate(conversations: Conversation[]): Record<string, Conversation[]> {
  const groups: Record<string, Conversation[]> = {};
  for (const conv of conversations) {
    const label = formatDate(conv.updated_at);
    if (!groups[label]) groups[label] = [];
    groups[label].push(conv);
  }
  return groups;
}

const DATE_ORDER = ["Today", "Yesterday", "This Week", "Older"];

export function Sidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const { conversations, deleteConversation, renameConversation } = useConversations();

  const activeId = pathname.startsWith("/chat/") ? pathname.slice(6) : null;
  const groups = groupByDate(conversations);

  return (
    <aside className="flex flex-col h-full" style={{ width: "var(--sidebar-width)", minWidth: "var(--sidebar-width)" }}>
      <div className="p-3">
        <Button
          variant="outline"
          className="w-full justify-start gap-2 text-sm"
          onClick={() => router.push("/chat")}
        >
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </div>

      <Separator />

      <ScrollArea className="flex-1 px-2 py-2">
        {conversations.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-8 text-center text-xs text-[--muted-foreground]">
            <MessageSquare className="h-8 w-8 opacity-30" />
            <p>No conversations yet</p>
          </div>
        )}

        {DATE_ORDER.filter((label) => groups[label]?.length).map((label) => (
          <div key={label} className="mb-3">
            <p className="px-2 py-1 text-xs font-medium text-[--muted-foreground]">{label}</p>
            {groups[label].map((conv) => (
              <ConversationItem
                key={conv.id}
                conversation={conv}
                isActive={conv.id === activeId}
                onDelete={deleteConversation}
                onRename={renameConversation}
              />
            ))}
          </div>
        ))}
      </ScrollArea>
    </aside>
  );
}
