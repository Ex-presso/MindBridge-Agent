"use client";
import { useState } from "react";
import { usePathname } from "next/navigation";
import { useRouter } from "next/navigation";
import { LogOut, MessageSquare, Search, Settings, SquarePen } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ConversationItem } from "./ConversationItem";
import { SearchDialog } from "./SearchDialog";
import { useAuth } from "@/hooks/useAuth";
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
  const { user, logout } = useAuth();
  const { conversations, deleteConversation, renameConversation } = useConversations();
  const [searchOpen, setSearchOpen] = useState(false);

  const activeId = pathname.startsWith("/chat/") ? pathname.slice(6) : null;
  const groups = groupByDate(conversations);

  const initials = user?.display_name
    ? user.display_name.slice(0, 2).toUpperCase()
    : user?.email.slice(0, 2).toUpperCase() ?? "MB";

  return (
    <aside className="flex flex-col h-full" style={{ width: "var(--sidebar-width)", minWidth: "var(--sidebar-width)" }}>
      <div className="flex flex-col gap-0.5 p-2 pt-3">
        <button
          onClick={() => router.push("/chat")}
          className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-(--foreground) transition-colors hover:bg-(--border)/60"
        >
          <SquarePen className="h-4 w-4 shrink-0 text-(--muted-foreground)" aria-hidden="true" />
          New Chat
        </button>
        <button
          onClick={() => setSearchOpen(true)}
          className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-(--foreground) transition-colors hover:bg-(--border)/60"
        >
          <Search className="h-4 w-4 shrink-0 text-(--muted-foreground)" aria-hidden="true" />
          Search
        </button>
      </div>

      <Separator />

      <ScrollArea className="flex-1 px-2 py-2">
        {conversations.length === 0 && (
          <div className="flex flex-col items-center gap-2 py-8 text-center text-xs text-(--muted-foreground)">
            <MessageSquare className="h-8 w-8 opacity-30" />
            <p>No conversations yet</p>
          </div>
        )}

        {DATE_ORDER.filter((label) => groups[label]?.length).map((label) => (
          <div key={label} className="mb-3">
            <p className="px-2 py-1 text-xs font-medium text-(--muted-foreground)">{label}</p>
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

      <Separator />
      <div className="p-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors hover:bg-(--border)/60">
              <Avatar className="h-7 w-7 shrink-0">
                <AvatarFallback className="text-xs">{initials}</AvatarFallback>
              </Avatar>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm">{user?.display_name ?? user?.email}</span>
                {user?.display_name && (
                  <span className="block truncate text-xs text-(--muted-foreground)">{user.email}</span>
                )}
              </span>
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" side="top" className="w-56">
            <DropdownMenuItem onClick={() => router.push("/chat/settings")}>
              <Settings className="mr-2 h-3.5 w-3.5" /> Settings
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={logout} className="text-(--destructive) focus:text-(--destructive)">
              <LogOut className="mr-2 h-3.5 w-3.5" /> Sign out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <SearchDialog open={searchOpen} onOpenChange={setSearchOpen} />
    </aside>
  );
}
