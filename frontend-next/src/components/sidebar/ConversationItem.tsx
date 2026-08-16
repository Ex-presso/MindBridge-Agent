"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import type { Conversation } from "@/types";

interface Props {
  conversation: Conversation;
  isActive: boolean;
  onDelete: (id: string) => Promise<void>;
  onRename: (id: string, title: string) => Promise<void>;
}

export function ConversationItem({ conversation, isActive, onDelete, onRename }: Props) {
  const router = useRouter();
  const [renameOpen, setRenameOpen] = useState(false);
  const [newTitle, setNewTitle] = useState(conversation.title);
  const [saving, setSaving] = useState(false);

  const handleRename = async () => {
    if (!newTitle.trim()) return;
    setSaving(true);
    try {
      await onRename(conversation.id, newTitle.trim());
      setRenameOpen(false);
      toast.success("Renamed");
    } catch {
      toast.error("Failed to rename");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    try {
      await onDelete(conversation.id);
      if (isActive) router.push("/chat");
      toast.success("Deleted");
    } catch {
      toast.error("Failed to delete");
    }
  };

  return (
    <>
      <div
        className={cn(
          "group flex items-center gap-2 rounded-lg px-2 py-2 text-sm cursor-pointer transition-colors hover:bg-[--border]",
          isActive && "bg-[--border] font-medium",
        )}
        onClick={() => router.push(`/chat/${conversation.id}`)}
      >
        <span className="flex-1 truncate text-[--foreground]">{conversation.title}</span>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 opacity-0 group-hover:opacity-100 shrink-0"
              onClick={(e) => e.stopPropagation()}
            >
              <MoreHorizontal className="h-3.5 w-3.5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={(e) => { e.stopPropagation(); setRenameOpen(true); }}>
              <Pencil className="mr-2 h-3.5 w-3.5" /> Rename
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              className="text-[--destructive] focus:text-[--destructive]"
              onClick={(e) => { e.stopPropagation(); handleDelete(); }}
            >
              <Trash2 className="mr-2 h-3.5 w-3.5" /> Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Rename conversation</DialogTitle></DialogHeader>
          <Input
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleRename()}
            autoFocus
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setRenameOpen(false)}>Cancel</Button>
            <Button onClick={handleRename} disabled={saving}>Save</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
