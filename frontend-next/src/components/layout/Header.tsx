"use client";
import { Brain, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/hooks/useAuth";

export function Header() {
  const { user, logout } = useAuth();

  const initials = user?.display_name
    ? user.display_name.slice(0, 2).toUpperCase()
    : user?.email.slice(0, 2).toUpperCase() ?? "MB";

  return (
    <header className="flex h-12 shrink-0 items-center justify-between border-b border-[--border] px-4">
      <div className="flex items-center gap-2">
        <Brain className="h-5 w-5 text-[--primary]" />
        <span className="font-semibold text-sm tracking-tight">MindBridge</span>
      </div>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="rounded-full h-8 w-8">
            <Avatar className="h-7 w-7">
              <AvatarFallback className="text-xs">{initials}</AvatarFallback>
            </Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          <div className="px-2 py-1.5 text-xs text-[--muted-foreground] truncate">{user?.email}</div>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={logout} className="text-[--destructive] focus:text-[--destructive]">
            <LogOut className="mr-2 h-3.5 w-3.5" /> Sign out
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </header>
  );
}
