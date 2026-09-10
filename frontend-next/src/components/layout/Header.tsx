"use client";
import { Brain, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Header({ onMenuClick }: { onMenuClick?: () => void }) {
  return (
    <header className="flex h-12 shrink-0 items-center border-b border-(--border) px-4">
      <div className="flex items-center gap-2">
        {onMenuClick && (
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8 md:hidden"
            onClick={onMenuClick}
            aria-label="Open menu"
          >
            <Menu className="h-4 w-4" />
          </Button>
        )}
        <Brain className="h-5 w-5 text-(--primary)" />
        <span className="font-semibold text-sm tracking-tight">MindBridge</span>
      </div>
    </header>
  );
}
