"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { Brain, User } from "lucide-react";
import { cn } from "@/lib/utils";
import type { OptimisticMessage } from "@/types";
import "highlight.js/styles/github.css";

interface Props {
  message: OptimisticMessage;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-3 px-4 py-3 max-w-3xl mx-auto w-full", isUser && "flex-row-reverse")}>
      {/* Avatar */}
      <div className={cn(
        "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-medium",
        isUser ? "bg-[--primary] text-[--primary-foreground]" : "bg-[--muted] text-[--muted-foreground]",
      )}>
        {isUser ? <User className="h-4 w-4" /> : <Brain className="h-4 w-4" />}
      </div>

      {/* Content */}
      <div className={cn("flex flex-col gap-1 min-w-0 max-w-[85%]", isUser && "items-end")}>
        <div className={cn(
          "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser
            ? "bg-[--primary] text-[--primary-foreground] rounded-tr-sm"
            : "bg-[--muted] text-[--foreground] rounded-tl-sm",
        )}>
          {isUser ? (
            <p className="whitespace-pre-wrap break-words">{message.content}</p>
          ) : (
            <div className="prose max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
                {message.content || ""}
              </ReactMarkdown>
              {message.isStreaming && (
                <span className="inline-block w-0.5 h-4 bg-current animate-pulse ml-0.5 align-middle" />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
