"use client";
import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import type { OptimisticMessage } from "@/types";
import "highlight.js/styles/github.css";

function formatDuration(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

/** Live seconds while the assistant works; "Worked for Xm Ys" once done. */
function WorkTimer({ startedAt, completedAt }: { startedAt: number; completedAt?: number }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (completedAt) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [completedAt]);

  const elapsed = formatDuration((completedAt ?? now) - startedAt);
  return (
    <span className="text-xs tabular-nums text-(--muted-foreground)">
      {completedAt ? `Worked for ${elapsed}` : elapsed}
    </span>
  );
}

interface Props {
  message: OptimisticMessage;
}

export function MessageBubble({ message }: Props) {
  if (message.role === "user") {
    return (
      <div className="mx-auto flex w-full max-w-3xl justify-end px-4 py-3">
        <div className="max-w-[85%] rounded-2xl rounded-tr-sm bg-(--primary) px-4 py-2.5 text-sm leading-relaxed text-(--primary-foreground)">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-4 py-3">
      {/* Dashed separator with the work timer above its left end */}
      <div className="mb-3">
        {message.startedAt !== undefined && (
          <div className="mb-1">
            <WorkTimer startedAt={message.startedAt} completedAt={message.completedAt} />
          </div>
        )}
        <div className="border-t border-dashed border-(--border)" />
      </div>

      <div className="prose max-w-none text-sm leading-relaxed text-(--foreground)">
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
          {message.content || ""}
        </ReactMarkdown>
        {message.isStreaming && (
          <span className="ml-0.5 inline-block h-4 w-0.5 animate-pulse bg-current align-middle" />
        )}
      </div>
    </div>
  );
}
