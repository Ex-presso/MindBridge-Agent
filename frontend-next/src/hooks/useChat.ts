"use client";
import { useCallback, useRef, useState } from "react";
import { useAuthStore } from "@/stores/authStore";
import { readSSEStream } from "@/lib/stream";
import type { Message, OptimisticMessage } from "@/types";

interface UseChatOptions {
  conversationId: string | null;
  initialMessages?: Message[];
  onNewConversation?: (id: string) => void;
}

export function useChat({
  conversationId,
  initialMessages = [],
  onNewConversation,
}: UseChatOptions) {
  const [messages, setMessages] = useState<OptimisticMessage[]>(
    initialMessages.map((m) => ({ ...m, isStreaming: false })),
  );
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const currentConvId = useRef<string | null>(conversationId);
  const accessToken = useAuthStore((s) => s.accessToken);
  const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

  const sendMessage = useCallback(
    async (
      text: string,
      model = "deepseek-v4-flash",
      provider = "openai_compatible",
    ) => {
      if (!text.trim() || isLoading) return;
      setError(null);
      setIsLoading(true);

      const userMsgId = `user-${Date.now()}`;
      const assistantMsgId = `streaming-${Date.now()}`;

      setMessages((prev) => [
        ...prev,
        { id: userMsgId, role: "user", content: text },
        { id: assistantMsgId, role: "assistant", content: "", isStreaming: true },
      ]);

      try {
        const res = await fetch(`${API_BASE}/api/v1/chat`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${accessToken}`,
          },
          credentials: "include",
          body: JSON.stringify({
            message: text,
            conversation_id: currentConvId.current,
            model,
            provider,
            stream: true,
          }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error((err as { detail?: string }).detail ?? `HTTP ${res.status}`);
        }

        const newConvId = res.headers.get("X-Conversation-Id");
        if (newConvId && !currentConvId.current) {
          currentConvId.current = newConvId;
        }

        let accumulated = "";
        for await (const token of readSSEStream(res)) {
          accumulated += token;
          const snap = accumulated;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, content: snap } : m,
            ),
          );
        }

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsgId ? { ...m, isStreaming: false } : m,
          ),
        );
        if (newConvId) onNewConversation?.(newConvId);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
        setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId));
      } finally {
        setIsLoading(false);
      }
    },
    [isLoading, accessToken, API_BASE, onNewConversation],
  );

  const resetMessages = useCallback(
    (msgs: Message[] = []) => {
      setMessages(msgs.map((m) => ({ ...m, isStreaming: false })));
      currentConvId.current = conversationId;
      setError(null);
    },
    [conversationId],
  );

  return { messages, isLoading, error, sendMessage, resetMessages };
}
