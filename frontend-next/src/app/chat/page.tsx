"use client";
import { useEffect, useRef } from "react";
import { Brain, Heart, Moon, Wind } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { useChat } from "@/hooks/useChat";
import { useConversations } from "@/hooks/useConversations";
import { useApiKeys } from "@/hooks/useApiKeys";

const STARTERS = [
  { icon: Heart, text: "I've been feeling anxious lately" },
  { icon: Moon, text: "I'm struggling with sleep and stress" },
  { icon: Wind, text: "I feel overwhelmed and don't know where to start" },
  { icon: Brain, text: "I'd like to talk about managing my emotions" },
];

export default function NewChatPage() {
  const { addConversation, refresh } = useConversations();
  const { availableModels } = useApiKeys();
  const refreshTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { messages, sendMessage, isLoading } = useChat({
    conversationId: null,
    onNewConversation: (id) => {
      // Keep this page (and the streamed reply on it) mounted: only swap the
      // URL so a reload or share lands on the conversation route.
      window.history.replaceState(null, "", `/chat/${id}`);
      addConversation({
        id,
        title: "New conversation",
        model: null,
        provider: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
      // Pick up the generated title once the backend has written it.
      refreshTimer.current = setTimeout(() => void refresh(), 4000);
    },
  });

  useEffect(() => () => {
    if (refreshTimer.current) clearTimeout(refreshTimer.current);
  }, []);

  const defaultModel = availableModels[0];
  const started = messages.length > 0;

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      {started ? (
        <ChatWindow messages={messages} />
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-6 overflow-y-auto px-4 py-8">
          <div className="flex flex-col items-center gap-3 text-center">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-(--primary)/10">
              <Brain className="h-7 w-7 text-(--primary)" />
            </div>
            <h1 className="text-2xl font-semibold">How are you feeling today?</h1>
            <p className="text-sm text-(--muted-foreground) max-w-sm">
              I&apos;m here to listen and support you. Share what&apos;s on your mind.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
            {STARTERS.map(({ icon: Icon, text }) => (
              <button
                key={text}
                onClick={() => sendMessage(text, defaultModel?.id ?? "", defaultModel?.provider ?? "")}
                className="flex items-center gap-3 rounded-xl border border-(--border) bg-(--background) px-4 py-3 text-left text-sm hover:bg-(--muted) transition-colors"
              >
                <Icon className="h-4 w-4 shrink-0 text-(--primary)" />
                <span className="text-(--foreground)">{text}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <ChatInput onSend={sendMessage} isLoading={isLoading} models={availableModels} />
    </div>
  );
}
