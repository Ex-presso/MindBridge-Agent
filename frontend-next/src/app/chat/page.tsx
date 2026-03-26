"use client";
import { useRouter } from "next/navigation";
import { Brain, Heart, Moon, Wind } from "lucide-react";
import { ChatInput } from "@/components/chat/ChatInput";
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
  const router = useRouter();
  const { addConversation } = useConversations();
  const { availableModels } = useApiKeys();

  const { sendMessage, isLoading } = useChat({
    conversationId: null,
    onNewConversation: (id) => {
      router.replace(`/chat/${id}`);
      addConversation({
        id,
        title: "New conversation",
        model: null,
        provider: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      });
    },
  });

  return (
    <div className="flex flex-1 flex-col">
      <div className="flex flex-1 flex-col items-center justify-center gap-6 px-4 py-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[--primary]/10">
            <Brain className="h-7 w-7 text-[--primary]" />
          </div>
          <h1 className="text-2xl font-semibold">How are you feeling today?</h1>
          <p className="text-sm text-[--muted-foreground] max-w-sm">
            I&apos;m here to listen and support you. Share what&apos;s on your mind.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
          {STARTERS.map(({ icon: Icon, text }) => (
            <button
              key={text}
              onClick={() => sendMessage(text)}
              className="flex items-center gap-3 rounded-xl border border-[--border] bg-[--background] px-4 py-3 text-left text-sm hover:bg-[--muted] transition-colors"
            >
              <Icon className="h-4 w-4 shrink-0 text-[--primary]" />
              <span className="text-[--foreground]">{text}</span>
            </button>
          ))}
        </div>
      </div>

      <ChatInput onSend={sendMessage} isLoading={isLoading} models={availableModels} />
    </div>
  );
}
