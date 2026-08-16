"use client";
import { use, useEffect } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { ChatWindow } from "@/components/chat/ChatWindow";
import { ChatInput } from "@/components/chat/ChatInput";
import { useChat } from "@/hooks/useChat";
import { useConversations } from "@/hooks/useConversations";
import { useApiKeys } from "@/hooks/useApiKeys";
import { useAuthStore } from "@/stores/authStore";
import useSWR from "swr";

interface Props {
  params: Promise<{ conversationId: string }>;
}

export default function ConversationPage({ params }: Props) {
  const { conversationId } = use(params);
  const router = useRouter();
  const accessToken = useAuthStore((state) => state.accessToken);
  const userId = useAuthStore((state) => state.user?.id);
  const { refresh } = useConversations();
  const { availableModels } = useApiKeys();
  const conversationKey = accessToken && userId && conversationId
    ? (["conversation", userId, conversationId] as const)
    : null;

  const { data, isLoading: historyLoading, error } = useSWR(
    conversationKey,
    () => api.conversations.get(conversationKey![1], conversationKey![2]),
  );

  const { messages, isLoading, sendMessage, resetMessages } = useChat({
    conversationId,
    initialMessages: data?.messages ?? [],
    onNewConversation: (id) => {
      router.replace(`/chat/${id}`);
      refresh();
    },
  });

  useEffect(() => {
    if (data?.messages) {
      resetMessages(data.messages);
    }
  }, [data?.id, data?.messages, resetMessages]);

  useEffect(() => {
    if (error) {
      router.replace("/chat");
    }
  }, [error, router]);

  if (historyLoading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <div className="h-6 w-6 rounded-full border-2 border-[--primary] border-t-transparent animate-spin" />
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden">
      <ChatWindow messages={messages} isLoading={isLoading} />
      <ChatInput onSend={sendMessage} isLoading={isLoading} models={availableModels} />
    </div>
  );
}
