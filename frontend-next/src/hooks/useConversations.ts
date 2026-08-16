"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { Conversation } from "@/types";

export function useConversations() {
  const accessToken = useAuthStore((s) => s.accessToken);
  const userId = useAuthStore((s) => s.user?.id);
  const conversationsKey = accessToken && userId
    ? (["conversations", userId] as const)
    : null;

  const { data, error, mutate, isLoading } = useSWR<Conversation[]>(
    conversationsKey,
    () => api.conversations.list(conversationsKey![1]),
    { revalidateOnFocus: false },
  );

  const deleteConversation = async (id: string) => {
    if (!userId) throw new Error("Authentication session changed.");
    await api.conversations.delete(userId, id);
    mutate(data?.filter((c) => c.id !== id));
  };

  const renameConversation = async (id: string, title: string) => {
    if (!userId) throw new Error("Authentication session changed.");
    await api.conversations.updateTitle(userId, id, title);
    mutate(data?.map((c) => (c.id === id ? { ...c, title } : c)));
  };

  const addConversation = (conv: Conversation) => {
    mutate([conv, ...(data ?? [])]);
  };

  const refresh = () => mutate();

  return {
    conversations: data ?? [],
    isLoading,
    error,
    deleteConversation,
    renameConversation,
    addConversation,
    refresh,
  };
}
