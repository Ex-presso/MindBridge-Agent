"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { Conversation } from "@/types";

export function useConversations() {
  const accessToken = useAuthStore((s) => s.accessToken);

  const { data, error, mutate, isLoading } = useSWR<Conversation[]>(
    accessToken ? "/conversations" : null,
    () => api.conversations.list(),
    { revalidateOnFocus: false },
  );

  const deleteConversation = async (id: string) => {
    await api.conversations.delete(id);
    mutate(data?.filter((c) => c.id !== id));
  };

  const renameConversation = async (id: string, title: string) => {
    await api.conversations.updateTitle(id, title);
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
