"use client";

import useSWR from "swr";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { MemoryList } from "@/types";

const MEMORY_PAGE_SIZE = 50;

export function useMemory() {
  const accessToken = useAuthStore((state) => state.accessToken);
  const userId = useAuthStore((state) => state.user?.id);
  const memoryKey = accessToken && userId
    ? (["memory", userId, MEMORY_PAGE_SIZE, 0] as const)
    : null;

  const { data, error, isLoading, mutate } = useSWR<MemoryList>(
    memoryKey,
    () => api.memory.get(memoryKey![1], MEMORY_PAGE_SIZE, 0),
    { revalidateOnFocus: false },
  );

  const setMemoryEnabled = async (memoryEnabled: boolean) => {
    if (!userId) throw new Error("Authentication session changed.");
    try {
      const status = await api.memory.toggle(userId, memoryEnabled);
      await mutate(
        (current) => current ? { ...current, ...status } : current,
        { revalidate: false },
      );
      void mutate().catch(() => undefined);
      return status;
    } catch (requestError) {
      await mutate().catch(() => undefined);
      throw requestError;
    }
  };

  const clearMemory = async () => {
    if (!userId) throw new Error("Authentication session changed.");
    try {
      const result = await api.memory.clear(userId);
      await mutate(
        (current) => current
          ? {
              ...current,
              memory_enabled: false,
              system_enabled: result.system_enabled,
              items: [],
              offset: 0,
              has_more: false,
            }
          : current,
        { revalidate: false },
      );
      void mutate().catch(() => undefined);
      return result;
    } catch (requestError) {
      await mutate().catch(() => undefined);
      throw requestError;
    }
  };

  return {
    memory: data,
    memoryLoading: isLoading,
    memoryError: error,
    setMemoryEnabled,
    clearMemory,
  };
}
