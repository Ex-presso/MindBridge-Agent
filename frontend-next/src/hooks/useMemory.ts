"use client";

import useSWR from "swr";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { MemoryList } from "@/types";

const MEMORY_PAGE_SIZE = 50;

export function useMemory() {
  const accessToken = useAuthStore((state) => state.accessToken);
  const userId = useAuthStore((state) => state.user?.id);

  const { data, error, isLoading, mutate } = useSWR<MemoryList>(
    accessToken && userId ? ["memory", userId, MEMORY_PAGE_SIZE, 0] : null,
    () => api.memory.get(MEMORY_PAGE_SIZE, 0),
    { revalidateOnFocus: false },
  );

  const setMemoryEnabled = async (memoryEnabled: boolean) => {
    try {
      const status = await api.memory.toggle(memoryEnabled);
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
    try {
      const result = await api.memory.clear();
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
