"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { ApiKeyConfig, ProviderModels } from "@/types";

export function useApiKeys() {
  const accessToken = useAuthStore((s) => s.accessToken);
  const userId = useAuthStore((s) => s.user?.id);
  const keysKey = accessToken && userId ? (["api-keys", userId] as const) : null;
  const modelsKey = accessToken && userId
    ? (["api-key-models", userId] as const)
    : null;

  const { data: keys, mutate: mutateKeys, isLoading: keysLoading } = useSWR<ApiKeyConfig[]>(
    keysKey,
    () => api.apiKeys.list(keysKey![1]),
  );

  const { data: models, mutate: mutateModels, isLoading: modelsLoading } = useSWR<ProviderModels[]>(
    modelsKey,
    () => api.apiKeys.models(modelsKey![1]),
  );

  const saveKey = async (data: { provider: string; api_key: string; base_url?: string; model_id?: string; display_name?: string }) => {
    if (!userId) throw new Error("Authentication session changed.");
    const result = await api.apiKeys.save(userId, data);
    mutateKeys();
    mutateModels();
    return result;
  };

  const deleteKey = async (provider: string) => {
    if (!userId) throw new Error("Authentication session changed.");
    await api.apiKeys.delete(userId, provider);
    mutateKeys();
    mutateModels();
  };

  // Build a flat list of available models (only from configured providers)
  const availableModels = (models ?? [])
    .filter((p) => p.configured && p.models.length > 0)
    .flatMap((p) => p.models.map((m) => ({ ...m, provider: p.provider })));

  return {
    keys: keys ?? [],
    models: models ?? [],
    availableModels,
    keysLoading,
    modelsLoading,
    saveKey,
    deleteKey,
  };
}
