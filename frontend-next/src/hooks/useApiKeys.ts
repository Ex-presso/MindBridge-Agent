"use client";
import useSWR from "swr";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import type { ApiKeyConfig, ProviderModels } from "@/types";

export function useApiKeys() {
  const accessToken = useAuthStore((s) => s.accessToken);

  const { data: keys, mutate: mutateKeys, isLoading: keysLoading } = useSWR<ApiKeyConfig[]>(
    accessToken ? "/api-keys" : null,
    () => api.apiKeys.list(),
  );

  const { data: models, mutate: mutateModels, isLoading: modelsLoading } = useSWR<ProviderModels[]>(
    accessToken ? "/api-keys/models" : null,
    () => api.apiKeys.models(),
  );

  const saveKey = async (data: { provider: string; api_key: string; base_url?: string; model_id?: string; display_name?: string }) => {
    const result = await api.apiKeys.save(data);
    mutateKeys();
    mutateModels();
    return result;
  };

  const deleteKey = async (provider: string) => {
    await api.apiKeys.delete(provider);
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
