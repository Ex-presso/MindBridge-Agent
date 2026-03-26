import { useAuthStore } from "@/stores/authStore";
import type { ApiKeyConfig, Conversation, ConversationDetail, ProviderModels, User } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

async function tryRefreshToken(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
      method: "POST",
      credentials: "include",
    });
    if (!res.ok) return false;
    const { access_token } = await res.json();
    const userRes = await fetch(`${API_BASE}/api/v1/auth/me`, {
      headers: { Authorization: `Bearer ${access_token}` },
    });
    if (!userRes.ok) return false;
    const user: User = await userRes.json();
    useAuthStore.getState().setAuth(access_token, user);
    return true;
  } catch {
    return false;
  }
}

async function request<T>(path: string, options: RequestInit = {}, retry = true): Promise<T> {
  const token = useAuthStore.getState().accessToken;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: "include",
  });

  if (res.status === 401 && retry) {
    const refreshed = await tryRefreshToken();
    if (refreshed) return request<T>(path, options, false);
    useAuthStore.getState().clearAuth();
    if (typeof window !== "undefined") window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (res.status === 204) return undefined as T;

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

export const api = {
  auth: {
    register: (data: { email: string; password: string; display_name?: string }) =>
      request<{ access_token: string }>("/api/v1/auth/register", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    login: (data: { email: string; password: string }) =>
      request<{ access_token: string }>("/api/v1/auth/login", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    logout: () => request<void>("/api/v1/auth/logout", { method: "POST" }),
    me: () => request<User>("/api/v1/auth/me"),
    tryRefresh: tryRefreshToken,
  },
  conversations: {
    list: () => request<Conversation[]>("/api/v1/conversations"),
    get: (id: string) => request<ConversationDetail>(`/api/v1/conversations/${id}`),
    delete: (id: string) =>
      request<void>(`/api/v1/conversations/${id}`, { method: "DELETE" }),
    updateTitle: (id: string, title: string) =>
      request<Conversation>(`/api/v1/conversations/${id}/title`, {
        method: "PATCH",
        body: JSON.stringify({ title }),
      }),
  },
  apiKeys: {
    list: () => request<ApiKeyConfig[]>("/api/v1/api-keys"),
    save: (data: { provider: string; api_key: string; base_url?: string; model_id?: string; display_name?: string }) =>
      request<ApiKeyConfig>("/api/v1/api-keys", { method: "PUT", body: JSON.stringify(data) }),
    delete: (provider: string) =>
      request<void>(`/api/v1/api-keys/${provider}`, { method: "DELETE" }),
    models: () => request<ProviderModels[]>("/api/v1/api-keys/models"),
  },
};
