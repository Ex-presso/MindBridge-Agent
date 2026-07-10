import { useAuthStore } from "@/stores/authStore";
import type {
  ApiKeyConfig,
  Conversation,
  ConversationDetail,
  MemoryDelete,
  MemoryList,
  MemoryStatus,
  ProviderModels,
  User,
} from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

function assertExpectedUser(expectedUserId?: string): void {
  if (expectedUserId && useAuthStore.getState().user?.id !== expectedUserId) {
    throw new Error("Authentication session changed.");
  }
}

async function tryRefreshToken(expectedUserId?: string): Promise<boolean> {
  try {
    assertExpectedUser(expectedUserId);
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
    if (expectedUserId && user.id !== expectedUserId) return false;
    assertExpectedUser(expectedUserId);
    useAuthStore.getState().setAuth(access_token, user);
    return true;
  } catch {
    return false;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  retry = true,
  expectedUserId?: string,
): Promise<T> {
  assertExpectedUser(expectedUserId);
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

  assertExpectedUser(expectedUserId);
  if (res.status === 401 && retry) {
    const refreshed = await tryRefreshToken(expectedUserId);
    if (refreshed) return request<T>(path, options, false, expectedUserId);
    if (!expectedUserId || useAuthStore.getState().user?.id === expectedUserId) {
      useAuthStore.getState().clearAuth();
      if (typeof window !== "undefined") window.location.href = "/login";
    }
    throw new Error("Unauthorized");
  }

  if (res.status === 204) {
    assertExpectedUser(expectedUserId);
    return undefined as T;
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
  }

  const body = await res.json();
  assertExpectedUser(expectedUserId);
  return body;
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
    deleteAccount: (expectedUserId: string, password: string) =>
      request<void>(
        "/api/v1/auth/me",
        { method: "DELETE", body: JSON.stringify({ password }) },
        true,
        expectedUserId,
      ),
    tryRefresh: tryRefreshToken,
  },
  conversations: {
    list: (expectedUserId: string) =>
      request<Conversation[]>("/api/v1/conversations", {}, true, expectedUserId),
    get: (expectedUserId: string, id: string) =>
      request<ConversationDetail>(
        `/api/v1/conversations/${id}`,
        {},
        true,
        expectedUserId,
      ),
    delete: (expectedUserId: string, id: string) =>
      request<void>(
        `/api/v1/conversations/${id}`,
        { method: "DELETE" },
        true,
        expectedUserId,
      ),
    updateTitle: (expectedUserId: string, id: string, title: string) =>
      request<Conversation>(`/api/v1/conversations/${id}/title`, {
        method: "PATCH",
        body: JSON.stringify({ title }),
      }, true, expectedUserId),
  },
  apiKeys: {
    list: (expectedUserId: string) =>
      request<ApiKeyConfig[]>("/api/v1/api-keys", {}, true, expectedUserId),
    save: (
      expectedUserId: string,
      data: { provider: string; api_key: string; base_url?: string; model_id?: string; display_name?: string },
    ) => request<ApiKeyConfig>(
      "/api/v1/api-keys",
      { method: "PUT", body: JSON.stringify(data) },
      true,
      expectedUserId,
    ),
    delete: (expectedUserId: string, provider: string) =>
      request<void>(
        `/api/v1/api-keys/${provider}`,
        { method: "DELETE" },
        true,
        expectedUserId,
      ),
    models: (expectedUserId: string) =>
      request<ProviderModels[]>(
        "/api/v1/api-keys/models",
        {},
        true,
        expectedUserId,
      ),
  },
  memory: {
    get: (expectedUserId: string, limit = 50, offset = 0) =>
      request<MemoryList>(
        `/api/v1/memory?limit=${limit}&offset=${offset}`,
        {},
        true,
        expectedUserId,
      ),
    toggle: (expectedUserId: string, memoryEnabled: boolean) =>
      request<MemoryStatus>("/api/v1/memory", {
        method: "PATCH",
        body: JSON.stringify({ memory_enabled: memoryEnabled }),
      }, true, expectedUserId),
    clear: (expectedUserId: string) =>
      request<MemoryDelete>(
        "/api/v1/memory",
        { method: "DELETE" },
        true,
        expectedUserId,
      ),
  },
};
