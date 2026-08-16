"use client";
import { useCallback } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { api } from "@/lib/api";

export function useAuth() {
  const { accessToken, user, setAuth, clearAuth } = useAuthStore();
  const userId = user?.id;
  const router = useRouter();

  const login = useCallback(
    async (email: string, password: string) => {
      const { access_token } = await api.auth.login({ email, password });
      const userData = await fetchMe(access_token);
      setAuth(access_token, userData);
      router.push("/chat");
    },
    [setAuth, router],
  );

  const register = useCallback(
    async (email: string, password: string, displayName?: string) => {
      const { access_token } = await api.auth.register({
        email,
        password,
        display_name: displayName,
      });
      const userData = await fetchMe(access_token);
      setAuth(access_token, userData);
      router.push("/chat");
    },
    [setAuth, router],
  );

  const logout = useCallback(async () => {
    await api.auth.logout().catch(() => {});
    clearAuth();
    router.push("/login");
  }, [clearAuth, router]);

  const deleteAccount = useCallback(async (password: string) => {
    if (!userId) throw new Error("Authentication session changed.");
    await api.auth.deleteAccount(userId, password);
    clearAuth();
    router.replace("/login");
  }, [userId, clearAuth, router]);

  return {
    user,
    accessToken,
    isAuthenticated: !!accessToken,
    login,
    register,
    logout,
    deleteAccount,
  };
}

async function fetchMe(token: string) {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";
  const res = await fetch(`${API_BASE}/api/v1/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
}
