"use client";
import { useEffect } from "react";
import { useAuthStore } from "@/stores/authStore";
import { api } from "@/lib/api";

export function AuthInitializer() {
  const { accessToken, setInitialized } = useAuthStore();

  useEffect(() => {
    if (accessToken) {
      setInitialized();
      return;
    }
    // Try to restore session via refresh token cookie
    api.auth.tryRefresh().finally(() => setInitialized());
  }, [accessToken, setInitialized]);

  return null;
}
