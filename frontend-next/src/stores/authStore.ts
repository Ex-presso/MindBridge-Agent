import { create } from "zustand";
import type { User } from "@/types";

interface AuthState {
  accessToken: string | null;
  user: User | null;
  initialized: boolean;
  setAuth: (token: string, user: User) => void;
  clearAuth: () => void;
  setInitialized: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  accessToken: null,
  user: null,
  initialized: false,
  setAuth: (token, user) => set({ accessToken: token, user, initialized: true }),
  clearAuth: () => set({ accessToken: null, user: null, initialized: true }),
  setInitialized: () => set({ initialized: true }),
}));
