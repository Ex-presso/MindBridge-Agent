"use client";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { Sidebar } from "@/components/sidebar/Sidebar";
import { Header } from "@/components/layout/Header";
import { useAuthStore } from "@/stores/authStore";

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const { accessToken, initialized } = useAuthStore();

  useEffect(() => {
    if (initialized && !accessToken) {
      router.replace("/login");
    }
  }, [accessToken, initialized, router]);

  if (!initialized) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="h-6 w-6 rounded-full border-2 border-[--primary] border-t-transparent animate-spin" />
      </div>
    );
  }

  if (!accessToken) return null;

  return (
    <div className="flex h-screen overflow-hidden bg-[--background]">
      <div className="hidden md:flex flex-col border-r border-[--border] bg-[--sidebar-bg]">
        <Sidebar />
      </div>
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header />
        <main className="flex flex-1 flex-col overflow-hidden">{children}</main>
      </div>
    </div>
  );
}
