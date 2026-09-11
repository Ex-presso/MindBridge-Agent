"use client";
import { useState } from "react";
import Link from "next/link";
import { Brain, CircleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/hooks/useAuth";

export default function LoginPage() {
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await login(email, password);
    } catch (err) {
      const raw = err instanceof Error ? err.message : "";
      setError(
        /invalid credentials|401/i.test(raw)
          ? "Incorrect email or password."
          : raw || "Sign-in failed. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-(--muted) px-4">
      <div className="w-full max-w-sm">
        <div className="flex flex-col items-center gap-2 mb-8">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-(--primary)">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <h1 className="text-xl font-semibold">Welcome back</h1>
          <p className="text-sm text-(--muted-foreground)">Sign in to MindBridge</p>
        </div>

        <div className="rounded-xl border border-(--border) bg-(--background) p-6 shadow-sm">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" required autoFocus />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="password">Password</Label>
              <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" required />
            </div>
            {error && (
              <p role="alert" className="flex items-center gap-1.5 rounded-lg bg-(--destructive)/10 px-3 py-2 text-xs text-(--destructive)">
                <CircleAlert className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                {error}
              </p>
            )}
            <Button type="submit" disabled={loading} className="w-full mt-1">
              {loading ? "Signing in…" : "Sign in"}
            </Button>
          </form>
        </div>

        <p className="mt-4 text-center text-sm text-(--muted-foreground)">
          Don&apos;t have an account?{" "}
          <Link href="/register" className="font-medium text-(--primary) hover:underline">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}
