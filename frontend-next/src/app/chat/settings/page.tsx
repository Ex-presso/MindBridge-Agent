"use client";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Brain, Cpu, Search, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { AccountSection } from "@/components/settings/AccountSection";
import { MemorySection } from "@/components/settings/MemorySection";
import { ModelSection } from "@/components/settings/ModelSection";

type SectionId = "model" | "memory" | "account";

const SECTIONS: {
  id: SectionId;
  label: string;
  icon: typeof Cpu;
  keywords: string[];
}[] = [
  {
    id: "model",
    label: "Model",
    icon: Cpu,
    keywords: ["model", "api", "key", "provider", "openai", "anthropic", "base url", "deepseek", "llm"],
  },
  {
    id: "memory",
    label: "Memory",
    icon: Brain,
    keywords: ["memory", "episode", "fact", "consent", "privacy", "clear", "recall", "long-term"],
  },
  {
    id: "account",
    label: "Account",
    icon: User,
    keywords: ["account", "password", "email", "name", "profile", "delete", "sign"],
  },
];

export default function SettingsPage() {
  const router = useRouter();
  const [active, setActive] = useState<SectionId>("model");
  const [query, setQuery] = useState("");

  const visibleSections = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return SECTIONS;
    return SECTIONS.filter(
      (s) => s.label.toLowerCase().includes(q) || s.keywords.some((k) => k.includes(q)),
    );
  }, [query]);

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      <div className="mx-auto w-full max-w-4xl px-4 py-6">
        <div className="mb-6 flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={() => router.push("/chat")} className="h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <h1 className="text-lg font-semibold">Settings</h1>
        </div>

        <div className="flex flex-col gap-6 md:flex-row">
          {/* Settings navigation */}
          <nav className="shrink-0 md:w-48" aria-label="Settings sections">
            <div className="mb-2 flex items-center gap-2 rounded-lg border border-(--border) bg-(--background) px-2.5">
              <Search className="h-3.5 w-3.5 shrink-0 text-(--muted-foreground)" aria-hidden="true" />
              <input
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  const q = e.target.value.trim().toLowerCase();
                  if (q) {
                    const match = SECTIONS.find(
                      (s) => s.label.toLowerCase().includes(q) || s.keywords.some((k) => k.includes(q)),
                    );
                    if (match) setActive(match.id);
                  }
                }}
                placeholder="Search settings"
                className="h-8 w-full bg-transparent text-xs outline-none placeholder:text-(--muted-foreground)"
              />
            </div>

            <div className="flex gap-1 md:flex-col">
              {visibleSections.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActive(id)}
                  aria-current={active === id ? "true" : undefined}
                  className={cn(
                    "flex flex-1 items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors md:flex-none",
                    active === id
                      ? "bg-(--muted) font-medium"
                      : "text-(--muted-foreground) hover:bg-(--muted)/60 hover:text-(--foreground)",
                  )}
                >
                  <Icon className="h-4 w-4 shrink-0" aria-hidden="true" />
                  {label}
                </button>
              ))}
              {visibleSections.length === 0 && (
                <p className="px-2.5 py-2 text-xs text-(--muted-foreground)">No matching settings</p>
              )}
            </div>
          </nav>

          {/* Active section */}
          <div className="min-w-0 flex-1">
            {active === "model" && <ModelSection />}
            {active === "memory" && <MemorySection />}
            {active === "account" && <AccountSection />}
          </div>
        </div>
      </div>
    </div>
  );
}
