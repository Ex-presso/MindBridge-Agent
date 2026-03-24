"use client";
import { useRef, useState, KeyboardEvent } from "react";
import { Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const MODELS = [
  { label: "Gemini 2.5 Flash", model: "gemini-2.5-flash", provider: "google_genai" },
  { label: "GPT-5", model: "gpt-5", provider: "openai" },
];

interface Props {
  onSend: (text: string, model: string, provider: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export function ChatInput({ onSend, isLoading, disabled }: Props) {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX = 4000;

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || isLoading || disabled) return;
    onSend(trimmed, selectedModel.model, selectedModel.provider);
    setText("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  };

  return (
    <div className="border-t border-[--border] bg-[--background] px-4 py-3">
      <div className="max-w-3xl mx-auto">
        {/* Model selector */}
        <div className="mb-2 flex items-center gap-2">
          <select
            value={selectedModel.model}
            onChange={(e) => {
              const m = MODELS.find((m) => m.model === e.target.value);
              if (m) setSelectedModel(m);
            }}
            className="text-xs border border-[--border] rounded px-2 py-1 bg-[--background] text-[--muted-foreground] cursor-pointer"
          >
            {MODELS.map((m) => (
              <option key={m.model} value={m.model}>{m.label}</option>
            ))}
          </select>
          {text.length > MAX * 0.8 && (
            <span className={cn("text-xs", text.length >= MAX ? "text-[--destructive]" : "text-[--muted-foreground]")}>
              {text.length}/{MAX}
            </span>
          )}
        </div>

        {/* Input row */}
        <div className="flex items-end gap-2 rounded-xl border border-[--border] bg-[--muted] px-3 py-2 focus-within:ring-2 focus-within:ring-[--ring]">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX))}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            rows={1}
            placeholder="Share what's on your mind…"
            disabled={isLoading || disabled}
            className="flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-[--muted-foreground] disabled:opacity-50 max-h-[140px]"
          />
          <Button
            size="icon"
            variant="default"
            onClick={submit}
            disabled={!text.trim() || isLoading || disabled}
            className="h-8 w-8 shrink-0 rounded-lg"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="mt-1 text-center text-xs text-[--muted-foreground]">
          MindBridge can make mistakes. Not a substitute for professional help.
        </p>
      </div>
    </div>
  );
}
