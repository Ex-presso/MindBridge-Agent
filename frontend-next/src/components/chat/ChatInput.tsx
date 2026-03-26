"use client";
import { useEffect, useRef, useState, KeyboardEvent } from "react";
import { Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export interface ModelOption {
  id: string;
  name: string;
  provider: string;
}

interface Props {
  onSend: (text: string, model: string, provider: string) => void;
  isLoading: boolean;
  disabled?: boolean;
  models?: ModelOption[];
}

export function ChatInput({ onSend, isLoading, disabled, models = [] }: Props) {
  const [text, setText] = useState("");
  const [selectedModel, setSelectedModel] = useState<ModelOption>(models[0] ?? { id: "", name: "No model", provider: "" });
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX = 4000;

  useEffect(() => {
    if (models.length > 0 && !models.find(m => m.id === selectedModel.id)) {
      setSelectedModel(models[0]);
    }
  }, [models]); // eslint-disable-line react-hooks/exhaustive-deps

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || isLoading || disabled || !selectedModel.id) return;
    onSend(trimmed, selectedModel.id, selectedModel.provider);
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
            value={selectedModel.id}
            onChange={(e) => {
              const m = models.find((m) => m.id === e.target.value);
              if (m) setSelectedModel(m);
            }}
            className="text-xs border border-[--border] rounded px-2 py-1 bg-[--background] text-[--muted-foreground] cursor-pointer"
          >
            {models.length === 0 && <option value="">No API keys configured</option>}
            {models.map((m) => (
              <option key={`${m.provider}-${m.id}`} value={m.id}>{m.name}</option>
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
            placeholder="Share what's on your mind..."
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
