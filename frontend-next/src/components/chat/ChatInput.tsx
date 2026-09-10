"use client";
import { useRef, useState, KeyboardEvent } from "react";
import { Send } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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
  const [selectedModelId, setSelectedModelId] = useState("");
  const selectedModel = models.find((model) => model.id === selectedModelId)
    ?? models[0]
    ?? { id: "", name: "No model", provider: "" };
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const MAX = 4000;

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
    <div className="border-t border-(--border) bg-(--background) px-4 py-3">
      <div className="max-w-3xl mx-auto">
        {/* Model selector */}
        <div className="mb-2 flex items-center gap-2">
          <Select
            value={selectedModel.id}
            onValueChange={setSelectedModelId}
            disabled={models.length === 0}
          >
            <SelectTrigger className="max-w-56 text-(--muted-foreground)" aria-label="Model">
              <SelectValue placeholder="No API keys configured" />
            </SelectTrigger>
            <SelectContent>
              {models.map((m) => (
                <SelectItem key={`${m.provider}-${m.id}`} value={m.id}>{m.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          {text.length > MAX * 0.8 && (
            <span className={cn("text-xs", text.length >= MAX ? "text-(--destructive)" : "text-(--muted-foreground)")}>
              {text.length}/{MAX}
            </span>
          )}
        </div>

        {/* Input row */}
        <div className="flex items-end gap-2 rounded-xl border border-(--border) bg-(--muted) px-3 py-2 focus-within:ring-2 focus-within:ring-(--ring)">
          <textarea
            ref={textareaRef}
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, MAX))}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            rows={1}
            placeholder="Share what's on your mind..."
            disabled={isLoading || disabled}
            className="flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-(--muted-foreground) disabled:opacity-50 max-h-[140px]"
          />
          <Button
            size="icon"
            variant="default"
            onClick={submit}
            disabled={!text.trim() || isLoading || disabled}
            className="h-8 w-8 shrink-0 rounded-lg"
            aria-label="Send"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="mt-1 text-center text-xs text-(--muted-foreground)">
          MindBridge can make mistakes. Not a substitute for professional help.
        </p>
      </div>
    </div>
  );
}
