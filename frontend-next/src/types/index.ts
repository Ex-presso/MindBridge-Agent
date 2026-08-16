export interface User {
  id: string;
  email: string;
  display_name: string | null;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  model: string | null;
  provider: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface OptimisticMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
}

export interface ApiKeyConfig {
  provider: string;
  api_key_masked: string;
  base_url: string | null;
  model_id: string | null;
  display_name: string | null;
  configured: boolean;
}

export interface ProviderModels {
  provider: string;
  configured: boolean;
  models: { id: string; name: string }[];
}

export interface MemoryItem {
  namespace: string[];
  key: string;
  value: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  score?: number | null;
}

export interface MemoryStatus {
  memory_enabled: boolean;
  system_enabled: boolean;
}

export interface MemoryList extends MemoryStatus {
  items: MemoryItem[];
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface MemoryDelete {
  memory_enabled: false;
  system_enabled: boolean;
  deleted_items: number;
}

export const PROVIDER_INFO: Record<string, { label: string; description: string; needsBaseUrl: boolean; needsModelId: boolean }> = {
  openai: { label: "OpenAI", description: "GPT-4o, o3, etc.", needsBaseUrl: false, needsModelId: false },
  anthropic: { label: "Anthropic", description: "Claude Sonnet, Haiku, Opus", needsBaseUrl: false, needsModelId: false },
  google_genai: { label: "Google Gemini", description: "Gemini 2.5 Flash, Pro", needsBaseUrl: false, needsModelId: false },
  openai_compatible: { label: "OpenAI Compatible", description: "DeepSeek, Ollama, LM Studio, Azure, etc.", needsBaseUrl: true, needsModelId: true },
  anthropic_compatible: { label: "Anthropic Compatible", description: "Custom Claude-compatible endpoint", needsBaseUrl: true, needsModelId: true },
};
