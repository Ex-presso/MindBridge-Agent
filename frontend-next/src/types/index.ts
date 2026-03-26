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

export const PROVIDER_INFO: Record<string, { label: string; description: string; needsBaseUrl: boolean; needsModelId: boolean }> = {
  openai: { label: "OpenAI", description: "GPT-4o, o3, etc.", needsBaseUrl: false, needsModelId: false },
  anthropic: { label: "Anthropic", description: "Claude Sonnet, Haiku, Opus", needsBaseUrl: false, needsModelId: false },
  google_genai: { label: "Google Gemini", description: "Gemini 2.5 Flash, Pro", needsBaseUrl: false, needsModelId: false },
  openai_compatible: { label: "OpenAI Compatible", description: "Ollama, LM Studio, Azure, etc.", needsBaseUrl: true, needsModelId: true },
  anthropic_compatible: { label: "Anthropic Compatible", description: "Custom Claude-compatible endpoint", needsBaseUrl: true, needsModelId: true },
};
