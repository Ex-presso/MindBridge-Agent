import { defineConfig } from "@playwright/test";

// The smoke test needs the full stack: the backend (with PostgreSQL) must
// already be running on PW_API_URL. The Next.js dev server and the OpenAI-
// compatible LLM stub are started here unless they are already up.
const BASE_URL = process.env.PW_BASE_URL ?? "http://localhost:3002";
const API_URL = process.env.PW_API_URL ?? "http://localhost:8082";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: BASE_URL,
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command: "node e2e/llm-stub.mjs",
      url: "http://127.0.0.1:8093/health",
      env: { LLM_STUB_PORT: "8093" },
      reuseExistingServer: true,
      timeout: 15_000,
    },
    {
      command: `npx next dev --port ${new URL(BASE_URL).port || "3002"}`,
      url: BASE_URL,
      env: { NEXT_PUBLIC_API_URL: API_URL },
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});
