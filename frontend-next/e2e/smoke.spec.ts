import { expect, test } from "@playwright/test";

/**
 * Full-stack smoke test: register -> configure a provider key pointing at the
 * local OpenAI-compatible stub -> send a message -> receive a streamed reply.
 * Requires the backend (with PostgreSQL) on PW_API_URL; the Next.js dev server
 * and the LLM stub are managed by playwright.config.ts.
 */

const STUB_BASE_URL = "http://127.0.0.1:8093/v1";

test("register, configure a key, chat and stream a reply", async ({ page }) => {
  const email = `smoke-${Date.now()}@example.com`;

  await page.goto("/register");
  await page.getByPlaceholder("Your name").fill("Smoke Test");
  await page.getByPlaceholder("you@example.com").fill(email);
  await page.getByPlaceholder("••••••••").fill("SmokePass1234");
  await page.getByRole("button", { name: "Create account" }).click();
  await expect(page).toHaveURL(/\/chat/, { timeout: 15_000 });

  // Configure the OpenAI-compatible provider against the stub.
  await page.goto("/chat/settings");
  const compatibleCard = page
    .locator("div.rounded-xl")
    .filter({ hasText: "OpenAI Compatible" })
    .first();
  await compatibleCard.getByPlaceholder("Enter API key…").fill("stub-key-not-secret");
  await compatibleCard.getByPlaceholder("https://api.example.com/v1").fill(STUB_BASE_URL);
  await compatibleCard.getByPlaceholder("e.g., llama3, mistral, etc.").fill("stub-model");
  await compatibleCard.getByPlaceholder("e.g., My Local Ollama").fill("Demo Model");
  await compatibleCard.getByRole("button", { name: "Save Key" }).click();
  await expect(compatibleCard.getByText("Configured")).toBeVisible({ timeout: 10_000 });

  // Send a message and wait for the streamed assistant reply.
  await page.goto("/chat");
  await expect(page.getByLabel("Model")).toContainText("Demo Model", { timeout: 10_000 });
  await page
    .getByPlaceholder("Share what's on your mind...")
    .fill("I've been feeling anxious lately and it's hard to sleep.");
  await page.getByRole("button", { name: "Send" }).click();

  await expect(page.getByText("It sounds like things have been weighing on you")).toBeVisible({
    timeout: 30_000,
  });

  // The conversation and both messages survive a reload (they were persisted).
  await page.reload();
  await expect(
    page.getByText("I've been feeling anxious lately and it's hard to sleep."),
  ).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText("It sounds like things have been weighing on you")).toBeVisible();

  // The memory panel renders with the consent switch.
  await page.goto("/chat/memory");
  await expect(page.getByRole("switch", { name: "Long-term memory" })).toBeVisible();
});
