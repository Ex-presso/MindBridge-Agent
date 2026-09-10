/**
 * Capture README screenshots from a running local stack.
 * Usage: node e2e/screenshots.mjs <email> <password> [outDir] [conversationId]
 * Requires frontend on PW_BASE_URL (default http://localhost:3002) with a
 * user that already has conversations and memory items.
 */
import { chromium } from "@playwright/test";

const BASE = process.env.PW_BASE_URL ?? "http://localhost:3002";
const [email, password, outDir = "../docs/img", conversationId] = process.argv.slice(2);

if (!email || !password) {
  console.error("usage: node e2e/screenshots.mjs <email> <password> [outDir]");
  process.exit(1);
}

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
});

await page.goto(`${BASE}/login`);
await page.getByPlaceholder("you@example.com").fill(email);
await page.getByPlaceholder("••••••••").fill(password);
await page.getByRole("button", { name: "Sign in" }).click();
await page.waitForURL(/\/chat/, { timeout: 15000 });
await page.waitForTimeout(1500);

// 1. Chat: open a conversation (by id when given, else the empty state).
await page.goto(conversationId ? `${BASE}/chat/${conversationId}` : `${BASE}/chat`);
await page.waitForTimeout(2000);
await page.screenshot({ path: `${outDir}/screenshot-chat.png` });

// 2. Memory panel.
await page.goto(`${BASE}/chat/memory`);
await page.waitForTimeout(1500);
await page.screenshot({ path: `${outDir}/screenshot-memory.png` });

// 3. Settings.
await page.goto(`${BASE}/chat/settings`);
await page.waitForTimeout(1500);
await page.screenshot({ path: `${outDir}/screenshot-settings.png` });

await browser.close();
console.log("screenshots written to", outDir);
