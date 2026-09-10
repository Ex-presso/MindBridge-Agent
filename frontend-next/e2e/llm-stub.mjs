/**
 * Minimal OpenAI-compatible stub for e2e smoke tests.
 *
 * Serves POST [/v1]/chat/completions in both streaming and non-streaming
 * modes with a canned reply, so the full register -> chat -> SSE pipeline
 * can run in CI without a real LLM provider or API key.
 */
import http from "node:http";

const PORT = Number(process.env.LLM_STUB_PORT ?? 8090);

const REPLIES = [
  {
    pattern: /anxious|anxiety|sleep/i,
    reply:
      "It sounds like things have been weighing on you lately, and it takes " +
      "courage to put that into words. Feeling anxious is a very human response " +
      "to pressure. Could you tell me a bit more about when the anxiety tends " +
      "to show up during your day?",
  },
  {
    pattern: /evening|night|bed/i,
    reply:
      "Evenings can be hard because the day's worries finally have room to " +
      "surface. One small thing that helps many people is a short wind-down " +
      "ritual, like writing tomorrow's three priorities on paper so your mind " +
      "can let go of them. Would you like to try something like that this week?",
  },
];

const DEFAULT_REPLY =
  "Thank you for sharing that with me. I'm here to listen. What feels most " +
  "important to talk through right now?";

function pickReply(messages) {
  const lastUser = [...(messages ?? [])].reverse().find((m) => m.role === "user");
  const text = typeof lastUser?.content === "string" ? lastUser.content : "";
  return REPLIES.find((r) => r.pattern.test(text))?.reply ?? DEFAULT_REPLY;
}

function chunkPayload(id, created, model, delta, finishReason) {
  return JSON.stringify({
    id,
    object: "chat.completion.chunk",
    created,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason ?? null }],
  });
}

const server = http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end('{"status":"ok"}');
    return;
  }
  if (req.method !== "POST" || !req.url.replace(/\/+$/, "").endsWith("/chat/completions")) {
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: { message: `no route: ${req.method} ${req.url}` } }));
    return;
  }

  let body = "";
  req.on("data", (part) => (body += part));
  req.on("end", () => {
    let parsed = {};
    try {
      parsed = JSON.parse(body);
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "invalid JSON" } }));
      return;
    }

    const id = "chatcmpl-stub";
    const created = Math.floor(Date.now() / 1000);
    const model = parsed.model ?? "stub-model";

    if (parsed.stream) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      res.write(`data: ${chunkPayload(id, created, model, { role: "assistant", content: "" })}\n\n`);
      const words = pickReply(parsed.messages).split(" ");
      let i = 0;
      const timer = setInterval(() => {
        if (i < words.length) {
          const content = (i === 0 ? "" : " ") + words[i];
          res.write(`data: ${chunkPayload(id, created, model, { content })}\n\n`);
          i += 1;
        } else {
          clearInterval(timer);
          res.write(`data: ${chunkPayload(id, created, model, {}, "stop")}\n\n`);
          res.write("data: [DONE]\n\n");
          res.end();
        }
      }, 25);
      // res 'close' fires when the client goes away; req 'close' fires as soon
      // as the request body is consumed and would kill the timer immediately.
      res.on("close", () => clearInterval(timer));
    } else {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          id,
          object: "chat.completion",
          created,
          model,
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Feeling anxious lately" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
      );
    }
  });
});

const HOST = process.env.LLM_STUB_HOST ?? "127.0.0.1";
server.listen(PORT, HOST, () => {
  console.log(`llm-stub listening on http://${HOST}:${PORT}`);
});
