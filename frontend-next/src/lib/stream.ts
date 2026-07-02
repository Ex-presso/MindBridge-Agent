/**
 * Reads an SSE stream from the MindBridge backend.
 * Backend sends JSON frames: `data: {"delta": "<token>"}\n\n`
 * Errors arrive as `data: {"error": "<msg>"}\n\n`; the stream ends with
 * `data: [DONE]\n\n`. JSON-encoding lets a token contain newlines without
 * breaking the `\n\n` frame delimiter.
 */
export async function* readSSEStream(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data.trim() === "[DONE]") return;
      const parsed = JSON.parse(data) as { delta?: string; error?: string };
      if (parsed.error) throw new Error(parsed.error);
      if (parsed.delta) yield parsed.delta;
    }
  }
}
