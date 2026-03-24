/**
 * Reads an SSE stream from the MindBridge backend.
 * Backend sends lines: `data: <token_text>\n\n`
 * Ends with: `data: [DONE]\n\n`
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
      const data = line.slice(6).trim();
      if (data === "[DONE]") return;
      if (data.startsWith("[ERROR]")) throw new Error(data.slice(8));
      yield data;
    }
  }
}
