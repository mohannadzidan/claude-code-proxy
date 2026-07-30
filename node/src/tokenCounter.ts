import { encode, encodeChat } from "gpt-tokenizer";

function contentToText(content: unknown): string {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  try {
    return JSON.stringify(content);
  } catch {
    return String(content);
  }
}

/**
 * Approximate prompt token count for a converted (OpenAI-shaped) message
 * array using the cl100k_base tokenizer and OpenAI's documented per-message
 * chat overhead formula. This is deliberately an approximation: like
 * server.py's litellm.token_counter() fallback for unrecognized/custom
 * gateway models, there is no way to know the real tokenizer a third-party
 * backend uses, so a generic-but-accurate-enough count is the best
 * available.
 */
export function approxTokenCount(messages: Record<string, unknown>[]): number {
  const chatMessages = messages.map((msg) => {
    const rawRole = typeof msg.role === "string" ? msg.role : "user";
    const role: "system" | "user" | "assistant" =
      rawRole === "system" || rawRole === "assistant" ? rawRole : "user";
    let text = contentToText(msg.content);
    if (msg.tool_calls) text += `\n${contentToText(msg.tool_calls)}`;
    if (msg.tool_call_id) text += `\n${contentToText(msg.tool_call_id)}`;
    return { role, name: typeof msg.name === "string" ? msg.name : undefined, content: text };
  });
  // encodeChat requires an explicit model to pick its per-message overhead
  // constants; the actual upstream model is usually not one gpt-tokenizer
  // knows about, so gpt-3.5-turbo's formula is used as a fixed, reasonable
  // approximation regardless of target model.
  return encodeChat(chatMessages, "gpt-3.5-turbo").length;
}

export function approxTextTokens(text: string): number {
  return encode(text).length;
}
