import type { ProviderChatRequest } from "../convert/toProvider.js";
import type { ProviderStreamChunk } from "./types.js";
import type { ProviderResponse } from "../convert/fromProvider.js";
import { ProviderHttpError } from "./httpError.js";

const DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1";

/** Build the outgoing OpenAI Chat Completions wire body from our internal request shape. */
function buildBody(req: ProviderChatRequest): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: req.model,
    messages: req.messages,
    stream: req.stream,
  };
  if (req.maxCompletionTokens !== undefined) body.max_completion_tokens = req.maxCompletionTokens;
  if (req.maxTokens !== undefined) body.max_tokens = req.maxTokens;
  if (req.temperature !== undefined) body.temperature = req.temperature;
  if (req.top_p !== undefined) body.top_p = req.top_p;
  if (req.stop !== undefined) body.stop = req.stop;
  if (req.reasoning_effort !== undefined) body.reasoning_effort = req.reasoning_effort;
  if (req.stream_options !== undefined) body.stream_options = req.stream_options;
  if (req.user !== undefined) body.user = req.user;
  if (req.tools !== undefined) body.tools = req.tools;
  if (req.tool_choice !== undefined) body.tool_choice = req.tool_choice;
  if (req.parallel_tool_calls !== undefined) body.parallel_tool_calls = req.parallel_tool_calls;
  return body;
}

function buildUrl(req: ProviderChatRequest): string {
  const base = (req.apiBase || DEFAULT_OPENAI_BASE_URL).replace(/\/+$/, "");
  return `${base}/chat/completions`;
}

function buildHeaders(req: ProviderChatRequest): Record<string, string> {
  return {
    "content-type": "application/json",
    authorization: `Bearer ${req.apiKey}`,
    ...(req.extraHeaders || {}),
  };
}

async function raiseForStatus(resp: Response): Promise<void> {
  if (resp.ok) return;
  const text = await resp.text();
  let parsed: unknown = text;
  try {
    parsed = JSON.parse(text);
  } catch {
    // leave as text
  }
  const message =
    parsed && typeof parsed === "object" && "error" in (parsed as Record<string, unknown>)
      ? JSON.stringify((parsed as Record<string, unknown>).error)
      : text || `HTTP ${resp.status}`;
  throw new ProviderHttpError(resp.status, message, parsed);
}

/** Non-streaming Chat Completions call against an OpenAI-compatible endpoint. */
export async function completeChatOpenAi(req: ProviderChatRequest): Promise<ProviderResponse> {
  const resp = await fetch(buildUrl(req), {
    method: "POST",
    headers: buildHeaders(req),
    body: JSON.stringify(buildBody(req)),
  });
  await raiseForStatus(resp);
  return (await resp.json()) as ProviderResponse;
}

/**
 * Streaming Chat Completions call. Parses the OpenAI-style
 * `data: {...}\n\n` SSE body into individual chunk objects, stopping at the
 * `data: [DONE]` sentinel.
 */
export async function* streamChatOpenAi(req: ProviderChatRequest): AsyncGenerator<ProviderStreamChunk> {
  const resp = await fetch(buildUrl(req), {
    method: "POST",
    headers: buildHeaders(req),
    body: JSON.stringify(buildBody(req)),
  });
  await raiseForStatus(resp);
  if (!resp.body) return;

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const rawEvent = buf.slice(0, idx);
        buf = buf.slice(idx + 2);

        for (const line of rawEvent.split("\n")) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const data = trimmed.slice(5).trim();
          if (data === "[DONE]") return;
          if (!data) continue;
          try {
            yield JSON.parse(data) as ProviderStreamChunk;
          } catch {
            // Malformed/partial JSON on a line the server chose to split
            // oddly; skip rather than crash the whole stream.
            continue;
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
