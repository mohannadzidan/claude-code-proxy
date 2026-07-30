import { randomUUID } from "node:crypto";
import type { MessagesRequest, MessagesResponse, Usage } from "../schema/types.js";
import { PROXY_THINKING_SIGNATURE } from "../env.js";
import { logger } from "../logger.js";
import { clientToolId } from "./toolId.js";

/** Anthropic ids start with msg_; some SDK paths assert on that prefix. */
export function anthropicMsgId(rawId: unknown): string {
  if (typeof rawId === "string" && rawId.startsWith("msg_")) return rawId;
  let suffix = randomUUID().replace(/-/g, "").slice(0, 24);
  if (typeof rawId === "string" && rawId) {
    const cleaned = rawId.replace(/[^A-Za-z0-9]/g, "").slice(-24);
    if (cleaned) suffix = cleaned;
  }
  return `msg_${suffix}`;
}

/**
 * Return [promptTokens, completionTokens, cacheReadTokens].
 *
 * Prompt-cache hits are reported by OpenAI under
 * prompt_tokens_details.cached_tokens; forwarding them lets Claude Code show
 * real cache savings instead of a flat zero.
 */
export function extractUsage(usageInfo: unknown): [number, number, number] {
  if (!usageInfo || typeof usageInfo !== "object") return [0, 0, 0];
  const u = usageInfo as Record<string, unknown>;

  let promptTokens = (u.prompt_tokens as number) || 0;
  const completionTokens = (u.completion_tokens as number) || 0;

  let cached = 0;
  const details = u.prompt_tokens_details;
  if (details && typeof details === "object") {
    cached = ((details as Record<string, unknown>).cached_tokens as number) || 0;
  }
  if (!cached) cached = (u.cache_read_input_tokens as number) || 0;

  // Anthropic counts cache reads separately from input_tokens; OpenAI
  // includes them, so subtract to keep the totals consistent.
  if (cached && cached <= promptTokens) promptTokens -= cached;

  return [promptTokens, completionTokens, cached];
}

/**
 * Map an OpenAI finish_reason onto an Anthropic stop_reason.
 *
 * `hasToolUse` is not optional in practice. Many OpenAI-compatible backends
 * report finish_reason "stop" even when the response carries tool_calls.
 * Passing that through as "end_turn" tells the client the turn is over, so
 * it delivers the tool_use block but never executes it. If we emitted a
 * tool_use block, the turn is a tool turn, whatever the backend called it.
 */
export function mapFinishReason(finishReason: unknown, hasToolUse = false): string {
  const mapping: Record<string, string> = {
    stop: "end_turn",
    length: "max_tokens",
    max_tokens: "max_tokens",
    tool_calls: "tool_use",
    function_call: "tool_use",
    content_filter: "refusal",
  };
  const reason = mapping[finishReason as string] || "end_turn";

  if (hasToolUse && reason !== "max_tokens" && reason !== "refusal") {
    if (reason !== "tool_use") {
      logger.debug(`Overriding stop_reason '${reason}' -> 'tool_use' (backend reported finish_reason=${JSON.stringify(finishReason)} with tool calls)`);
    }
    return "tool_use";
  }
  return reason;
}

interface ProviderMessage {
  content?: string | null;
  tool_calls?: Array<{ id?: string; function?: { name?: string; arguments?: string } }> | null;
  reasoning_content?: string | null;
  reasoning?: string | null;
}

export interface ProviderResponse {
  id?: string;
  choices?: Array<{ message?: ProviderMessage; finish_reason?: string | null }>;
  usage?: unknown;
}

/** Convert an OpenAI-shaped (litellm-normalized) response to Anthropic API response format. */
export function convertProviderToAnthropic(
  providerResponse: ProviderResponse,
  originalRequest: MessagesRequest
): MessagesResponse {
  try {
    const choices = providerResponse.choices || [];
    const message = choices.length > 0 ? choices[0].message || {} : {};
    let contentText = message.content ?? "";
    const toolCalls = message.tool_calls ?? null;
    const reasoningText = message.reasoning_content || message.reasoning || null;
    const finishReason = choices.length > 0 ? choices[0].finish_reason || "stop" : "stop";
    const usageInfo = providerResponse.usage;
    const responseId = providerResponse.id || `msg_${randomUUID()}`;

    const content: Record<string, unknown>[] = [];

    // Thinking must precede text in an Anthropic content array.
    if (reasoningText) {
      content.push({ type: "thinking", thinking: reasoningText, signature: PROXY_THINKING_SIGNATURE });
    }

    if (contentText !== null && contentText !== "") {
      content.push({ type: "text", text: contentText });
    }

    if (toolCalls && toolCalls.length > 0) {
      for (const toolCall of toolCalls) {
        const func = toolCall.function || {};
        const toolId = toolCall.id || `tool_${randomUUID()}`;
        const name = func.name || "";
        let args: unknown = func.arguments ?? "{}";

        if (typeof args === "string") {
          try {
            args = JSON.parse(args);
          } catch {
            logger.warn(`Failed to parse tool arguments as JSON: ${args}`);
            args = { raw: args };
          }
        }

        content.push({
          type: "tool_use",
          id: clientToolId(toolId, name),
          name,
          input: args,
        });
      }
    }

    const [promptTokens, completionTokens, cacheReadTokens] = extractUsage(usageInfo);

    let stopReason = mapFinishReason(
      finishReason,
      content.some((b) => b.type === "tool_use")
    );

    // Anthropic reports which stop sequence fired; OpenAI does not, so detect it.
    let stopSequence: string | null = null;
    if (originalRequest.stop_sequences && contentText) {
      for (const seq of originalRequest.stop_sequences) {
        if (seq && contentText.endsWith(seq)) {
          stopReason = "stop_sequence";
          stopSequence = seq;
          contentText = contentText.slice(0, -seq.length);
          for (const blk of content) {
            if (blk.type === "text") blk.text = contentText;
          }
          break;
        }
      }
    }

    if (content.length === 0) content.push({ type: "text", text: "" });

    // Echo the model id the client asked for, not the internal mapped name.
    const responseModel = originalRequest.original_model || originalRequest.model;

    const usage: Usage = {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
      cache_read_input_tokens: cacheReadTokens,
    };

    return {
      id: anthropicMsgId(responseId),
      model: responseModel,
      role: "assistant",
      content,
      type: "message",
      stop_reason: stopReason as MessagesResponse["stop_reason"],
      stop_sequence: stopSequence,
      usage,
    };
  } catch (e) {
    logger.error(`Error converting response: ${(e as Error).message}\n${(e as Error).stack}`);
    return {
      id: `msg_${randomUUID()}`,
      model: originalRequest.model,
      role: "assistant",
      content: [{ type: "text", text: `Error converting response: ${(e as Error).message}. Please check server logs.` }],
      type: "message",
      stop_reason: "end_turn",
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    };
  }
}
