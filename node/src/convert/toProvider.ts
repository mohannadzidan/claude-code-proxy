import { randomUUID } from "node:crypto";
import type { ResolvedModel } from "../router/config.js";
import type { MessagesRequest, Tool } from "../schema/types.js";
import { thinkingIsEnabled } from "../schema/types.js";
import {
  isReasoningModel,
  supportsTemperature,
  systemRoleFor,
} from "../schema/modelHelpers.js";
import { cleanGeminiSchema } from "../schema/geminiSchema.js";
import { sanitizeJsonSchemaForOpenai } from "../schema/openaiSchema.js";
import { CUSTOM_HEADERS, DISABLE_STREAM_OPTIONS, MAX_TOKENS_LIMIT } from "../env.js";
import { logger } from "../logger.js";
import { effectiveCustomLlmProvider } from "../router/config.js";

/**
 * Shape produced by convertAnthropicToProviderRequest: an OpenAI Chat
 * Completions request body plus a handful of routing-only fields
 * (customLlmProvider/apiKey/apiBase/dropParams/extraHeaders) that never go on
 * the wire -- the provider adapter strips them before the HTTP call. This
 * mirrors how litellm itself splits routing kwargs from the outgoing body.
 */
export interface ProviderChatRequest {
  model: string;
  customLlmProvider: string;
  apiKey: string;
  apiBase?: string;
  messages: Record<string, unknown>[];
  stream: boolean;
  dropParams: true;
  maxTokens?: number;
  maxCompletionTokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop?: string[];
  thinking?: { type: string; budget_tokens?: number };
  output_config?: { effort: string };
  reasoning_effort?: string;
  stream_options?: { include_usage: boolean };
  user?: string;
  extraHeaders?: Record<string, string>;
  tools?: Record<string, unknown>[];
  tool_choice?: unknown;
  parallel_tool_calls?: boolean;
}

function blockGet(block: unknown, key: string, def: unknown = undefined): unknown {
  if (block && typeof block === "object") {
    const v = (block as Record<string, unknown>)[key];
    return v === undefined ? def : v;
  }
  return def;
}

/**
 * Convert an Anthropic image source into an OpenAI image_url part.
 */
function anthropicImageToOpenai(source: unknown): Record<string, unknown> | null {
  if (!source || typeof source !== "object") return null;
  const src = source as Record<string, unknown>;
  const srcType = src.type;
  if (srcType === "base64") {
    const mediaType = (src.media_type as string) || "image/png";
    const data = (src.data as string) || "";
    if (!data) return null;
    return { type: "image_url", image_url: { url: `data:${mediaType};base64,${data}` } };
  }
  if (srcType === "url" && src.url) {
    return { type: "image_url", image_url: { url: src.url } };
  }
  return null;
}

/**
 * Collapse to a plain string when there is no multimodal part. Plain strings
 * are accepted by every OpenAI-compatible server; the parts array is not
 * (older/self-hosted endpoints often reject it), so only use it when an image
 * actually needs to be carried.
 */
function normalizeOpenaiContent(
  parts: Record<string, unknown>[]
): string | Record<string, unknown>[] | null {
  if (parts.length === 0) return null;
  if (parts.every((p) => p.type === "text" && !("cache_control" in p))) {
    return parts
      .map((p) => (p.text as string) || "")
      .filter(Boolean)
      .join("\n");
  }
  return parts;
}

/** Flatten Anthropic tool_result content into the string OpenAI expects. */
function toolResultToText(content: unknown): string {
  if (content === null || content === undefined) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const chunks: string[] = [];
    for (const item of content) {
      const itemType = blockGet(item, "type");
      if (itemType === "text") {
        chunks.push((blockGet(item, "text", "") as string) || "");
      } else if (itemType === "image") {
        chunks.push("[image omitted: tool results cannot carry images]");
      } else if (typeof item === "string") {
        chunks.push(item);
      } else {
        try {
          chunks.push(JSON.stringify(item));
        } catch {
          chunks.push(String(item));
        }
      }
    }
    return chunks.filter(Boolean).join("\n");
  }
  if (content && typeof content === "object") {
    const obj = content as Record<string, unknown>;
    if (obj.type === "text") return (obj.text as string) || "";
    try {
      return JSON.stringify(obj);
    } catch {
      return String(obj);
    }
  }
  return String(content);
}

/**
 * Guarantee every assistant tool_call is followed by a matching tool message.
 *
 * Claude Code truncates history mid-loop and users hit ESC between a tool
 * call and its result, both of which leave dangling calls that OpenAI
 * rejects with "messages with role 'tool' must be a response to a preceding
 * message with 'tool_calls'".
 */
function repairToolCallPairs(messages: Record<string, unknown>[]): Record<string, unknown>[] {
  const repaired: Record<string, unknown>[] = [];
  for (let idx = 0; idx < messages.length; idx++) {
    const msg = messages[idx];
    repaired.push(msg);
    const toolCalls = msg.role === "assistant" ? (msg.tool_calls as Record<string, unknown>[] | undefined) : undefined;
    if (!toolCalls || toolCalls.length === 0) continue;

    const replied = new Set<unknown>();
    let j = idx + 1;
    while (j < messages.length && messages[j].role === "tool") {
      replied.add(messages[j].tool_call_id);
      j++;
    }

    for (const call of toolCalls) {
      if (!replied.has(call.id)) {
        logger.debug(`Synthesizing missing tool result for ${call.id}`);
        repaired.push({
          role: "tool",
          tool_call_id: call.id,
          content: "(tool call was interrupted; no result returned)",
        });
      }
    }
  }
  return repaired;
}

const OPENAI_MESSAGE_KEYS = new Set(["role", "content", "name", "tool_call_id", "tool_calls"]);

/**
 * Last-mile sanity pass on the outgoing message array (mutates in place).
 * Everything structural is already handled by the converter; this only
 * guards against the invariants that make OpenAI-compatible servers return a
 * bare 400.
 */
export function validateOpenaiMessages(providerRequest: ProviderChatRequest): void {
  const messages = providerRequest.messages || [];
  const cleaned: Record<string, unknown>[] = [];

  for (const msg of messages) {
    for (const key of Object.keys(msg)) {
      if (!OPENAI_MESSAGE_KEYS.has(key)) {
        logger.debug(`Removing unsupported message field: ${key}`);
        delete msg[key];
      }
    }

    const role = msg.role;
    const content = msg.content;

    if (content === null || content === undefined) {
      if (!(role === "assistant" && msg.tool_calls)) {
        msg.content = "";
      }
    }

    if (role === "assistant" && !msg.tool_calls) {
      const body = msg.content;
      if (typeof body === "string" && !body.trim()) continue;
    }

    if (role === "tool" && !msg.tool_call_id) {
      logger.warn("Dropping tool message with no tool_call_id");
      continue;
    }

    cleaned.push(msg);
  }

  if (cleaned.length === 0) {
    cleaned.push({ role: "user", content: "(empty request)" });
  }

  const last = cleaned[cleaned.length - 1];
  if (last.role === "assistant" && last.tool_calls) {
    for (const call of last.tool_calls as Record<string, unknown>[]) {
      cleaned.push({ role: "tool", tool_call_id: call.id, content: "(no result returned)" });
    }
  }

  providerRequest.messages = cleaned;
}

/**
 * True when `reasoning_effort` should be forwarded for this upstream model.
 *
 * Only consulted once the client has already requested extended thinking, so
 * defaulting to true is correct for the common case: most OpenAI-compatible
 * gateway models (glm, minimax, nemotron, ...) are reasoning-capable even
 * though their name doesn't match the OpenAI o1-o4/gpt-5 pattern. Set
 * `disable_reasoning: true` on a model entry for the rare non-reasoning model
 * that should ignore thinking requests instead.
 */
function supportsReasoningEffort(resolved: ResolvedModel): boolean {
  return !resolved.disableReasoning;
}

/**
 * Read the /effort level Claude Code actually varies.
 *
 * `thinking.budget_tokens` is not it: for any model name Claude Code doesn't
 * recognize as an older fixed-thinking model (every router.yaml id/alias
 * qualifies), it sends adaptive thinking with no budget_tokens at all and
 * carries the effort level in the separate `output_config.effort` field
 * instead. Passed through as-is unless the model entry's optional
 * reasoningEffortMap remaps it.
 */
function clientEffortLevel(anthropicRequest: MessagesRequest, resolved: ResolvedModel): string | null {
  const outputConfig = (anthropicRequest as unknown as Record<string, unknown>).output_config;
  if (!outputConfig || typeof outputConfig !== "object") return null;
  const effort = (outputConfig as Record<string, unknown>).effort;
  if (typeof effort !== "string" || !effort) return null;
  return (resolved.reasoningEffortMap || {})[effort] ?? effort;
}

/**
 * Convert an Anthropic Messages request into OpenAI/provider-agnostic shape.
 *
 * The important part is that Anthropic tool_use / tool_result blocks become
 * real OpenAI `tool_calls` and `role:"tool"` messages, not stringified prose:
 * that would destroy the tool-calling contract and degrade multi-turn
 * agentic loops into the model re-describing tools in prose.
 *
 * `resolved` is the router.yaml entry chosen for this request's tier: it
 * supplies the exact upstream model name/provider/credentials, separate from
 * whatever Claude model id the client sent.
 */
export function convertAnthropicToProviderRequest(
  anthropicRequest: MessagesRequest,
  resolved: ResolvedModel
): ProviderChatRequest {
  const targetModel = resolved.litellmModel;
  const isAnthropicModel = resolved.litellmProvider === "anthropic";
  const isGeminiModel = resolved.litellmProvider === "gemini";
  const messages: Record<string, unknown>[] = [];

  // ---------- system ----------
  let systemText = "";
  // Only collected/forwarded for isAnthropicModel: cache_control is an
  // Anthropic-specific field. Sending it to an OpenAI-compatible backend
  // would just be an unrecognized field, so plain OpenAI-shape backends keep
  // getting the flattened systemText string as before.
  const systemBlocks: Record<string, unknown>[] = [];
  if (anthropicRequest.system) {
    if (typeof anthropicRequest.system === "string") {
      systemText = anthropicRequest.system;
    } else if (Array.isArray(anthropicRequest.system)) {
      const parts: string[] = [];
      for (const block of anthropicRequest.system) {
        if (blockGet(block, "type") === "text") {
          const text = (blockGet(block, "text", "") as string) || "";
          if (!text) continue;
          parts.push(text);
          if (isAnthropicModel) {
            const blockDict: Record<string, unknown> = { type: "text", text };
            const cacheControl = blockGet(block, "cache_control");
            if (cacheControl !== undefined) blockDict.cache_control = cacheControl;
            systemBlocks.push(blockDict);
          }
        }
      }
      systemText = parts.filter(Boolean).join("\n\n");
    }
  }

  const systemRole = systemRoleFor(targetModel);
  let pendingSystemPrefix = "";
  if (systemText.trim()) {
    if (systemRole === null) {
      // Fold into the first user turn for models with no system role.
      pendingSystemPrefix = systemText.trim() + "\n\n";
    } else {
      const hasCacheControl = systemBlocks.some((b) => "cache_control" in b);
      const content = hasCacheControl ? systemBlocks : systemText.trim();
      messages.push({ role: systemRole, content });
    }
  }

  // ---------- conversation ----------
  // Track which tool_use ids the assistant actually emitted. OpenAI returns a
  // hard 400 for a tool message whose tool_call_id was never announced, and
  // for an assistant tool_calls entry with no matching tool reply.
  const announcedToolIds = new Set<string>();

  for (const msg of anthropicRequest.messages) {
    const role = msg.role === "user" || msg.role === "assistant" ? msg.role : "user";
    const content = msg.content;

    if (typeof content === "string") {
      let text = content;
      if (pendingSystemPrefix && role === "user") {
        text = pendingSystemPrefix + text;
        pendingSystemPrefix = "";
      }
      messages.push({ role, content: text });
      continue;
    }

    const blocks = content || [];

    if (role === "assistant") {
      const textParts: Record<string, unknown>[] = [];
      const toolCalls: Record<string, unknown>[] = [];
      for (const block of blocks) {
        const btype = blockGet(block, "type");
        if (btype === "text") {
          const t = (blockGet(block, "text", "") as string) || "";
          if (t) {
            const part: Record<string, unknown> = { type: "text", text: t };
            if (isAnthropicModel) {
              const cacheControl = blockGet(block, "cache_control");
              if (cacheControl !== undefined) part.cache_control = cacheControl;
            }
            textParts.push(part);
          }
        } else if (btype === "tool_use") {
          const toolId = (blockGet(block, "id") as string) || `call_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
          let toolInput = blockGet(block, "input", {});
          if (!toolInput || typeof toolInput !== "object" || Array.isArray(toolInput)) {
            toolInput = { value: toolInput };
          }
          announcedToolIds.add(toolId);
          toolCalls.push({
            id: toolId,
            type: "function",
            function: {
              name: (blockGet(block, "name", "") as string) || "unknown",
              arguments: JSON.stringify(toolInput),
            },
          });
        } else if (btype === "thinking" || btype === "redacted_thinking") {
          // Reasoning traces are provider-specific and are not replayable as
          // assistant content on OpenAI; drop them from history.
          continue;
        }
      }

      const assistantMsg: Record<string, unknown> = { role: "assistant" };
      const normalized = normalizeOpenaiContent(textParts);
      if (toolCalls.length > 0) {
        // content may be null alongside tool_calls, and must be when empty.
        assistantMsg.content = normalized || null;
        assistantMsg.tool_calls = toolCalls;
      } else {
        assistantMsg.content = normalized || "";
      }
      messages.push(assistantMsg);
      continue;
    }

    // ----- user turn -----
    // Tool results must be emitted as their own `tool` messages, immediately
    // after the assistant call and before any new user text.
    const toolMessages: Record<string, unknown>[] = [];
    const userParts: Record<string, unknown>[] = [];

    for (const block of blocks) {
      const btype = blockGet(block, "type");
      if (btype === "tool_result") {
        const toolUseId = (blockGet(block, "tool_use_id") as string) || "";
        let resultText = toolResultToText(blockGet(block, "content", ""));
        if (blockGet(block, "is_error")) {
          resultText = resultText ? `Error: ${resultText}` : "Error";
        }
        if (!resultText) resultText = "(no output)";
        if (toolUseId && announcedToolIds.has(toolUseId)) {
          const toolMsg: Record<string, unknown> = {
            role: "tool",
            tool_call_id: toolUseId,
            content: resultText,
          };
          if (isAnthropicModel) {
            const cacheControl = blockGet(block, "cache_control");
            if (cacheControl !== undefined) toolMsg.cache_control = cacheControl;
          }
          toolMessages.push(toolMsg);
        } else {
          // Orphaned result (no matching call in this window, e.g. after
          // client-side history truncation). Demote to user text rather than
          // letting the backend 400 the whole request.
          logger.debug(`Orphan tool_result ${JSON.stringify(toolUseId)}; demoting to user text`);
          userParts.push({ type: "text", text: `Tool result:\n${resultText}` });
        }
      } else if (btype === "text") {
        const t = (blockGet(block, "text", "") as string) || "";
        if (t) {
          const part: Record<string, unknown> = { type: "text", text: t };
          if (isAnthropicModel) {
            const cacheControl = blockGet(block, "cache_control");
            if (cacheControl !== undefined) part.cache_control = cacheControl;
          }
          userParts.push(part);
        }
      } else if (btype === "image") {
        const part = anthropicImageToOpenai(blockGet(block, "source"));
        if (part) userParts.push(part);
      } else if (btype === "document") {
        const src = (blockGet(block, "source") as Record<string, unknown>) || {};
        if (src.type === "text") {
          userParts.push({ type: "text", text: (src.data as string) || "" });
        } else {
          userParts.push({ type: "text", text: "[document omitted: unsupported by backend]" });
        }
      }
    }

    messages.push(...toolMessages);

    if (pendingSystemPrefix && userParts.length > 0) {
      userParts.unshift({ type: "text", text: pendingSystemPrefix.trim() });
      pendingSystemPrefix = "";
    }

    const normalizedUser = normalizeOpenaiContent(userParts);
    if (normalizedUser) {
      messages.push({ role: "user", content: normalizedUser });
    }
  }

  // Any tool_call the assistant announced but that has no reply must be
  // answered, or OpenAI rejects the conversation outright.
  const repairedMessages = repairToolCallPairs(messages);

  // ---------- parameters ----------
  // Passed through as the client sent it. MAX_TOKENS_LIMIT is the only cap,
  // and it's opt-in -- set it explicitly if a gateway actually needs a ceiling.
  let maxTokens = anthropicRequest.max_tokens;
  if (MAX_TOKENS_LIMIT) maxTokens = Math.min(maxTokens, MAX_TOKENS_LIMIT);

  const providerRequest: ProviderChatRequest = {
    model: targetModel,
    customLlmProvider: effectiveCustomLlmProvider(resolved),
    apiKey: resolved.providerApiKey,
    messages: repairedMessages,
    stream: Boolean(anthropicRequest.stream),
    dropParams: true,
  };
  if (resolved.providerBaseUrl) providerRequest.apiBase = resolved.providerBaseUrl;

  // Reasoning models require max_completion_tokens; other models use max_tokens.
  if (isReasoningModel(targetModel)) {
    providerRequest.maxCompletionTokens = maxTokens;
  } else {
    providerRequest.maxTokens = maxTokens;
  }

  // Reasoning models reject temperature / top_p / top_k / stop outright.
  if (supportsTemperature(targetModel)) {
    if (anthropicRequest.temperature !== null && anthropicRequest.temperature !== undefined) {
      providerRequest.temperature = anthropicRequest.temperature;
    }
    if (anthropicRequest.top_p !== null && anthropicRequest.top_p !== undefined) {
      providerRequest.top_p = anthropicRequest.top_p;
    }
    if (anthropicRequest.top_k !== null && anthropicRequest.top_k !== undefined && isGeminiModel) {
      // top_k is not an OpenAI parameter; only forward where it is real.
      providerRequest.top_k = anthropicRequest.top_k;
    }
    if (anthropicRequest.stop_sequences && anthropicRequest.stop_sequences.length > 0) {
      // OpenAI caps `stop` at 4 entries.
      providerRequest.stop = anthropicRequest.stop_sequences.slice(0, 4);
    }
  }

  // Map Anthropic's thinking budget onto OpenAI's reasoning_effort.
  //
  // Claude Code's `/effort` command does NOT vary `thinking.budget_tokens`.
  // For any model name it doesn't recognize as an older fixed-thinking model
  // -- which is every router.yaml id/alias -- it sends adaptive thinking
  // (`thinking: {"type": "adaptive"}`, no budget_tokens at all) and carries
  // the actual effort level in the separate `output_config.effort` field
  // instead, so clientEffort takes priority over the budget_tokens heuristic,
  // which now only matters for older Anthropic clients with an explicit
  // numeric budget and no effort field.
  const clientEffort = clientEffortLevel(anthropicRequest, resolved);
  if (anthropicRequest.thinking && thinkingIsEnabled(anthropicRequest.thinking)) {
    if (isAnthropicModel) {
      const thinkingPayload: { type: string; budget_tokens?: number } = { type: "enabled" };
      if (anthropicRequest.thinking.budget_tokens) {
        thinkingPayload.budget_tokens = anthropicRequest.thinking.budget_tokens;
      }
      providerRequest.thinking = thinkingPayload;
      if (clientEffort) {
        providerRequest.output_config = { effort: clientEffort };
      }
    } else if (supportsReasoningEffort(resolved)) {
      if (clientEffort) {
        providerRequest.reasoning_effort = clientEffort;
      } else {
        const budget = anthropicRequest.thinking.budget_tokens || 0;
        if (budget && budget <= 4096) providerRequest.reasoning_effort = "low";
        else if (budget && budget >= 16384) providerRequest.reasoning_effort = "high";
        else providerRequest.reasoning_effort = "medium";
      }
    }
  }

  // Streaming usage is opt-in on OpenAI. Without this the proxy reported
  // output_tokens: 0 for every streamed reply. Some third-party gateways
  // reject the unknown field, so it can be turned off with
  // DISABLE_STREAM_OPTIONS=true.
  if (anthropicRequest.stream && !DISABLE_STREAM_OPTIONS) {
    providerRequest.stream_options = { include_usage: true };
  }

  // metadata.user_id -> `user` (abuse-tracking / caching hints)
  if (anthropicRequest.metadata && typeof anthropicRequest.metadata === "object") {
    const userId = (anthropicRequest.metadata as Record<string, unknown>).user_id;
    if (typeof userId === "string" && userId) providerRequest.user = userId.slice(0, 128);
  }

  const extraHeaders: Record<string, string> = {};
  if (resolved.requestHeaders) Object.assign(extraHeaders, resolved.requestHeaders);
  if (CUSTOM_HEADERS && Object.keys(CUSTOM_HEADERS).length) Object.assign(extraHeaders, CUSTOM_HEADERS);
  if (Object.keys(extraHeaders).length) providerRequest.extraHeaders = extraHeaders;

  // ---------- tools ----------
  if (anthropicRequest.tools && anthropicRequest.tools.length > 0) {
    const openaiTools: Record<string, unknown>[] = [];
    for (const tool of anthropicRequest.tools as (Tool & { cache_control?: unknown })[]) {
      const name = tool.name;
      if (!name) continue;
      // Anthropic server-side tools have a `type` and no usable schema.
      if (tool.type && !tool.input_schema) {
        logger.warn(`Dropping unsupported server-side tool: ${name}`);
        continue;
      }

      let inputSchema: unknown = tool.input_schema && Object.keys(tool.input_schema).length
        ? tool.input_schema
        : { type: "object", properties: {} };
      inputSchema = isGeminiModel ? cleanGeminiSchema(inputSchema) : sanitizeJsonSchemaForOpenai(inputSchema);

      const openaiTool: Record<string, unknown> = {
        type: "function",
        function: {
          name,
          description: tool.description || "",
          parameters: inputSchema,
        },
      };
      if (isAnthropicModel && tool.cache_control !== undefined && tool.cache_control !== null) {
        // Claude Code marks the *last* tool with cache_control to cache the
        // whole tool-definitions prefix.
        openaiTool.cache_control = tool.cache_control;
      }
      openaiTools.push(openaiTool);
    }
    if (openaiTools.length > 0) providerRequest.tools = openaiTools;
  }

  // ---------- tool_choice ----------
  if (anthropicRequest.tool_choice !== null && anthropicRequest.tool_choice !== undefined && providerRequest.tools) {
    const tc = anthropicRequest.tool_choice;
    if (typeof tc === "string") {
      providerRequest.tool_choice = tc;
    } else {
      const choiceType = (tc as Record<string, unknown>).type;
      if (choiceType === "auto") {
        providerRequest.tool_choice = "auto";
      } else if (choiceType === "any") {
        // Anthropic "any" == must call some tool == OpenAI "required".
        providerRequest.tool_choice = "required";
      } else if (choiceType === "none") {
        providerRequest.tool_choice = "none";
      } else if (choiceType === "tool" && (tc as Record<string, unknown>).name) {
        providerRequest.tool_choice = {
          type: "function",
          function: { name: (tc as Record<string, unknown>).name },
        };
      } else {
        providerRequest.tool_choice = "auto";
      }

      // Anthropic disables parallel calls via the tool_choice object.
      if ((tc as Record<string, unknown>).disable_parallel_tool_use === true) {
        providerRequest.parallel_tool_calls = false;
      }
    }
  }

  return providerRequest;
}
