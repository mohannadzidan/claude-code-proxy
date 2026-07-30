import { z } from "zod";
import { extractTier, type ModelTier } from "./modelHelpers.js";

// Every content block model allows extra fields. Real Anthropic clients
// (Claude Code in particular) attach `cache_control`, `citations`,
// `signature`, etc. to blocks. Rejecting those is the single most common
// cause of "the proxy works with curl but not with Claude Code".
const Passthrough = <T extends z.ZodRawShape>(shape: T) => z.object(shape).passthrough();

export const ContentBlockTextSchema = Passthrough({
  type: z.literal("text"),
  text: z.string(),
});

export const ContentBlockImageSchema = Passthrough({
  type: z.literal("image"),
  source: z.record(z.string(), z.any()),
});

export const ContentBlockDocumentSchema = Passthrough({
  type: z.literal("document"),
  source: z.record(z.string(), z.any()),
  title: z.string().nullable().optional(),
  context: z.string().nullable().optional(),
});

export const ContentBlockThinkingSchema = Passthrough({
  type: z.literal("thinking"),
  thinking: z.string().default(""),
  signature: z.string().nullable().optional(),
});

export const ContentBlockRedactedThinkingSchema = Passthrough({
  type: z.literal("redacted_thinking"),
  data: z.string().nullable().optional(),
});

export const ContentBlockToolUseSchema = Passthrough({
  type: z.literal("tool_use"),
  id: z.string(),
  name: z.string(),
  input: z.record(z.string(), z.any()).default({}),
});

export const ContentBlockToolResultSchema = Passthrough({
  type: z.literal("tool_result"),
  tool_use_id: z.string(),
  content: z.any().default(""),
  is_error: z.boolean().nullable().optional(),
});

// Any block type not modeled explicitly (server_tool_use,
// web_search_tool_result, mcp_tool_use, future additions...) falls through to
// this instead of being rejected.
export const ContentBlockUnknownSchema = Passthrough({
  type: z.string(),
});

export const AnyContentBlockSchema = z.union([
  ContentBlockTextSchema,
  ContentBlockImageSchema,
  ContentBlockDocumentSchema,
  ContentBlockThinkingSchema,
  ContentBlockRedactedThinkingSchema,
  ContentBlockToolUseSchema,
  ContentBlockToolResultSchema,
  ContentBlockUnknownSchema,
]);
export type AnyContentBlock = z.infer<typeof AnyContentBlockSchema>;

export const SystemContentSchema = Passthrough({
  type: z.literal("text"),
  text: z.string(),
});

// Anthropic's Messages API only accepts user/assistant here; system goes in
// the top-level `system` field. Permissive on input; normalized later.
export const MessageSchema = Passthrough({
  role: z.string(),
  content: z.union([z.string(), z.array(AnyContentBlockSchema)]),
});
export type Message = z.infer<typeof MessageSchema>;

export const ToolSchema = Passthrough({
  name: z.string(),
  description: z.string().nullable().optional(),
  input_schema: z.record(z.string(), z.any()).default({}),
  // Anthropic server-side tools (web_search, text_editor, bash...) carry a
  // `type` instead of a schema. They have no OpenAI equivalent and are dropped.
  type: z.string().nullable().optional(),
});
export type Tool = z.infer<typeof ToolSchema>;

// Anthropic sends {"type": "enabled"|"disabled", "budget_tokens": N}.
export const ThinkingConfigSchema = Passthrough({
  type: z.string().nullable().optional(),
  budget_tokens: z.number().int().nullable().optional(),
  enabled: z.boolean().default(true),
});
export type ThinkingConfig = z.infer<typeof ThinkingConfigSchema>;

export function thinkingIsEnabled(t: ThinkingConfig): boolean {
  if (t.type !== null && t.type !== undefined) return t.type !== "disabled";
  return t.enabled;
}

const MessagesRequestBase = Passthrough({
  model: z.string(),
  max_tokens: z.number().int(),
  messages: z.array(MessageSchema),
  system: z.union([z.string(), z.array(SystemContentSchema)]).nullable().optional(),
  stop_sequences: z.array(z.string()).nullable().optional(),
  stream: z.boolean().nullable().optional().default(false),
  // Default undefined (not 1.0): sending an explicit temperature to a
  // reasoning model is a hard 400, and forwarding a default the caller never
  // set is wrong anyway.
  temperature: z.number().nullable().optional(),
  top_p: z.number().nullable().optional(),
  top_k: z.number().int().nullable().optional(),
  metadata: z.record(z.string(), z.any()).nullable().optional(),
  tools: z.array(ToolSchema).nullable().optional(),
  tool_choice: z.union([z.record(z.string(), z.any()), z.string()]).nullable().optional(),
  thinking: ThinkingConfigSchema.nullable().optional(),
  original_model: z.string().nullable().optional(),
  model_tier: z.enum(["haiku", "sonnet", "opus"]).nullable().optional(),
});

export type MessagesRequest = z.infer<typeof MessagesRequestBase>;

/**
 * Parse + apply the two model_validators server.py ran on MessagesRequest:
 * capture_original_model (before) and compute_model_tier (after).
 */
export function parseMessagesRequest(data: unknown): MessagesRequest {
  const obj = { ...(data as Record<string, unknown>) };
  if (obj.model !== undefined && obj.model !== null && obj.original_model === undefined) {
    obj.original_model = obj.model;
  }
  const parsed = MessagesRequestBase.parse(obj);
  parsed.model_tier = extractTier(parsed.original_model || parsed.model) as ModelTier | null;
  return parsed;
}

const TokenCountRequestBase = Passthrough({
  model: z.string(),
  messages: z.array(MessageSchema),
  system: z.union([z.string(), z.array(SystemContentSchema)]).nullable().optional(),
  tools: z.array(ToolSchema).nullable().optional(),
  thinking: ThinkingConfigSchema.nullable().optional(),
  tool_choice: z.union([z.record(z.string(), z.any()), z.string()]).nullable().optional(),
  original_model: z.string().nullable().optional(),
  model_tier: z.enum(["haiku", "sonnet", "opus"]).nullable().optional(),
});
export type TokenCountRequest = z.infer<typeof TokenCountRequestBase>;

export function parseTokenCountRequest(data: unknown): TokenCountRequest {
  const obj = { ...(data as Record<string, unknown>) };
  if (obj.model !== undefined && obj.model !== null && obj.original_model === undefined) {
    obj.original_model = obj.model;
  }
  const parsed = TokenCountRequestBase.parse(obj);
  parsed.model_tier = extractTier(parsed.original_model || parsed.model) as ModelTier | null;
  return parsed;
}

export interface TokenCountResponse {
  input_tokens: number;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

export type AnthropicStopReason =
  | "end_turn"
  | "max_tokens"
  | "stop_sequence"
  | "tool_use"
  | "pause_turn"
  | "refusal";

export interface MessagesResponse {
  id: string;
  model: string;
  role: "assistant";
  content: Array<Record<string, unknown>>;
  type: "message";
  stop_reason: AnthropicStopReason | null;
  stop_sequence: string | null;
  usage: Usage;
}
