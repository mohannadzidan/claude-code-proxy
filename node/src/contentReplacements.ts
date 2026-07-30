import { ENABLE_CONTENT_REPLACEMENTS } from "./env.js";

// Define your regex patterns and their static replacement strings here.
export const CONTENT_REPLACEMENTS: [RegExp, string][] = [
  [/^\s*x-anthropic-billing-header.+\n*/, ""],
  [/\bclaude\b|\bAnthropic\b/gi, "Open"],
  // Add more patterns as needed
];

function applyPatterns(text: string): string {
  let out = text;
  for (const [pattern, replacement] of CONTENT_REPLACEMENTS) {
    out = out.replace(pattern, replacement);
  }
  return out;
}

interface LooseMessage {
  role?: string;
  content?: unknown;
  [key: string]: unknown;
}

/**
 * Applies regex replacements to message content before sending to the LLM.
 *
 * Disabled unless ENABLE_CONTENT_REPLACEMENTS=true. For a coding agent this
 * rewriting is actively destructive: the default rules rewrite every
 * occurrence of "claude"/"Anthropic" inside source files, paths and tool
 * output, so the model proposes edits against text that does not match the
 * user's disk.
 */
export function applyContentReplacements(messages: LooseMessage[]): LooseMessage[] {
  if (!ENABLE_CONTENT_REPLACEMENTS) return messages;

  for (const msg of messages) {
    // Never rewrite tool results: they are verbatim command/file output.
    if (msg.role === "tool") continue;
    const content = msg.content;

    if (typeof content === "string") {
      msg.content = applyPatterns(content);
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (block && typeof block === "object" && (block as Record<string, unknown>).type === "text") {
          const b = block as Record<string, unknown>;
          b.text = applyPatterns((b.text as string) || "");
        }
      }
    }
  }
  return messages;
}

/**
 * Same rewrite as applyContentReplacements, for Anthropic's separate
 * top-level `system` field (which the OpenAI-shaped pipeline folds into the
 * messages list, but the passthrough path forwards as-is).
 */
export function applyContentReplacementsToSystem(system: unknown): unknown {
  if (!ENABLE_CONTENT_REPLACEMENTS) return system;
  if (typeof system === "string") return applyPatterns(system);
  if (Array.isArray(system)) {
    for (const block of system) {
      if (block && typeof block === "object" && (block as Record<string, unknown>).type === "text") {
        const b = block as Record<string, unknown>;
        b.text = applyPatterns((b.text as string) || "");
      }
    }
  }
  return system;
}
