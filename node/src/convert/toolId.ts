import { randomUUID } from "node:crypto";
import { PRESERVE_UPSTREAM_TOOL_IDS } from "../env.js";
import { logger } from "../logger.js";

// Characters Anthropic tool_use ids never contain. An id outside this set is
// certainly synthesized by the backend rather than a random handle.
const PLAIN_TOOL_ID = /^[A-Za-z0-9_-]+$/;

/**
 * Return a tool_use id that is unique across the whole conversation.
 *
 * Some backends derive the id from the tool name and its position, e.g.
 * Kimi emits "Edit:0". Those repeat on every turn that calls the same tool
 * in the same slot. Anthropic requires tool_use ids to be unique within a
 * conversation and clients key tool results by id, so a repeat collides with
 * an earlier call and gets reported as an interrupted tool use.
 *
 * Only ids that look name-derived (or contain illegal characters) are
 * rewritten; genuinely random handles like `call_x7fa...` are passed through
 * untouched.
 */
export function clientToolId(upstreamId: unknown, toolName?: unknown): string {
  if (typeof upstreamId === "string" && upstreamId) {
    if (PRESERVE_UPSTREAM_TOOL_IDS) return upstreamId;
    let derived = false;
    if (typeof toolName === "string" && toolName.length >= 3) {
      // e.g. "Edit:0", "Edit_1", "Read.2", "Bash3" -- tool name plus a
      // positional index. Real tool names are >= 3 chars, so a short name
      // cannot accidentally match a random id.
      const escaped = toolName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const re = new RegExp(`^${escaped}[-_:.]?\\d*$`, "i");
      derived = re.test(upstreamId);
    }
    if (PLAIN_TOOL_ID.test(upstreamId) && !derived) return upstreamId;
    logger.debug(
      `Rewriting tool id ${JSON.stringify(upstreamId)} for tool ${JSON.stringify(toolName)}: looks derived from the tool name and will repeat across turns`
    );
  }
  return `toolu_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
}
