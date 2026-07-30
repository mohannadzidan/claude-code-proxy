import { DUMP_EVENTS_CLAUDE } from "../env.js";
import { logger } from "../logger.js";
import { PROXY_THINKING_SIGNATURE } from "../env.js";

/** Arrow direction is relative to the proxy: "CLAUDE ->" is the proxy sending to the client. */
export function sse(eventType: string, payload: Record<string, unknown>): string {
  if (DUMP_EVENTS_CLAUDE) {
    logger.info(`CLAUDE → ${eventType}: ${JSON.stringify(payload).slice(0, 600)}`);
  }
  return `event: ${eventType}\ndata: ${JSON.stringify(payload)}\n\n`;
}

export type OpenBlockType = "text" | "tool_use" | "thinking";

/**
 * Tracks the open Anthropic content block while translating a provider
 * stream.
 *
 * Anthropic requires that a text_delta only ever targets an open `text`
 * block and an input_json_delta only ever targets an open `tool_use` block,
 * and that block indices match their position in the final content array.
 */
export class BlockTracker {
  index = -1;
  openType: OpenBlockType | null = null;

  close(): string {
    if (this.openType !== null) {
      let out = "";
      if (this.openType === "thinking") {
        // Anthropic closes a thinking block with a signature_delta, and
        // clients that verify block shape drop thinking blocks that never
        // receive one. We cannot mint a real Anthropic signature for a
        // third-party backend, so emit a clearly-marked placeholder: it is
        // only ever consumed by this proxy, which strips thinking blocks
        // back out of conversation history before they reach any backend.
        out += sse("content_block_delta", {
          type: "content_block_delta",
          index: this.index,
          delta: { type: "signature_delta", signature: PROXY_THINKING_SIGNATURE },
        });
      }
      out += sse("content_block_stop", { type: "content_block_stop", index: this.index });
      this.openType = null;
      return out;
    }
    return "";
  }

  open(blockType: OpenBlockType, block: Record<string, unknown>): string {
    this.index += 1;
    this.openType = blockType;
    return sse("content_block_start", { type: "content_block_start", index: this.index, content_block: block });
  }
}
