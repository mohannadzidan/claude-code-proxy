import { fileURLToPath } from "node:url";
import path from "node:path";
import dotenv from "dotenv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// node/src (or node/dist) -> node -> project root. Shared with server.py so
// both implementations can read the same router.yaml/.env during migration.
export const NODE_DIR = path.resolve(__dirname, "..");
export const PROJECT_ROOT = path.resolve(NODE_DIR, "..");

// Load the shared root .env first, then let a node/.env add or override
// node-only settings (e.g. PORT) without touching the file server.py reads.
dotenv.config({ path: path.join(PROJECT_ROOT, ".env") });
dotenv.config({ path: path.join(NODE_DIR, ".env"), override: true });

export const ROUTER_CONFIG_PATH =
  process.env.ROUTER_CONFIG_PATH || path.join(PROJECT_ROOT, "router.yaml");

function envBool(name: string, def: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined) return def;
  return raw.toLowerCase() === "true";
}

function envFloat(name: string, def: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return def;
  const n = Number(raw);
  return Number.isFinite(n) ? n : def;
}

/**
 * Extract custom headers from CUSTOM_HEADER_<NAME> env vars.
 *
 * CUSTOM_HEADER_ORIGINATOR -> originator, CUSTOM_HEADER_CONTENT_TYPE -> content-type.
 */
export function getCustomHeadersFromEnv(): Record<string, string> {
  const headers: Record<string, string> = {};
  const prefix = "CUSTOM_HEADER_";
  for (const [key, value] of Object.entries(process.env)) {
    if (key.startsWith(prefix) && key.length > prefix.length && value !== undefined) {
      const headerName = key.slice(prefix.length).replace(/_/g, "-").toLowerCase();
      headers[headerName] = value;
    }
  }
  return headers;
}

export const CUSTOM_HEADERS = getCustomHeadersFromEnv();

// Headers that must never be overwritten on an outgoing response. Replacing
// content-type on a streamed reply breaks SSE parsing outright; the rest are
// hop-by-hop/framing headers owned by the server.
export const RESPONSE_PROTECTED_HEADERS = new Set([
  "content-type",
  "content-length",
  "transfer-encoding",
  "content-encoding",
  "connection",
]);

export const LOG_LEVEL = (process.env.LOG_LEVEL || "info").toLowerCase();
export const LITELLM_LOG_LEVEL = (process.env.LITELLM_LOG_LEVEL || "warning").toLowerCase();

// Output-token ceiling. 0/unset disables the cap.
export const MAX_TOKENS_LIMIT = (() => {
  const n = parseInt(process.env.MAX_TOKENS_LIMIT || "0", 10);
  return Number.isFinite(n) && n > 0 ? n : null;
})();

// The claude->Open text rewriting is destructive for a coding agent (it rewrites
// source code, file paths and tool output). Off unless explicitly enabled.
export const ENABLE_CONTENT_REPLACEMENTS = envBool("ENABLE_CONTENT_REPLACEMENTS", false);

// Some OpenAI-compatible gateways reject the stream_options field outright.
export const DISABLE_STREAM_OPTIONS = envBool("DISABLE_STREAM_OPTIONS", false);

const VALID_DUMP_EVENTS_MODES = new Set(["none", "claude", "upstream", "all"]);

function parseDumpEventsMode(): string {
  const raw = process.env.DUMP_EVENTS;
  if (!raw) return "none";
  const mode = raw.trim().toLowerCase();
  if (!VALID_DUMP_EVENTS_MODES.has(mode)) {
    // eslint-disable-next-line no-console
    console.warn(
      `Invalid DUMP_EVENTS=${JSON.stringify(raw)} (expected one of none/claude/upstream/all); falling back to 'upstream'`
    );
    return "upstream";
  }
  return mode;
}

export const DUMP_EVENTS_MODE = parseDumpEventsMode();
export const DUMP_EVENTS_CLAUDE = DUMP_EVENTS_MODE === "claude" || DUMP_EVENTS_MODE === "all";
export const DUMP_EVENTS_UPSTREAM = DUMP_EVENTS_MODE === "upstream" || DUMP_EVENTS_MODE === "all";

// Seconds of silence before the proxy emits a keep-alive ping mid-stream. 0 disables.
export const STREAM_KEEPALIVE_SECONDS = envFloat("STREAM_KEEPALIVE_SECONDS", 3);

// Abort if the upstream sends no data for this long. 0 disables.
export const UPSTREAM_IDLE_TIMEOUT = envFloat("UPSTREAM_IDLE_TIMEOUT", 90);

// Report an empty assistant turn as an error rather than letting it look like a
// normal finish.
export const ERROR_ON_EMPTY_RESPONSE = envBool("ERROR_ON_EMPTY_RESPONSE", true);

export const PRESERVE_UPSTREAM_TOOL_IDS = envBool("PRESERVE_UPSTREAM_TOOL_IDS", false);

export const PORT = parseInt(process.env.PORT || "8082", 10);

// Placeholder signature attached to synthesized thinking blocks. Real Anthropic
// signatures are cryptographic and cannot be produced for a third-party backend;
// this only has to be non-empty and stable, since the proxy strips thinking
// blocks out of conversation history before forwarding them upstream.
export const PROXY_THINKING_SIGNATURE = Buffer.from("litellm-proxy-unsigned-thinking").toString(
  "base64"
);
