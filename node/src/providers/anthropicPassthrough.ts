import type { IncomingHttpHeaders } from "node:http";
import type { ResolvedModel } from "../router/config.js";
import { applyContentReplacements, applyContentReplacementsToSystem } from "../contentReplacements.js";
import { CUSTOM_HEADERS, DUMP_EVENTS_CLAUDE, DUMP_EVENTS_UPSTREAM, RESPONSE_PROTECTED_HEADERS } from "../env.js";
import { logger, logRequestBeautifully } from "../logger.js";

// Real default for endpointType: anthropic when router.yaml leaves
// providerBaseUrl unset.
const ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com";

// Overall request timeout (matches server.py's httpx.Timeout(600.0, connect=30.0);
// fetch has no separate connect-phase timeout, so this covers the whole call).
const PASSTHROUGH_TIMEOUT_MS = 600_000;

// Headers that must not be forwarded verbatim: framing/hop-by-hop headers
// tied to the client<->proxy connection, and auth, which is replaced with
// the key configured for this upstream in router.yaml.
const STRIP_HEADERS = new Set(["host", "content-length", "connection", "transfer-encoding", "x-api-key", "authorization"]);

/**
 * Headers for a transparent /v1/messages call to a real Anthropic-compatible
 * backend. Forwards every header the client sent as-is (anthropic-version,
 * anthropic-beta, user-agent, x-stainless-*, ...) except the handful that
 * must change because the destination differs, and auth.
 */
export function anthropicPassthroughHeaders(clientHeaders: IncomingHttpHeaders, resolved: ResolvedModel): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const [k, v] of Object.entries(clientHeaders)) {
    if (v === undefined || STRIP_HEADERS.has(k.toLowerCase())) continue;
    headers[k] = Array.isArray(v) ? v.join(", ") : v;
  }
  headers["x-api-key"] = resolved.providerApiKey;
  if (!headers["anthropic-version"]) headers["anthropic-version"] = "2023-06-01";
  if (resolved.requestHeaders) Object.assign(headers, resolved.requestHeaders);
  Object.assign(headers, CUSTOM_HEADERS);
  return headers;
}

/** Upstream response headers to relay back to the client as-is (rate-limit counters, request-id, ...). */
function passthroughResponseHeaders(resp: Response): Record<string, string> {
  const out: Record<string, string> = {};
  resp.headers.forEach((value, key) => {
    if (!RESPONSE_PROTECTED_HEADERS.has(key.toLowerCase())) out[key] = value;
  });
  return out;
}

/**
 * Rewrite the `model` field in a message_start SSE event to what the client
 * asked for. router.yaml can map a tier to a different specific upstream
 * model id than the client requested, and Claude Code checks this field
 * against what it sent -- everything else in the event passes through
 * untouched.
 */
function patchMessageStartModel(eventText: string, originalModel: string): string {
  for (const line of eventText.split("\n")) {
    if (line.startsWith("data: ")) {
      let payload: Record<string, unknown>;
      try {
        payload = JSON.parse(line.slice("data: ".length));
      } catch {
        return eventText;
      }
      if (payload.message && typeof payload.message === "object") {
        (payload.message as Record<string, unknown>).model = originalModel;
        const patchedLine = `data: ${JSON.stringify(payload)}`;
        return eventText.replace(line, patchedLine);
      }
      return eventText;
    }
  }
  return eventText;
}

/**
 * Relay a real Anthropic backend's SSE stream to the client byte-for-byte,
 * splitting only on event boundaries (blank lines) so each event can be
 * logged and the message_start model field patched.
 */
async function* streamAnthropicPassthrough(body: ReadableStream<Uint8Array>, originalModel: string): AsyncGenerator<Uint8Array> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  const encoder = new TextEncoder();
  let buf = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        let event = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        event = patchMessageStartModel(event, originalModel);
        if (DUMP_EVENTS_UPSTREAM) logger.info(`UPSTREAM ← ${event}`);
        yield encoder.encode(`${event}\n\n`);
      }
    }
    if (buf) {
      if (DUMP_EVENTS_UPSTREAM) logger.info(`UPSTREAM ← ${buf}`);
      yield encoder.encode(buf);
    }
  } finally {
    reader.releaseLock();
  }
}

export interface PassthroughJsonResult {
  kind: "json";
  status: number;
  body: unknown;
  headers: Record<string, string>;
}

export interface PassthroughStreamResult {
  kind: "stream";
  status: number;
  headers: Record<string, string>;
  body: AsyncIterable<Uint8Array>;
}

export type PassthroughResult = PassthroughJsonResult | PassthroughStreamResult;

/**
 * Forward an /v1/messages request straight to a real Anthropic-compatible
 * backend (router.yaml endpointType: anthropic) with no OpenAI-shape round
 * trip. Those backends already speak the exact wire format Claude Code
 * sends, so this patches only what the router itself is responsible for --
 * `model` and, if router.yaml sets reasoning_effort_map, `output_config.effort`
 * -- and otherwise sends the client's JSON straight through.
 */
export async function anthropicPassthrough(params: {
  bodyJson: Record<string, unknown>;
  resolved: ResolvedModel;
  originalModel: string;
  clientHeaders: IncomingHttpHeaders;
  requestPath: string;
  displayModel: string;
  resolvedDisplay: string;
}): Promise<PassthroughResult> {
  const { bodyJson, resolved, originalModel, clientHeaders, requestPath, displayModel, resolvedDisplay } = params;

  const outgoing: Record<string, unknown> = { ...bodyJson };
  outgoing.model = resolved.providerModelName;
  outgoing.messages = applyContentReplacements(
    (outgoing.messages as Record<string, unknown>[] | undefined) || []
  );
  if ("system" in outgoing) {
    outgoing.system = applyContentReplacementsToSystem(outgoing.system);
  }

  if (resolved.reasoningEffortMap) {
    const outputConfig = outgoing.output_config;
    if (outputConfig && typeof outputConfig === "object" && typeof (outputConfig as Record<string, unknown>).effort === "string") {
      const oc = outputConfig as Record<string, unknown>;
      const effort = oc.effort as string;
      oc.effort = resolved.reasoningEffortMap[effort] ?? effort;
    }
  }

  const url = `${(resolved.providerBaseUrl || ANTHROPIC_DEFAULT_BASE_URL).replace(/\/+$/, "")}/v1/messages`;
  const headers = anthropicPassthroughHeaders(clientHeaders, resolved);

  if (DUMP_EVENTS_UPSTREAM) {
    const dump: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(outgoing)) {
      if (k === "messages" || k === "tools" || k === "system") continue;
      dump[k] = v;
    }
    dump.message_roles = ((outgoing.messages as Record<string, unknown>[]) || []).map((m) => m.role);
    logger.info(`UPSTREAM → ${JSON.stringify(dump)}`);
  }

  const numTools = ((outgoing.tools as unknown[]) || []).length;
  const numMessages = ((outgoing.messages as unknown[]) || []).length;
  logRequestBeautifully("POST", requestPath, displayModel, resolvedDisplay, numMessages, numTools, null);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PASSTHROUGH_TIMEOUT_MS);

  try {
    const resp = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(outgoing),
      signal: controller.signal,
    });

    if (outgoing.stream) {
      if (resp.status >= 300) {
        const errorBody = await resp.text();
        logger.error(`UPSTREAM error ${resp.status}: ${errorBody.slice(0, 2000)}`);
        let content: unknown = errorBody;
        try {
          content = JSON.parse(errorBody);
        } catch {
          // leave as text
        }
        return { kind: "json", status: resp.status, body: content, headers: passthroughResponseHeaders(resp) };
      }

      if (!resp.body) {
        return { kind: "stream", status: resp.status, headers: passthroughResponseHeaders(resp), body: (async function* () {})() };
      }

      return {
        kind: "stream",
        status: resp.status,
        headers: {
          ...passthroughResponseHeaders(resp),
          "cache-control": "no-cache",
          connection: "keep-alive",
          "x-accel-buffering": "no",
        },
        body: streamAnthropicPassthrough(resp.body, originalModel),
      };
    }

    const text = await resp.text();
    if (DUMP_EVENTS_UPSTREAM) logger.info(`UPSTREAM ← ${text.slice(0, 6000)}`);

    let content: unknown;
    try {
      content = JSON.parse(text);
    } catch {
      content = text;
    }

    if (resp.status < 300 && content && typeof content === "object") {
      (content as Record<string, unknown>).model = originalModel;
      if (DUMP_EVENTS_CLAUDE) logger.info(`CLAUDE → ${JSON.stringify(content)}`);
    }

    return { kind: "json", status: resp.status, body: content, headers: passthroughResponseHeaders(resp) };
  } finally {
    clearTimeout(timer);
  }
}
