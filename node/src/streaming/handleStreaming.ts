import { randomUUID } from "node:crypto";
import type { MessagesRequest } from "../schema/types.js";
import type { ProviderStreamChunk } from "../providers/types.js";
import { BlockTracker, sse } from "./sse.js";
import { iterWithIdleTimeout } from "./idleIterator.js";
import { repairTruncatedJson } from "./repairJson.js";
import { extractUsage, mapFinishReason } from "../convert/fromProvider.js";
import { clientToolId } from "../convert/toolId.js";
import {
  DUMP_EVENTS_MODE,
  DUMP_EVENTS_UPSTREAM,
  ERROR_ON_EMPTY_RESPONSE,
  STREAM_KEEPALIVE_SECONDS,
  UPSTREAM_IDLE_TIMEOUT,
} from "../env.js";
import { logger } from "../logger.js";

function argsComplete(buf: string): boolean {
  if (!buf.trim()) return true;
  try {
    JSON.parse(buf);
    return true;
  } catch {
    return false;
  }
}

interface ToolState {
  id: string | null;
  name: string | null;
  args: string;
  order: number;
}

/** Translate a provider-style (OpenAI-shaped) stream into Anthropic SSE events. */
export async function* handleStreaming(
  responseGenerator: AsyncIterable<ProviderStreamChunk>,
  originalRequest: MessagesRequest
): AsyncGenerator<string> {
  const messageId = `msg_${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const responseModel = originalRequest.original_model || originalRequest.model;

  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadTokens = 0;
  let stopReason = "end_turn";
  let stopSequence: string | null = null;
  let accumulatedText = "";

  const tracker = new BlockTracker();
  // Per-upstream-tool-call state. Tool calls are buffered in full and emitted
  // as complete blocks once the stream ends, because Anthropic content
  // blocks cannot be reopened once closed.
  const toolStates = new Map<number, ToolState>();
  let emittedStop = false;

  function* emitText(txt: string): Generator<string> {
    if (tracker.openType !== "text") {
      const closed = tracker.close();
      if (closed) yield closed;
      yield tracker.open("text", { type: "text", text: "" });
    }
    yield sse("content_block_delta", {
      type: "content_block_delta",
      index: tracker.index,
      delta: { type: "text_delta", text: txt },
    });
  }

  try {
    // message_start must carry input_tokens; sending 0 here made Claude
    // Code's context/cost accounting read as zero for every streamed turn.
    yield sse("message_start", {
      type: "message_start",
      message: {
        id: messageId,
        type: "message",
        role: "assistant",
        model: responseModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: inputTokens,
          cache_creation_input_tokens: 0,
          cache_read_input_tokens: cacheReadTokens,
          output_tokens: 0,
        },
      },
    });
    yield sse("ping", { type: "ping" });

    let lastEmitAt = Date.now();
    const streamStartedAt = lastEmitAt;
    let upstreamChunks = 0;
    let firstChunkAt: number | null = null;
    let textDeltaCount = 0;

    for await (const chunk of iterWithIdleTimeout(responseGenerator, UPSTREAM_IDLE_TIMEOUT)) {
      try {
        upstreamChunks++;
        if (firstChunkAt === null) firstChunkAt = Date.now();

        if (DUMP_EVENTS_UPSTREAM) {
          logger.info(`UPSTREAM ← ${JSON.stringify(chunk)}`);
        }

        // While tool arguments accumulate the proxy emits no content at all.
        // On a large Edit/Write that is tens of seconds of dead air, and
        // clients time the connection out mid-generation. Ping to keep it
        // alive; pings are valid anywhere in an Anthropic stream.
        if (STREAM_KEEPALIVE_SECONDS && (Date.now() - lastEmitAt) / 1000 >= STREAM_KEEPALIVE_SECONDS) {
          lastEmitAt = Date.now();
          yield sse("ping", { type: "ping" });
        }

        // Usage can arrive on any chunk; with stream_options.include_usage it
        // lands on a final chunk that has an empty choices array.
        if (chunk.usage !== undefined && chunk.usage !== null) {
          const [p, c, cached] = extractUsage(chunk.usage);
          inputTokens = p || inputTokens;
          outputTokens = c || outputTokens;
          cacheReadTokens = cached || cacheReadTokens;
        }

        const choices = chunk.choices;
        if (!choices || choices.length === 0) continue;

        const choice = choices[0];
        const delta = choice.delta || {};
        const finishReason = choice.finish_reason;

        // ---- reasoning trace ----
        const reasoning = delta.reasoning_content || delta.reasoning;
        if (reasoning) {
          if (tracker.openType !== "thinking") {
            const closed = tracker.close();
            if (closed) yield closed;
            yield tracker.open("thinking", { type: "thinking", thinking: "" });
          }
          yield sse("content_block_delta", {
            type: "content_block_delta",
            index: tracker.index,
            delta: { type: "thinking_delta", thinking: reasoning },
          });
        }

        // ---- text ----
        const deltaContent = delta.content;
        if (deltaContent) {
          // Tool blocks are no longer opened mid-stream, so text can always
          // stream immediately without risk of landing on a tool_use block.
          for (const s of emitText(deltaContent)) yield s;
          accumulatedText += deltaContent;
          textDeltaCount++;
          lastEmitAt = Date.now();
        }

        // ---- tool calls ----
        // Accumulate only. Anthropic content blocks cannot be reopened once
        // closed, so opening a tool block while its arguments are still
        // arriving would drop any fragment that shows up after another block
        // has opened. Buffer everything and emit complete, well-formed
        // tool_use blocks at the end of the stream.
        let deltaToolCalls = delta.tool_calls;

        // Some gateways still emit the pre-2023 `function_call` shape
        // instead of `tool_calls`. Ignoring it means the tool call is
        // invisible to us, the turn looks like an empty reply, and the agent
        // loop halts with no error anywhere.
        if (!deltaToolCalls && delta.function_call) {
          const legacy = delta.function_call;
          deltaToolCalls = [{ index: 0, id: legacy.id, function: { name: legacy.name, arguments: legacy.arguments } }];
          logger.debug("Translated legacy function_call delta");
        }

        if (deltaToolCalls && deltaToolCalls.length > 0) {
          for (const toolCall of deltaToolCalls) {
            const upstreamIndex = toolCall.index ?? 0;
            const func = toolCall.function;
            const name = func?.name;
            let args: unknown = func?.arguments;
            const callId = toolCall.id;

            let st = toolStates.get(upstreamIndex);
            if (!st) {
              st = { id: null, name: null, args: "", order: toolStates.size };
              toolStates.set(upstreamIndex, st);
            }
            if (callId && !st.id) st.id = callId;
            // First non-empty name wins; some backends send the name in a
            // later chunk than the one that opens the call.
            if (name && !st.name) st.name = name;
            if (args) {
              if (typeof args !== "string") args = JSON.stringify(args);
              st.args += args as string;
            }
          }
        }

        // ---- completion ----
        if (finishReason && !emittedStop) {
          emittedStop = true;
          stopReason = mapFinishReason(finishReason, toolStates.size > 0);

          if (originalRequest.stop_sequences && accumulatedText) {
            for (const seq of originalRequest.stop_sequences) {
              if (seq && accumulatedText.endsWith(seq)) {
                stopReason = "stop_sequence";
                stopSequence = seq;
                break;
              }
            }
          }

          const closed = tracker.close();
          if (closed) yield closed;
          // Do not return here: the usage-bearing chunk from
          // stream_options.include_usage arrives *after* finish_reason.
        }
      } catch (chunkError) {
        logger.error(`Error processing chunk: ${(chunkError as Error).message}`);
        continue;
      }
    }

    if (DUMP_EVENTS_MODE !== "none") {
      const now = Date.now();
      const ttfc = firstChunkAt !== null ? (firstChunkAt - streamStartedAt) / 1000 : -1;
      const reasoningOnly = upstreamChunks - textDeltaCount;
      logger.info(
        `STREAM-PROFILE: upstream_chunks=${upstreamChunks} text_deltas=${textDeltaCount} non_text_chunks=${reasoningOnly} tool_calls=${toolStates.size} time_to_first_chunk=${ttfc.toFixed(2)}s total=${((now - streamStartedAt) / 1000).toFixed(2)}s`
      );
    }

    // An assistant turn with no text and no tool calls ends the agent loop
    // with nothing shown. That is almost always a backend problem, but as a
    // well-formed empty message it is indistinguishable from the model
    // simply choosing to stop. Surface it.
    const producedNothing = !accumulatedText.trim() && toolStates.size === 0;
    if (producedNothing) {
      logger.warn(
        `EMPTY COMPLETION: backend returned no text and no tool calls (upstream_chunks=${upstreamChunks}, finish_reason=${stopReason}, input_tokens=${inputTokens || "unknown"}). Run with DUMP_EVENTS=upstream to see the raw stream.`
      );
      if (ERROR_ON_EMPTY_RESPONSE) {
        yield sse("error", {
          type: "error",
          error: {
            type: "api_error",
            message: `Backend returned an empty completion (no text, no tool calls) after ${upstreamChunks} chunks. This usually means the prompt exceeded the backend's context window. Set ERROR_ON_EMPTY_RESPONSE=false to suppress.`,
          },
        });
        yield sse("message_stop", { type: "message_stop" });
        yield "data: [DONE]\n\n";
        return;
      }
    }

    // Close whatever text block is still open.
    const closedFinal = tracker.close();
    if (closedFinal) yield closedFinal;

    // Emit every buffered tool call as a complete, well-formed block, in the
    // order the backend introduced them.
    const orderedKeys = [...toolStates.keys()].sort(
      (a, b) => (toolStates.get(a)?.order ?? 0) - (toolStates.get(b)?.order ?? 0)
    );
    for (const key of orderedKeys) {
      const st = toolStates.get(key)!;
      if (!(st.id || st.name || st.args)) continue;
      let args = st.args;
      if (!argsComplete(args)) {
        const repaired = repairTruncatedJson(args);
        logger.warn(
          `Tool ${JSON.stringify(st.name)} arguments were not valid JSON (${args.length} chars); ${repaired !== null ? "repaired" : "sending as-is"}`
        );
        if (repaired !== null) args = repaired;
      }
      yield tracker.open("tool_use", {
        type: "tool_use",
        id: clientToolId(st.id, st.name),
        name: st.name || "unknown",
        input: {},
      });
      if (args) {
        yield sse("content_block_delta", {
          type: "content_block_delta",
          index: tracker.index,
          delta: { type: "input_json_delta", partial_json: args },
        });
      }
      const closed = tracker.close();
      if (closed) yield closed;
    }

    // Anthropic requires at least one content block in the message.
    if (tracker.index === -1) {
      yield tracker.open("text", { type: "text", text: "" });
      const closed = tracker.close();
      if (closed) yield closed;
    }

    // Some backends end the stream without ever sending a finish_reason. If
    // we emitted tool calls, the turn is still a tool turn.
    if (toolStates.size > 0 && stopReason === "end_turn") stopReason = "tool_use";

    yield sse("message_delta", {
      type: "message_delta",
      delta: { stop_reason: stopReason, stop_sequence: stopSequence },
      usage: {
        input_tokens: inputTokens,
        cache_read_input_tokens: cacheReadTokens,
        output_tokens: outputTokens,
      },
    });
    yield sse("message_stop", { type: "message_stop" });
    yield "data: [DONE]\n\n";
  } catch (e) {
    logger.error(`Error in streaming: ${(e as Error).message}\n${(e as Error).stack}`);

    try {
      const closed = tracker.close();
      if (closed) yield closed;
    } catch {
      // ignore
    }

    yield sse("error", { type: "error", error: { type: "api_error", message: (e as Error).message } });
    yield sse("message_stop", { type: "message_stop" });
    yield "data: [DONE]\n\n";
  }
}
