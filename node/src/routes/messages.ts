import { Readable } from "node:stream";
import type { FastifyInstance, FastifyReply } from "fastify";
import { parseMessagesRequest, type MessagesRequest } from "../schema/types.js";
import { getModel, type ModelTier } from "../router/config.js";
import { convertAnthropicToProviderRequest, validateOpenaiMessages } from "../convert/toProvider.js";
import { convertProviderToAnthropic } from "../convert/fromProvider.js";
import { handleStreaming } from "../streaming/handleStreaming.js";
import { completeChat, streamChat } from "../providers/dispatch.js";
import { anthropicPassthrough, type PassthroughResult } from "../providers/anthropicPassthrough.js";
import { ProviderHttpError } from "../providers/httpError.js";
import { applyContentReplacements } from "../contentReplacements.js";
import { DUMP_EVENTS_CLAUDE, DUMP_EVENTS_UPSTREAM } from "../env.js";
import { logger, logRequestBeautifully } from "../logger.js";
import { anthropicErrorBody } from "../errors.js";

function displayModelName(model: string): string {
  return model.includes("/") ? model.split("/").pop()! : model;
}

// Fastify races an async handler's own resolved return value against
// anything reply.send() is still asynchronously writing (e.g. piping a
// stream): if the handler doesn't `return` the reply.send() call itself,
// Fastify can treat the handler's implicit `undefined` return as the actual
// payload and clobber the stream with an empty 0-length body. Every
// send-a-stream path below must be `return`ed all the way up to the route
// handler for this reason.
function sendPassthroughResult(reply: FastifyReply, result: PassthroughResult): FastifyReply {
  reply.code(result.status);
  for (const [k, v] of Object.entries(result.headers)) reply.header(k, v);
  if (result.kind === "json") {
    return reply.send(result.body);
  }
  reply.header("content-type", "text/event-stream");
  return reply.send(Readable.from(result.body));
}

export function registerMessagesRoute(app: FastifyInstance): void {
  app.post("/v1/messages", async (request, reply) => {
    let parsed: MessagesRequest | undefined;
    try {
      const bodyJson = request.body as Record<string, unknown>;

      if (DUMP_EVENTS_CLAUDE) {
        const dump: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(bodyJson)) {
          if (k === "messages" || k === "tools" || k === "system") continue;
          dump[k] = v;
        }
        dump.message_roles = ((bodyJson.messages as Record<string, unknown>[]) || []).map((m) => m.role);
        logger.info(`CLAUDE ← ${JSON.stringify(dump)}`);
      }

      parsed = parseMessagesRequest(bodyJson);
      const originalModel = (bodyJson.model as string) || "unknown";
      const displayModel = displayModelName(originalModel);

      const resolved = getModel(parsed.model_tier, parsed.original_model || parsed.model);
      const resolvedDisplay = resolved.displayName || resolved.id;

      logger.debug(
        `MODEL ROUTING: tier=${parsed.model_tier} -> id=${resolved.id} (provider=${resolved.litellmProvider}, upstream=${resolved.providerModelName})`
      );

      // endpointType: anthropic backends speak Claude Code's own wire format,
      // so skip the OpenAI-shaped conversion pipeline entirely and forward
      // the request straight through.
      if (resolved.litellmProvider === "anthropic") {
        const result = await anthropicPassthrough({
          bodyJson,
          resolved,
          originalModel: parsed.original_model || parsed.model,
          clientHeaders: request.headers,
          requestPath: request.url,
          displayModel,
          resolvedDisplay,
        });
        return sendPassthroughResult(reply, result);
      }

      const providerRequest = convertAnthropicToProviderRequest(parsed, resolved);
      validateOpenaiMessages(providerRequest);

      if (DUMP_EVENTS_UPSTREAM) {
        const dump: Record<string, unknown> = {};
        for (const [k, v] of Object.entries(providerRequest)) {
          if (k === "apiKey" || k === "messages" || k === "tools" || k === "extraHeaders") continue;
          dump[k] = v;
        }
        dump.message_roles = providerRequest.messages.map((m) => m.role);
        logger.info(`UPSTREAM → ${JSON.stringify(dump)}`);
      }

      providerRequest.messages = applyContentReplacements(providerRequest.messages);

      const numTools = (providerRequest.tools || []).length;

      if (parsed.stream) {
        logRequestBeautifully("POST", request.url, displayModel, resolvedDisplay, providerRequest.messages.length, numTools, null);

        const generator = streamChat(providerRequest);
        return reply
          .header("content-type", "text/event-stream")
          .header("cache-control", "no-cache")
          .header("connection", "keep-alive")
          .header("x-accel-buffering", "no")
          .send(Readable.from(handleStreaming(generator, parsed)));
      }

      logRequestBeautifully("POST", request.url, displayModel, resolvedDisplay, providerRequest.messages.length, numTools, null);

      const providerResponse = await completeChat(providerRequest);
      if (DUMP_EVENTS_UPSTREAM) {
        logger.info(`UPSTREAM ← ${JSON.stringify(providerResponse)}`);
      }

      const anthropicResponse = convertProviderToAnthropic(providerResponse, parsed);
      if (DUMP_EVENTS_CLAUDE) {
        logger.info(`CLAUDE → ${JSON.stringify(anthropicResponse)}`);
      }

      logRequestBeautifully("POST", request.url, displayModel, resolvedDisplay, providerRequest.messages.length, numTools, 200);
      return reply.send(anthropicResponse);
    } catch (e) {
      const err = e as Error;
      logger.error(`Error processing request: ${err.message}\n${err.stack}`);

      const statusCode = e instanceof ProviderHttpError ? e.statusCode : 500;
      const message = err.message || String(e);

      let resolvedDisplayForError = parsed?.model || "unknown";
      try {
        if (parsed) {
          const errorResolved = getModel(parsed.model_tier as ModelTier | null, parsed.original_model || parsed.model);
          resolvedDisplayForError = errorResolved.displayName || errorResolved.id;
        }
      } catch {
        // keep fallback display value
      }

      logRequestBeautifully(
        "POST",
        request.url,
        parsed?.original_model || parsed?.model || "unknown",
        resolvedDisplayForError,
        parsed?.messages.length || 0,
        parsed?.tools?.length || 0,
        statusCode
      );

      return reply.code(statusCode).send(anthropicErrorBody(statusCode, message));
    }
  });
}
