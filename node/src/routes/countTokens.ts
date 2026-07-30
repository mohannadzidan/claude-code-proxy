import type { FastifyInstance } from "fastify";
import { parseTokenCountRequest, type MessagesRequest } from "../schema/types.js";
import { getModel } from "../router/config.js";
import { convertAnthropicToProviderRequest } from "../convert/toProvider.js";
import { approxTextTokens, approxTokenCount } from "../tokenCounter.js";
import { logger, logRequestBeautifully } from "../logger.js";
import { anthropicErrorBody } from "../errors.js";

export function registerCountTokensRoute(app: FastifyInstance): void {
  app.post("/v1/messages/count_tokens", async (request, reply) => {
    try {
      const tcr = parseTokenCountRequest(request.body);
      const originalModel = tcr.original_model || tcr.model;
      const displayModel = originalModel.includes("/") ? originalModel.split("/").pop()! : originalModel;

      const resolved = getModel(tcr.model_tier, tcr.original_model || tcr.model);
      const resolvedDisplay = resolved.displayName || resolved.id;

      // Convert the messages to the same OpenAI-shaped form the real request
      // path uses, so token counting reflects what actually gets sent
      // upstream (tool_calls / role:"tool" structure, not raw Anthropic blocks).
      const pseudoRequest = {
        model: tcr.model,
        max_tokens: 100,
        messages: tcr.messages,
        system: tcr.system,
        tools: tcr.tools,
        tool_choice: tcr.tool_choice,
        thinking: tcr.thinking,
        stream: false,
      } as unknown as MessagesRequest;

      const converted = convertAnthropicToProviderRequest(pseudoRequest, resolved);

      logRequestBeautifully(
        "POST",
        request.url,
        displayModel,
        resolvedDisplay,
        converted.messages.length,
        (converted.tools || []).length,
        null
      );

      let tokenCount = approxTokenCount(converted.messages);

      // Tool definitions are part of the billed prompt; Claude Code uses this
      // endpoint to decide when to compact, so undercounting the whole
      // toolset made it compact far too late and overflow the real context
      // window.
      if (converted.tools && converted.tools.length > 0) {
        tokenCount += approxTextTokens(JSON.stringify(converted.tools));
      }

      return { input_tokens: tokenCount };
    } catch (e) {
      logger.error(`Error counting tokens: ${(e as Error).message}\n${(e as Error).stack}`);
      reply.code(500);
      return anthropicErrorBody(500, `Error counting tokens: ${(e as Error).message}`);
    }
  });
}
