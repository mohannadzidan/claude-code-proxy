import type { ProviderChatRequest } from "../convert/toProvider.js";
import type { ProviderStreamChunk } from "./types.js";
import type { ProviderResponse } from "../convert/fromProvider.js";
import { completeChatOpenAi, streamChatOpenAi } from "./openai.js";

/**
 * Dispatch a provider-agnostic chat request to the adapter matching
 * `req.customLlmProvider`. Only reached for router.yaml endpointType
 * openai/hosted_vllm/gemini -- endpointType: anthropic is handled entirely
 * by providers/anthropicPassthrough.ts before a ProviderChatRequest is ever
 * built (see convert/toProvider.ts's isAnthropicModel branches, which only
 * affect the shape of a request that a passthrough call never sends).
 */
export async function completeChat(req: ProviderChatRequest): Promise<ProviderResponse> {
  if (req.customLlmProvider === "openai" || req.customLlmProvider === "hosted_vllm") {
    return completeChatOpenAi(req);
  }
  if (req.customLlmProvider === "gemini") {
    throw new Error("endpointType: gemini is not yet implemented in the node port");
  }
  throw new Error(`Unsupported provider: ${req.customLlmProvider}`);
}

export async function* streamChat(req: ProviderChatRequest): AsyncGenerator<ProviderStreamChunk> {
  if (req.customLlmProvider === "openai" || req.customLlmProvider === "hosted_vllm") {
    yield* streamChatOpenAi(req);
    return;
  }
  if (req.customLlmProvider === "gemini") {
    throw new Error("endpointType: gemini is not yet implemented in the node port");
  }
  throw new Error(`Unsupported provider: ${req.customLlmProvider}`);
}
