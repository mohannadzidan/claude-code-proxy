/**
 * Provider-agnostic streaming chunk shape (OpenAI chat.completion.chunk
 * shape). Every provider adapter (openai.ts, gemini.ts) normalizes into
 * this so handleStreaming/convertProviderToAnthropic never need to know
 * which backend produced the data -- the same role litellm plays in
 * server.py.
 */
export interface ProviderStreamDelta {
  content?: string | null;
  reasoning_content?: string | null;
  reasoning?: string | null;
  tool_calls?: Array<{
    index?: number;
    id?: string | null;
    function?: { name?: string | null; arguments?: string | null };
  }> | null;
  function_call?: { id?: string | null; name?: string | null; arguments?: string | null } | null;
}

export interface ProviderStreamChunk {
  choices?: Array<{ delta?: ProviderStreamDelta; finish_reason?: string | null }>;
  usage?: unknown;
}
