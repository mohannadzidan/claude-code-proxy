export type ModelTier = "haiku" | "sonnet" | "opus";

const PROVIDER_PREFIXES = ["anthropic/", "openai/", "gemini/", "vertex_ai/", "azure/"];

export function stripProviderPrefix(model: string): string {
  for (const prefix of PROVIDER_PREFIXES) {
    if (model.startsWith(prefix)) return model.slice(prefix.length);
  }
  return model;
}

// Reasoning models reject temperature/top_p/stop and need max_completion_tokens.
const REASONING_MODEL_PATTERN = /^(o[1-4]|gpt-5|gpt-4\.5-preview|deepseek-r|qwq|grok-.*-reasoning)/i;

/** True for models that use the restricted reasoning-model parameter set. */
export function isReasoningModel(model: string): boolean {
  return REASONING_MODEL_PATTERN.test(stripProviderPrefix(model));
}

export function supportsTemperature(model: string): boolean {
  return !isReasoningModel(model);
}

/**
 * Extract the Claude tier (haiku/sonnet/opus) from an incoming model id.
 * router.yaml resolves the actual upstream model; this only determines which
 * tier's preferredModel* the request belongs to.
 */
export function extractTier(v: string | null | undefined): ModelTier | null {
  const lowered = stripProviderPrefix(v || "").toLowerCase();
  if (lowered.includes("haiku")) return "haiku";
  if (lowered.includes("sonnet")) return "sonnet";
  if (lowered.includes("opus")) return "opus";
  return null;
}

/** Pick the role name the target model accepts for system instructions. */
export function systemRoleFor(model: string): "system" | "developer" | null {
  const clean = stripProviderPrefix(model).toLowerCase();
  // o1-mini / o1-preview accept no system-ish role at all.
  if (clean.startsWith("o1-mini") || clean.startsWith("o1-preview")) return null;
  if (isReasoningModel(model)) return "developer";
  return "system";
}
