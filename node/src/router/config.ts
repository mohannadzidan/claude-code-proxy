import fs from "node:fs";
import yaml from "js-yaml";
import { ROUTER_CONFIG_PATH } from "../env.js";

export class RouterConfigError extends Error {}

// "hosted_vllm" mirrors litellm's provider type for self-hosted/third-party
// OpenAI-compatible inference servers. It is not meant to be written in
// router.yaml -- see effectiveCustomLlmProvider below, which upgrades
// "openai" entries to it automatically based on providerBaseUrl. Kept as an
// explicit-override escape hatch.
export const VALID_ENDPOINT_TYPES = new Set(["openai", "hosted_vllm", "anthropic", "gemini"]);

const REQUIRED_MODEL_FIELDS = ["id", "providerModelName", "endpointType", "providerApiKey"] as const;

// Real OpenAI API hosts; anything else behind endpointType: openai is a
// third-party OpenAI-compatible gateway.
const REAL_OPENAI_HOSTS = new Set(["api.openai.com"]);

export interface ModelEntry {
  id: string;
  providerModelName: string;
  endpointType: string;
  providerApiKey: string;
  providerBaseUrl?: string | null;
  displayName?: string | null;
  disableReasoning?: boolean | null;
  reasoningEffortMap?: Record<string, string> | null;
  requestHeaders?: Record<string, string> | null;
}

export type ResolvedModel = ModelEntry & {
  litellmModel: string;
  litellmProvider: string;
};

export interface RouterConfig {
  preferredModel: string;
  models: ModelEntry[];
  preferredModelHaiku?: string | null;
  preferredModelSonnet?: string | null;
  preferredModelOpus?: string | null;
}

/**
 * Pick the outgoing provider type actually used for an "openai" entry.
 *
 * router.yaml's `endpointType: openai` is meant to cover both the real
 * OpenAI API and any third-party OpenAI-compatible gateway (vLLM, SGLang,
 * NVIDIA NIM, aggregators...). Some of those gateways stream
 * `reasoning_content` deltas that carry no `content` in the same chunk;
 * treating them identically to plain "openai" is fine here since our own
 * OpenAI-compatible adapter (unlike litellm's plain "openai" provider) does
 * not discard reasoning-only chunks. This distinction is kept mainly so
 * `custom_llm_provider`/diagnostics reflect what litellm would have called
 * it, in case that value is surfaced anywhere (logs, health endpoint).
 */
export function effectiveCustomLlmProvider(resolved: ResolvedModel): string {
  if (resolved.endpointType !== "openai") return resolved.endpointType;
  if (!resolved.providerBaseUrl) return "openai";
  let host = "";
  try {
    host = new URL(resolved.providerBaseUrl).hostname.toLowerCase();
  } catch {
    return "hosted_vllm";
  }
  if (REAL_OPENAI_HOSTS.has(host) || host.endsWith(".openai.com")) return "openai";
  return "hosted_vllm";
}

// ${ENV_VAR} or ${ENV_VAR:-default_value}
const ENV_VAR_PATTERN = /\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}/g;

function interpolateEnvString(value: string): string {
  return value.replace(ENV_VAR_PATTERN, (_match, varName: string, hasDefaultGroup: string | undefined, def: string | undefined) => {
    const envValue = process.env[varName];
    if (envValue !== undefined) return envValue;
    return hasDefaultGroup !== undefined ? def ?? "" : "";
  });
}

function interpolateEnv(value: unknown): unknown {
  if (typeof value === "string") return interpolateEnvString(value);
  if (Array.isArray(value)) return value.map(interpolateEnv);
  if (value !== null && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = interpolateEnv(v);
    }
    return out;
  }
  return value;
}

function isStringRecord(v: unknown): v is Record<string, string> {
  return (
    v !== null &&
    typeof v === "object" &&
    !Array.isArray(v) &&
    Object.entries(v as Record<string, unknown>).every(
      ([k, val]) => typeof k === "string" && typeof val === "string"
    )
  );
}

export function parseRouterConfig(raw: unknown): RouterConfig {
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw new RouterConfigError("router.yaml must contain a YAML mapping at the top level");
  }
  const rawObj = raw as Record<string, unknown>;

  const rawModels = rawObj.models;
  if (!Array.isArray(rawModels) || rawModels.length === 0) {
    throw new RouterConfigError("router.yaml: 'models' must be a non-empty list");
  }

  const models: ModelEntry[] = [];
  const idsSeen = new Set<string>();

  rawModels.forEach((rawEntryUnknown, idx) => {
    if (rawEntryUnknown === null || typeof rawEntryUnknown !== "object" || Array.isArray(rawEntryUnknown)) {
      throw new RouterConfigError(`Entry ${idx}: must be a mapping`);
    }
    const entry = interpolateEnv(rawEntryUnknown) as Record<string, unknown>;

    for (const fieldName of REQUIRED_MODEL_FIELDS) {
      if (!entry[fieldName]) {
        throw new RouterConfigError(`Entry ${idx}: missing required field '${fieldName}'`);
      }
    }

    const endpointType = entry.endpointType as string;
    if (!VALID_ENDPOINT_TYPES.has(endpointType)) {
      throw new RouterConfigError(
        `Entry ${idx}: invalid endpointType '${endpointType}' (must be one of ${[...VALID_ENDPOINT_TYPES].sort().join(", ")})`
      );
    }

    const rawDisableReasoning = entry.disable_reasoning;
    const rawEffortMap = entry.reasoning_effort_map;
    if (rawEffortMap !== undefined && rawEffortMap !== null && !isStringRecord(rawEffortMap)) {
      throw new RouterConfigError(`Entry ${idx}: 'reasoning_effort_map' must be a mapping of string to string`);
    }

    const rawRequestHeaders = entry.requestHeaders;
    if (rawRequestHeaders !== undefined && rawRequestHeaders !== null && !isStringRecord(rawRequestHeaders)) {
      throw new RouterConfigError(`Entry ${idx}: 'requestHeaders' must be a mapping of string to string`);
    }

    models.push({
      id: entry.id as string,
      providerModelName: entry.providerModelName as string,
      endpointType,
      providerApiKey: entry.providerApiKey as string,
      providerBaseUrl: (entry.providerBaseUrl as string) || null,
      displayName: (entry.displayName as string) || null,
      disableReasoning: rawDisableReasoning !== undefined && rawDisableReasoning !== null ? Boolean(rawDisableReasoning) : null,
      reasoningEffortMap: (rawEffortMap as Record<string, string>) ?? null,
      requestHeaders: (rawRequestHeaders as Record<string, string>) ?? null,
    });
    idsSeen.add(entry.id as string);
  });

  const preferredModel = interpolateEnv(rawObj.preferredModel) as string | undefined;
  if (!preferredModel) {
    throw new RouterConfigError("router.yaml: 'preferredModel' is required");
  }

  const tierOverride = (key: string): string | null => {
    const v = rawObj[key];
    return v ? (interpolateEnv(v) as string) : null;
  };

  const preferredHaiku = tierOverride("preferredModelHaiku");
  const preferredSonnet = tierOverride("preferredModelSonnet");
  const preferredOpus = tierOverride("preferredModelOpus");

  for (const [label, target] of [
    ["preferredModel", preferredModel],
    ["preferredModelHaiku", preferredHaiku],
    ["preferredModelSonnet", preferredSonnet],
    ["preferredModelOpus", preferredOpus],
  ] as const) {
    if (target !== null && target !== undefined && !idsSeen.has(target)) {
      throw new RouterConfigError(`router.yaml: ${label}='${target}' does not match any models[].id`);
    }
  }

  return {
    preferredModel,
    models,
    preferredModelHaiku: preferredHaiku,
    preferredModelSonnet: preferredSonnet,
    preferredModelOpus: preferredOpus,
  };
}

export function loadRouterConfig(path: string): RouterConfig {
  if (!fs.existsSync(path)) {
    throw new RouterConfigError("router.yaml not found. Please create it.");
  }
  let raw: unknown;
  try {
    raw = yaml.load(fs.readFileSync(path, "utf-8"));
  } catch (e) {
    throw new RouterConfigError(`router.yaml: malformed YAML: ${(e as Error).message}`);
  }
  return parseRouterConfig(raw ?? {});
}

export let ROUTER_CONFIG: RouterConfig;
try {
  ROUTER_CONFIG = loadRouterConfig(ROUTER_CONFIG_PATH);
} catch (e) {
  if (e instanceof RouterConfigError) {
    // eslint-disable-next-line no-console
    console.error(e.message);
    process.exit(1);
  }
  throw e;
}

/**
 * Id Claude Code's gateway model discovery will actually surface.
 *
 * Claude Code's /v1/models picker (enabled client-side with
 * CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1) ignores any discovered id
 * that doesn't start with "claude" or "anthropic", so router.yaml ids like
 * "glm-5.2" need a "claude-" prefix to show up as selectable entries.
 */
export function discoveryModelId(entryId: string): string {
  const lowered = entryId.toLowerCase();
  if (lowered.startsWith("claude") || lowered.startsWith("anthropic")) return entryId;
  return `claude-${entryId}`;
}

function resolveEntry(entry: ModelEntry): ResolvedModel {
  return {
    ...entry,
    litellmModel: entry.providerModelName,
    litellmProvider: entry.endpointType,
  };
}

/**
 * Match a client-sent `model` string directly against a router.yaml id.
 *
 * Lets users pick a specific router.yaml model from Claude Code's `/model`
 * picker (via gateway model discovery, see discoveryModelId) instead of only
 * ever landing on one of the haiku/sonnet/opus tier mappings.
 */
export function findEntryByClientModel(modelStr: string | undefined | null): ModelEntry | null {
  if (!modelStr) return null;
  for (const entry of ROUTER_CONFIG.models) {
    if (modelStr === entry.id || modelStr === discoveryModelId(entry.id)) return entry;
  }
  return null;
}

export type ModelTier = "haiku" | "sonnet" | "opus";

/**
 * Resolve a request onto its configured upstream model.
 *
 * A `directModel` that names a router.yaml id (or its discovery-prefixed
 * form) wins outright; otherwise falls back to the haiku/sonnet/opus tier
 * mapping.
 */
export function getModel(tier?: ModelTier | null, directModel?: string | null): ResolvedModel {
  const directEntry = findEntryByClientModel(directModel);
  if (directEntry !== null) return resolveEntry(directEntry);

  let targetId: string;
  if (tier === "haiku") targetId = ROUTER_CONFIG.preferredModelHaiku || ROUTER_CONFIG.preferredModel;
  else if (tier === "sonnet") targetId = ROUTER_CONFIG.preferredModelSonnet || ROUTER_CONFIG.preferredModel;
  else if (tier === "opus") targetId = ROUTER_CONFIG.preferredModelOpus || ROUTER_CONFIG.preferredModel;
  else targetId = ROUTER_CONFIG.preferredModel;

  for (const entry of ROUTER_CONFIG.models) {
    if (entry.id === targetId) return resolveEntry(entry);
  }

  // Startup validation guarantees every preferred* id resolves; this only
  // triggers if router.yaml is mutated after the process started.
  throw new RouterConfigError(`No model entry found for id '${targetId}' (tier=${String(tier)})`);
}
