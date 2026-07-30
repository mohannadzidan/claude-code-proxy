import type { FastifyInstance } from "fastify";
import { getModel, ROUTER_CONFIG, discoveryModelId } from "../router/config.js";

export function registerMiscRoutes(app: FastifyInstance): void {
  /** Liveness probe plus the effective router.yaml mapping, for debugging setups. */
  app.get("/health", async () => {
    const resolved = getModel(null);
    let baseUrlDisplay: string | null = null;
    if (resolved.providerBaseUrl) {
      try {
        baseUrlDisplay = new URL(resolved.providerBaseUrl).host || resolved.providerBaseUrl;
      } catch {
        baseUrlDisplay = resolved.providerBaseUrl;
      }
    }
    return {
      status: "ok",
      preferredModel: resolved.id,
      endpointType: resolved.endpointType,
      providerBaseUrl: baseUrlDisplay,
      providerModelName: resolved.providerModelName,
      modelCount: ROUTER_CONFIG.models.length,
      source: "router.yaml",
    };
  });

  /**
   * Model ids configured in router.yaml, exposed under their
   * discovery-prefixed form so Claude Code's gateway model discovery
   * (CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1) surfaces each one as a
   * directly selectable "From gateway" entry in `/model`.
   */
  app.get("/v1/models", async () => {
    const seen = new Set<string>();
    const entries = [];
    for (const entry of ROUTER_CONFIG.models) {
      if (seen.has(entry.id)) continue;
      seen.add(entry.id);
      entries.push(entry);
    }
    return {
      data: entries.map((entry) => ({
        id: discoveryModelId(entry.id),
        type: "model",
        display_name: entry.displayName || entry.id,
        created_at: "2025-01-01T00:00:00Z",
      })),
      has_more: false,
    };
  });

  app.get("/", async () => ({ message: "Anthropic Proxy for LiteLLM" }));
}
