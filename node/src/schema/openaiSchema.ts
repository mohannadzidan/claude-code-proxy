const DROP_KEYS = new Set(["$schema", "$id", "$comment", "additionalItems", "cache_control"]);
const ALLOWED_STRING_FORMATS = new Set(["date-time", "date", "time", "duration", "email", "uuid"]);

/**
 * Strip JSON-Schema keywords that OpenAI-compatible backends reject.
 *
 * Anthropic tool schemas are full JSON Schema. Many OpenAI-compatible servers
 * (vLLM, Ollama, Azure, several gateways) hard-fail on $schema/$id or on
 * exotic string formats, which shows up as a blanket 400 on every request
 * that carries Claude Code's toolset.
 */
export function sanitizeJsonSchemaForOpenai(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(sanitizeJsonSchemaForOpenai);
  }
  if (schema === null || typeof schema !== "object") return schema;

  const obj = schema as Record<string, unknown>;
  const out: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (DROP_KEYS.has(key)) continue;
    if (key === "format" && obj.type === "string") {
      if (!ALLOWED_STRING_FORMATS.has(value as string)) continue;
    }
    if ((key === "properties" || key === "$defs" || key === "definitions") && value !== null && typeof value === "object" && !Array.isArray(value)) {
      const inner: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
        inner[k] = sanitizeJsonSchemaForOpenai(v);
      }
      out[key] = inner;
    } else if (key === "items" || key === "additionalProperties") {
      out[key] = sanitizeJsonSchemaForOpenai(value);
    } else if ((key === "anyOf" || key === "oneOf" || key === "allOf" || key === "prefixItems") && Array.isArray(value)) {
      out[key] = value.map(sanitizeJsonSchemaForOpenai);
    } else {
      out[key] = sanitizeJsonSchemaForOpenai(value);
    }
  }

  // An object schema with no `properties` is rejected by strict validators.
  if (out.type === "object" && !("properties" in out)) {
    out.properties = {};
  }
  return out;
}
