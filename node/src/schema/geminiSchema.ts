/** Recursively removes unsupported fields from a JSON schema for Gemini. */
export function cleanGeminiSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) {
    return schema.map(cleanGeminiSchema);
  }
  if (schema !== null && typeof schema === "object") {
    const obj = schema as Record<string, unknown>;
    delete obj.additionalProperties;
    delete obj.default;

    if (obj.type === "string" && "format" in obj) {
      const allowedFormats = new Set(["enum", "date-time"]);
      if (!allowedFormats.has(obj.format as string)) {
        delete obj.format;
      }
    }

    for (const key of Object.keys(obj)) {
      obj[key] = cleanGeminiSchema(obj[key]);
    }
    return obj;
  }
  return schema;
}
