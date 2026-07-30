const STATUS_TO_TYPE: Record<number, string> = {
  400: "invalid_request_error",
  401: "authentication_error",
  403: "permission_error",
  404: "not_found_error",
  413: "request_too_large",
  422: "invalid_request_error",
  429: "rate_limit_error",
  500: "api_error",
  529: "overloaded_error",
};

/** Map an HTTP status onto Anthropic's error `type` vocabulary. */
export function anthropicErrorType(statusCode: number): string {
  return STATUS_TO_TYPE[statusCode] || "api_error";
}

/**
 * Anthropic clients parse {"type":"error","error":{...}}. A framework's
 * default error envelope makes every upstream failure surface in Claude
 * Code as an opaque "API Error" with no message.
 */
export function anthropicErrorBody(statusCode: number, message: string): Record<string, unknown> {
  return {
    type: "error",
    error: { type: anthropicErrorType(statusCode), message },
  };
}
