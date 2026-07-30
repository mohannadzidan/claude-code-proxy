/** Thrown by a provider adapter on a non-2xx upstream response. */
export class ProviderHttpError extends Error {
  statusCode: number;
  body: unknown;

  constructor(statusCode: number, message: string, body: unknown) {
    super(message);
    this.name = "ProviderHttpError";
    this.statusCode = statusCode;
    this.body = body;
  }
}
