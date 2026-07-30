import Fastify from "fastify";
import { PORT, CUSTOM_HEADERS, RESPONSE_PROTECTED_HEADERS } from "./env.js";
import { logger } from "./logger.js";
import { ROUTER_CONFIG } from "./router/config.js";
import { registerMiscRoutes } from "./routes/misc.js";
import { registerCountTokensRoute } from "./routes/countTokens.js";

// Fastify's own request logger is disabled: all logging in this port goes
// through our pino instance (logger.ts) plus the human-friendly
// logRequestBeautifully summary line, mirroring server.py's own log calls
// rather than uvicorn's access log.
const app = Fastify({ logger: false });

app.addHook("onRequest", async (request) => {
  logger.debug(`Request: ${request.method} ${request.url}`);
});

// Inject custom headers from environment variables. Protocol headers are
// never overridden: CUSTOM_HEADER_CONTENT_TYPE is meant for the *upstream*
// request (where application/json is correct), but applying it to the
// response would replace text/event-stream on streamed replies, which stops
// the client from parsing the body as SSE.
app.addHook("onSend", async (_request, reply, payload) => {
  for (const [name, value] of Object.entries(CUSTOM_HEADERS)) {
    if (RESPONSE_PROTECTED_HEADERS.has(name.toLowerCase())) continue;
    reply.header(name, value);
  }
  return payload;
});

registerMiscRoutes(app);
registerCountTokensRoute(app);

if (Object.keys(CUSTOM_HEADERS).length > 0) {
  const upstreamOnly = Object.keys(CUSTOM_HEADERS).filter((n) => RESPONSE_PROTECTED_HEADERS.has(n.toLowerCase()));
  const applied = Object.keys(CUSTOM_HEADERS).filter((n) => !RESPONSE_PROTECTED_HEADERS.has(n.toLowerCase()));
  logger.info(`Custom headers: ${applied.length} sent upstream and echoed on responses, ${upstreamOnly.length} sent upstream only`);
} else {
  logger.debug("No CUSTOM_HEADER_* environment variables found.");
}

logger.info(`Loaded router.yaml: ${ROUTER_CONFIG.models.length} model entries. Preferred: ${ROUTER_CONFIG.preferredModel}`);

app
  .listen({ port: PORT, host: "0.0.0.0" })
  .then((address) => {
    logger.info(`Listening on ${address}`);
  })
  .catch((err) => {
    logger.error(err);
    process.exit(1);
  });
