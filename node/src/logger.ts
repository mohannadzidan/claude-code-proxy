import pino from "pino";
import { LOG_LEVEL } from "./env.js";

/**
 * Rendering mirrors server.py's rich-based setup: a compact single-line
 * format, with DUMP_EVENTS traffic dumps ("CLAUDE -> ...", "UPSTREAM <- ...")
 * left readable. Arrow direction is relative to the proxy: "->" is data
 * leaving the proxy, "<-" is data arriving at the proxy.
 */
export const logger = pino({
  level: LOG_LEVEL,
  transport: {
    target: "pino-pretty",
    options: {
      colorize: true,
      translateTime: "HH:MM:ss",
      ignore: "pid,hostname",
      singleLine: true,
    },
  },
});

// ANSI color codes for the human-friendly request/response summary line
// (log_request_beautifully in server.py).
const Colors = {
  CYAN: "\x1b[96m",
  GREEN: "\x1b[92m",
  RED: "\x1b[91m",
  MAGENTA: "\x1b[95m",
  BLUE: "\x1b[94m",
  DIM: "\x1b[2m",
  BOLD: "\x1b[1m",
  RESET: "\x1b[0m",
};

/**
 * Log requests in a compact, human-friendly format showing Claude -> upstream
 * model mapping. `resolvedModelDisplay` must already be the router.yaml
 * displayName/id to show (not a raw providerModelName), since
 * providerModelName can itself contain slashes that would be mangled by a
 * naive split.
 */
export function logRequestBeautifully(
  method: string,
  path: string,
  claudeModel: string,
  resolvedModelDisplay: string,
  numMessages: number,
  numTools: number,
  statusCode: number | null
): void {
  const claudeDisplay = `${Colors.CYAN}${claudeModel}${Colors.RESET}`;
  const endpoint = path.split("?")[0];
  const openaiDisplay = `${Colors.GREEN}${resolvedModelDisplay}${Colors.RESET}`;
  const toolsStr = `${Colors.MAGENTA}${numTools} tools${Colors.RESET}`;
  const messagesStr = `${Colors.BLUE}${numMessages} messages${Colors.RESET}`;

  let statusStr: string;
  if (statusCode === null) {
    statusStr = `${Colors.DIM}… in flight${Colors.RESET}`;
  } else if (statusCode === 200) {
    statusStr = `${Colors.GREEN}✓ ${statusCode} OK${Colors.RESET}`;
  } else {
    statusStr = `${Colors.RED}✗ ${statusCode}${Colors.RESET}`;
  }

  const logLine = `${Colors.BOLD}${method} ${endpoint}${Colors.RESET} ${statusStr}`;
  const modelLine = `${claudeDisplay} → ${openaiDisplay} ${toolsStr} ${messagesStr}`;

  // eslint-disable-next-line no-console
  console.log(logLine);
  // eslint-disable-next-line no-console
  console.log(modelLine);
}
