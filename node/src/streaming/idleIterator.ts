import { logger } from "../logger.js";

/**
 * Yield from an upstream stream, failing loudly if it goes silent.
 *
 * A gateway can return 200 with `content-type: text/event-stream` and then
 * never send a body chunk. Awaiting that forever leaves the proxy hung with
 * nothing in the log after the response headers. Time-boxing each read turns
 * it into a reportable error.
 */
export async function* iterWithIdleTimeout<T>(
  iterable: AsyncIterable<T>,
  timeoutSeconds: number
): AsyncGenerator<T> {
  const iterator = iterable[Symbol.asyncIterator]();
  let first = true;
  while (true) {
    let result: IteratorResult<T>;
    if (timeoutSeconds && timeoutSeconds > 0) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const timedOut = Symbol("timeout");
      const timeoutPromise = new Promise<typeof timedOut>((resolve) => {
        timer = setTimeout(() => resolve(timedOut), timeoutSeconds * 1000);
      });
      try {
        const raced = await Promise.race([iterator.next(), timeoutPromise]);
        if (raced === timedOut) {
          throw new Error(
            `Upstream sent no data for ${timeoutSeconds}s after ` +
              (first ? "the response headers" : "the previous chunk") +
              ". The backend accepted the request but is not streaming; check the gateway, or try DISABLE_STREAM_OPTIONS=true and a lower MAX_TOKENS_LIMIT."
          );
        }
        result = raced as IteratorResult<T>;
      } finally {
        if (timer) clearTimeout(timer);
      }
    } else {
      result = await iterator.next();
    }
    if (result.done) return;
    if (first) {
      first = false;
      logger.debug("First upstream chunk received");
    }
    yield result.value;
  }
}
