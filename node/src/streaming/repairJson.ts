/**
 * Best-effort completion of argument JSON that was cut off mid-value.
 *
 * A truncated tool argument is unrunnable, and reporting it as max_tokens
 * halts the agent loop entirely. Closing the open strings/brackets at least
 * yields a runnable call the model can correct on the next turn.
 */
export function repairTruncatedJson(buf: string): string | null {
  if (!buf.trim()) return "{}";
  const stack: string[] = [];
  let inString = false;
  let escaped = false;
  for (const ch of buf) {
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === "\\") escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === "{" || ch === "[") stack.push(ch === "{" ? "}" : "]");
    else if (ch === "}" || ch === "]") {
      if (stack.length) stack.pop();
    }
  }

  let candidate = buf;
  if (escaped) candidate = candidate.slice(0, -1);
  if (inString) candidate += '"';
  // A dangling key with no value ("foo": ) cannot be closed meaningfully.
  const stripped = candidate.trimEnd();
  if (stripped.endsWith(":")) candidate = `${stripped}null`;
  else if (stripped.endsWith(",")) candidate = stripped.slice(0, -1);
  candidate += stack.reverse().join("");

  try {
    JSON.parse(candidate);
    return candidate;
  } catch {
    return null;
  }
}
