# Anthropic API Proxy for Gemini & OpenAI Models 🔄

**Use Anthropic clients (like Claude Code) with Gemini, OpenAI, or direct Anthropic backends.** 🤝

A proxy server that lets you use Anthropic clients with Gemini, OpenAI, or Anthropic models themselves (a transparent proxy of sorts), all via LiteLLM. 🌉


![Anthropic API Proxy](pic.png)

## Quick Start ⚡

### Prerequisites

- API key(s) for whichever upstream provider(s) you route to 🔑
- [uv](https://github.com/astral-sh/uv) installed.

### Setup 🛠️

#### From source

1. **Clone this repository**:
   ```bash
   git clone https://github.com/1rgs/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **Install uv** (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *(`uv` will handle dependencies based on `pyproject.toml` when you run the server)*

3. **Configure routing**:
   Model routing (which upstream models to use, their endpoint URLs, and API keys)
   lives in `router.yaml`, which is git-ignored since it holds secrets. Copy the
   template and fill it in:
   ```bash
   cp router.yaml.example router.yaml
   ```
   Since `router.yaml` is git-ignored, you can put real key values directly in
   it -- there's no extra risk versus keeping them in `.env`:
   ```yaml
   preferredModel: openai/gpt-4.1

   models:
     - id: openai/gpt-4.1
       providerModelName: gpt-4.1
       endpointType: openai
       providerApiKey: sk-...
   ```
   `router.yaml` also supports `${ENV_VAR}` / `${ENV_VAR:-default}`
   interpolation if you'd rather keep the actual values in `.env` (copy
   `.env.example` to `.env`) and reference them instead -- useful when the
   same `router.yaml` needs to work across multiple environments/machines
   without editing the file itself:
   ```yaml
       providerApiKey: ${OPENAI_API_KEY}
   ```
   The server refuses to start if `router.yaml` is missing, malformed, or if
   `preferredModel` (or any tier override) doesn't match a `models[].id`. See
   [Model Routing](#model-routing-) below for the full schema.

4. **Run the server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *(`--reload` is optional, for development)*

#### Docker

If using docker, download `router.yaml.example` and (optionally) `.env.example`, then edit them as described above.
```bash
curl -o router.yaml https://raw.githubusercontent.com/1rgs/claude-code-proxy/refs/heads/main/router.yaml.example
curl -o .env https://raw.githubusercontent.com/1rgs/claude-code-proxy/refs/heads/main/.env.example
```
`router.yaml` must be mounted into the container (e.g. `-v ./router.yaml:/app/router.yaml:ro`) since it isn't baked into the image.

Then, you can either start the container with [docker compose](https://docs.docker.com/compose/) (preferred):

```yml
services:
  proxy:
    image: ghcr.io/1rgs/claude-code-proxy:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - 8082:8082
```

Or with a command:

```bash
docker run -d --env-file .env -p 8082:8082 ghcr.io/1rgs/claude-code-proxy:latest
```

### Using with Claude Code 🎮

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **That's it!** Your Claude Code client will now use the configured backend models (defaulting to Gemini) through the proxy. 🎯

## Model Routing 🗺️

Model routing is entirely configured in `router.yaml` (git-ignored — copy
`router.yaml.example` to get started). It replaces the old
`BIG_MODEL`/`SMALL_MODEL`/`MIDDLE_MODEL`/`PREFERRED_PROVIDER` environment
variables, which are no longer read by the server.

Claude Code addresses three tiers (`haiku`, `sonnet`, `opus`), detected from the
`model` string the client sends. Each tier resolves to a model `id` in
`router.yaml`:

1. `preferredModelHaiku` / `preferredModelSonnet` / `preferredModelOpus`, if set
2. Otherwise `preferredModel` (required)

```yaml
# router.yaml
preferredModel: minimax/minimax-m3
preferredModelHaiku: openai/gpt-4.1-mini   # optional per-tier override
preferredModelSonnet: minimax/minimax-m3
preferredModelOpus: anthropic/claude-opus-4-20250514

models:
  - id: minimax/minimax-m3              # LiteLLM routing id (lookup key)
    providerModelName: minimaxai/minimax-m3  # exact string sent upstream
    endpointType: openai                 # openai | anthropic | gemini
    providerBaseUrl: https://api.example.com/v1  # empty/omitted = provider default
    providerApiKey: ${MINIMAX_API_KEY}  # supports ${VAR} / ${VAR:-default}
    displayName: "MiniMax M3"           # optional, logging/health only
    disable_reasoning: false            # set true to opt a non-reasoning
                                         # model out of reasoning_effort
    reasoning_effort_map:                # optional, remap Claude's /effort
      xhigh: high                        # levels this model/gateway rejects
      max: high
    requestHeaders:                      # optional, extra headers for just
      x-gateway-route: opus-pool         # this model (any endpointType)
```

`id` and `providerModelName` are deliberately separate: some upstream gateways
expect a model name (`minimaxai/minimax-m3`) that LiteLLM would otherwise try to
route internally as a different provider (`minimax/minimax-m3`). `id` is only
ever used for lookup within this proxy; `providerModelName` is the literal
string sent to the upstream API, and `endpointType` tells LiteLLM which
protocol/provider to speak.

`endpointType: openai` covers both the real OpenAI API and any third-party
OpenAI-compatible gateway (vLLM, SGLang, NVIDIA NIM, aggregators like
agentrouter...) — you never need to distinguish them in router.yaml. Under the
hood, the proxy auto-detects non-OpenAI base URLs and routes those through
LiteLLM's `hosted_vllm` provider instead of `openai`, because LiteLLM's plain
`openai` provider silently drops streamed reasoning traces for third-party
gateways: any SSE chunk that carries only `reasoning_content` (no `content`)
gets treated as empty and discarded, so extended-thinking output vanishes
mid-stream even though the upstream bytes clearly contain it. `hosted_vllm`
handles the identical request/response shape but has a working
reasoning_content path in streaming. This is invisible from router.yaml.

`disable_reasoning` defaults to unset, meaning `thinking` requests from the
client are forwarded upstream as `reasoning_effort` for every model — most
OpenAI-compatible gateway models (glm, minimax, nemotron, ...) are
reasoning-capable even though their name doesn't match the server's built-in
reasoning-model pattern (`o1-o4`/`gpt-5`/`deepseek-r`/`qwq`/`grok-reasoning`).
Set `disable_reasoning: true` on an entry for a genuinely non-reasoning model
(e.g. `gpt-4.1`) that should just ignore thinking requests instead.

### `endpointType: anthropic` is a transparent passthrough

Unlike `openai`/`hosted_vllm`/`gemini` entries, an `endpointType: anthropic`
model skips the OpenAI-shaped translation pipeline entirely for `/v1/messages`:
the client's request body and headers are forwarded to the configured backend
essentially untouched, and its response (including the raw SSE stream) is
relayed back the same way. The proxy only patches two things: the `model`
field in the response (so a tier can still resolve to a different specific
upstream model id than the client asked for, while Claude Code still sees the
model name it originally sent) and, if configured, `reasoning_effort_map`'s
remapping of `output_config.effort`. This exists because a real
Anthropic-compatible backend already speaks Claude Code's exact
wire format, so round-tripping through OpenAI shape and back only risks losing
fidelity (thinking blocks, `cache_control`, tool_use shape, `stop_reason`, ...)
for no benefit. `DUMP_EVENTS` logging still applies; it's the only
interception on this path.

### Mapping Claude's `/effort` levels per model

Claude Code's `/effort` slash command doesn't touch `thinking.budget_tokens`
for any model it doesn't recognize as one of Anthropic's own fixed-thinking
models — which is every `router.yaml` id. Instead it sends the selected level
(`low`/`medium`/`high`/`xhigh`/`max`) in a separate `output_config.effort`
field. By default the proxy passes that value through **unchanged**:
`reasoning_effort` for OpenAI-style models, or `output_config.effort` for
`endpointType: anthropic` models.

Some backends don't accept every level Claude Code can send. Use the optional
`reasoning_effort_map` on a model entry to remap specific levels to something
that backend understands:

```yaml
models:
  - id: anthropic/claude-opus-4-20250514
    providerModelName: claude-opus-4-20250514
    endpointType: anthropic
    providerApiKey: ${ANTHROPIC_API_KEY}
    reasoning_effort_map:
      xhigh: high   # this level, mapped to a level the backend accepts
      max: high
```

Only the keys you list are remapped — every other level (including ones not
yet invented) still passes through untouched. This is opt-in and per-model on
purpose: it's the operator's call whether a given backend/litellm version
needs the safety net, not something the proxy should silently impose.

**Known gap this exists to cover:** litellm 1.82.x validates
`output_config.effort` against a fixed list that does not include `"xhigh"`,
and only allows `"max"` for model names matching its own
`claude-opus-4-6`-ish pattern. Sending `xhigh`/`max` straight through to a
real Anthropic-backed entry on an affected litellm version raises a
`ValueError` from inside litellm's transformation code — set
`reasoning_effort_map` on that entry (as above) to avoid it. This is not
proxy-side clamping; if you're on a litellm version/backend that already
accepts these values, leave the map unset and effort levels flow through
verbatim.

### Per-model header overrides

`requestHeaders` on a model entry adds/overrides headers sent upstream for
that model only, regardless of `endpointType`:

```yaml
models:
  - id: minimax/minimax-m3
    providerModelName: minimaxai/minimax-m3
    endpointType: openai
    providerApiKey: ${MINIMAX_API_KEY}
    requestHeaders:
      x-gateway-route: pool-a
```

For `endpointType: anthropic` this layers on top of the forwarded client
headers; for the other endpoint types it's added to the request LiteLLM sends
upstream. Precedence is: forwarded/default headers < `requestHeaders` <
`CUSTOM_HEADER_<NAME>` env vars (see below) — a global `CUSTOM_HEADER_*` always
wins if it sets the same header name, so it stays the one true override
regardless of what any individual model configures.

The server validates `router.yaml` at startup and refuses to start if it's
missing, malformed, has a model entry missing a required field, or if
`preferredModel`/a tier override references an `id` that doesn't exist.

**Discovering the current config:**
- `GET /health` — the resolved default model, endpoint type, host, and model count.
- `GET /v1/models` — the list of `id`s configured in `router.yaml`.

### Picking a model directly instead of haiku/sonnet/opus

Claude Code can list every `router.yaml` model as its own entry in the
`/model` picker, instead of only ever landing on whichever backend the
haiku/sonnet/opus tier happens to map to. Enable it client-side:

```bash
CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 ANTHROPIC_BASE_URL=http://localhost:8082 claude
```

Claude Code only shows discovered ids that start with `claude` or
`anthropic` ([gateway protocol reference](https://code.claude.com/docs/en/llm-gateway-protocol#model-discovery)),
so `GET /v1/models` reports each `router.yaml` id under a `claude-`-prefixed
alias (e.g. `glm-5.2` → `claude-glm-5.2`; an id already starting with
`claude`/`anthropic` is left as-is). Picking one of these in `/model` sends
that alias back as the `model` field, and the proxy resolves it straight to
the matching entry, bypassing tier mapping entirely. The haiku/sonnet/opus
tiers keep working unchanged for any model string that doesn't match a
discovered alias.

## Environment Variables ⚙️

Everything model-routing-related lives in `router.yaml` (see above). These
env vars (read from the process environment or a local `.env` file) control
proxy-wide behavior that isn't per-model. All are optional; the server runs
with sane defaults if none are set.

| Variable | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` | `INFO` | This proxy's own log verbosity. At the default `INFO`, startup config (router.yaml load, custom headers) and `DUMP_EVENTS` traffic dumps are visible as soon as `DUMP_EVENTS` is set — no need to also raise the log level. `DEBUG` additionally surfaces model-mapping, stop_reason corrections, and tool-block diagnostics. `WARNING` silences everything except actual anomalies (dropped tools/messages, malformed tool-call JSON, empty completions, request validation failures). |
| `LITELLM_LOG_LEVEL` | `WARNING` | LiteLLM's internal logger verbosity, kept separate so `LOG_LEVEL=DEBUG` shows *this proxy's* diagnostics without being buried under LiteLLM's own (often harmless) traceback noise. |
| `DUMP_EVENTS` | `none` | Which side of the translation to log: `claude` (the Claude Code ↔ proxy conversation, both directions, in Anthropic shape), `upstream` (the proxy ↔ upstream conversation, both directions, in OpenAI/litellm shape), `all` (both), or `none`. An unrecognized value logs a warning and falls back to `upstream`. Logged at `INFO`, colorized (dimmed timestamp, per-level color, dimmed `CLAUDE`/`UPSTREAM` tags, and a bold green/yellow `→`/`←` arrow showing which way data is moving relative to the proxy). |
| `MAX_TOKENS_LIMIT` | unset (no cap) | Hard ceiling on `max_tokens`/`max_completion_tokens` sent upstream, regardless of what the client requested. Set this only if a specific gateway needs one — the proxy otherwise passes the client's value through unchanged. |
| `STREAM_KEEPALIVE_SECONDS` | `3` | Seconds of silence mid-stream before the proxy emits an Anthropic `ping` event, so clients don't drop the connection while a large tool-call payload is buffered. `0` disables. |
| `UPSTREAM_IDLE_TIMEOUT` | `90` | Seconds to wait for any data from the upstream before aborting the request. Guards against a gateway that returns `200` and then never streams a body. `0` disables. |
| `ERROR_ON_EMPTY_RESPONSE` | `true` | Report a completion with no text and no tool calls as an error instead of a silent, empty turn (which would otherwise end the agent loop with nothing shown). |
| `DISABLE_STREAM_OPTIONS` | `false` | Don't send `stream_options: {include_usage: true}`. Needed for OpenAI-compatible gateways that reject the field outright; without it, streamed responses report `output_tokens: 0` and Claude Code's context meter never moves. |
| `ENABLE_CONTENT_REPLACEMENTS` | `false` | Enables text-rewriting passes on request/response content. Off by default because it's destructive for a coding agent (it can rewrite source code, file paths, and tool output). |
| `PRESERVE_UPSTREAM_TOOL_IDS` | `false` | Use the backend's own `tool_use` id verbatim instead of the proxy's synthesized one. Leave this off for backends (e.g. Kimi) that derive ids from tool name + position and repeat them across turns — Anthropic requires ids to be unique per conversation, and a repeat is reported as an interrupted tool call. |
| `CUSTOM_HEADER_<NAME>` | none | Any number of these inject a literal header on the upstream request, for every model regardless of `router.yaml`'s per-model `requestHeaders`. `<NAME>` is upper/underscore, lowercased and hyphenated to form the header name — e.g. `CUSTOM_HEADER_USER_AGENT=opencode` sends `user-agent: opencode`. Always wins over a model's own `requestHeaders` if both set the same header. Headers are also echoed back on the response to the client, except protocol-owned ones (`content-type` and similar) where doing so would break SSE streaming — the startup log states which of your configured headers fall into each group. |

Reasoning-related behavior (whether a model is asked to reason at all, and how
Claude Code's `/effort` levels map onto it) is controlled per-model in
`router.yaml` via `disable_reasoning` and `reasoning_effort_map` — see
[Model Routing](#model-routing-) above, not an env var. Whenever the upstream
response actually contains a reasoning trace, the proxy surfaces it to Claude
Code as a `thinking` block unconditionally.

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Resolving** the Claude tier (`haiku`/`sonnet`/`opus`) to a `router.yaml` model entry 🗺️
3. **Translating** the request into LiteLLM/OpenAI shape, targeting that entry's provider 🔄
4. **Sending** the translated request upstream 📤
5. **Converting** the response back to Anthropic format, echoing the original Claude model id 🔄
6. **Returning** the formatted response to the client ✅

For `endpointType: anthropic` entries, steps 3 and 5 are skipped — see
[`endpointType: anthropic` is a transparent passthrough](#endpointtype-anthropic-is-a-transparent-passthrough).

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
