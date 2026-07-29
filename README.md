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

## How It Works 🧩

This proxy works by:

1. **Receiving requests** in Anthropic's API format 📥
2. **Resolving** the Claude tier (`haiku`/`sonnet`/`opus`) to a `router.yaml` model entry 🗺️
3. **Translating** the request into LiteLLM/OpenAI shape, targeting that entry's provider 🔄
4. **Sending** the translated request upstream 📤
5. **Converting** the response back to Anthropic format, echoing the original Claude model id 🔄
6. **Returning** the formatted response to the client ✅

The proxy handles both streaming and non-streaming responses, maintaining compatibility with all Claude clients. 🌊

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request. 🎁
