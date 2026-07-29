from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, model_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import base64
import asyncio
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass
import yaml
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables from .env file
load_dotenv()


# Configure logging. LOG_LEVEL=DEBUG surfaces the model mapping, stop_reason
# corrections and tool-block diagnostics; the default stays quiet.
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "WARN").upper(), logging.WARN),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn

# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

def get_custom_headers_from_env() -> Dict[str, str]:
    """Extract custom headers from environment variables.
    
    Variables should be in the format CUSTOM_HEADER_<HEADER_NAME> where
    the header name is derived from the part after CUSTOM_HEADER_ with
    underscores replaced by hyphens and lowercased.
    
    Examples:
        CUSTOM_HEADER_ORIGINATOR -> originator
        CUSTOM_HEADER_CONTENT_TYPE -> content-type
        CUSTOM_HEADER_USER_AGENT -> user-agent
    """
    headers = {}
    prefix = "CUSTOM_HEADER_"
    for key, value in os.environ.items():
        if key.startswith(prefix) and len(key) > len(prefix):
            header_name = key[len(prefix):]
            header_name = header_name.replace("_", "-").lower()
            headers[header_name] = value
            logger.debug(f"Loaded custom header from env: {header_name}")
    return headers

# Pre-compute custom headers at startup for efficiency
CUSTOM_HEADERS = get_custom_headers_from_env()

# Headers that must never be overwritten on an outgoing response. Replacing
# content-type on a StreamingResponse breaks SSE parsing outright; the rest are
# hop-by-hop or framing headers owned by the server.
RESPONSE_PROTECTED_HEADERS = {
    "content-type",
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "connection",
}

import re

# Define your regex patterns and their static replacement strings here
# Format: (regex_pattern, replacement_string)
CONTENT_REPLACEMENTS = [
    (re.compile(r"^\s*x-anthropic-billing-header.+\n*"), ""),
    (re.compile(r"\bclaude\b|\bAnthropic\b", flags=re.IGNORECASE), "Open"),
    # Add more patterns as needed
]

def apply_content_replacements(messages: list) -> list:
    """Applies regex replacements to message content before sending to the LLM.

    Disabled unless ENABLE_CONTENT_REPLACEMENTS=true. For a coding agent this
    rewriting is actively destructive: the default rules rewrite every occurrence
    of "claude"/"Anthropic" inside source files, paths and tool output, so the
    model proposes edits against text that does not match the user's disk.
    """
    if not ENABLE_CONTENT_REPLACEMENTS:
        return messages

    for msg in messages:
        # Never rewrite tool results: they are verbatim command/file output.
        if msg.get("role") == "tool":
            continue
        content = msg.get("content")
        
        # 1. If content is a simple string (common for OpenAI/flattened messages)
        if isinstance(content, str):
            for pattern, replacement in CONTENT_REPLACEMENTS:
                content = pattern.sub(replacement, content)
            msg["content"] = content
            
        # 2. If content is a list of blocks (common for Anthropic/multimodal messages)
        elif isinstance(content, list):
            for block in content:
                # Only modify text blocks
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    for pattern, replacement in CONTENT_REPLACEMENTS:
                        text = pattern.sub(replacement, text)
                    block["text"] = text
                    
    return messages
# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator",
        ]

        if hasattr(record, "msg") and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True


# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# LiteLLM emits a large traceback for every request when it cannot price the
# model (any model not in its cost map, e.g. a self-hosted or gateway model).
# It is harmless post-response noise, but at DEBUG it buries the proxy's own
# output. Keep LiteLLM's internal logger quiet unless LITELLM_LOG_LEVEL says
# otherwise, so LOG_LEVEL=DEBUG shows *our* diagnostics.
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(
    getattr(logging, os.environ.get("LITELLM_LOG_LEVEL", "WARNING").upper(), logging.WARNING)
)
for _n in ("litellm", "LiteLLM Proxy", "LiteLLM Router"):
    logging.getLogger(_n).setLevel(
        getattr(logging, os.environ.get("LITELLM_LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    )


# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""

    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)


# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(
            ColorizedFormatter("%(asctime)s - %(levelname)s - %(message)s")
        )

app = FastAPI()


# ---------------------------------------------------------------------------
# router.yaml: centralized model routing configuration.
#
# Replaces the old BIG_MODEL/SMALL_MODEL/MIDDLE_MODEL/PREFERRED_PROVIDER/
# OPENAI_BASE_URL/*_API_KEY env vars. Those are no longer read by this server.
# ---------------------------------------------------------------------------

class RouterConfigError(Exception):
    """Raised when router.yaml is missing, malformed, or fails validation."""


# "hosted_vllm" is litellm's provider type for self-hosted/third-party
# OpenAI-compatible inference servers (vLLM, SGLang, NVIDIA NIM, ...). It is
# not meant to be written in router.yaml -- see _effective_custom_llm_provider
# below, which upgrades "openai" entries to it automatically based on
# providerBaseUrl. Kept in this set only as an explicit-override escape hatch.
VALID_ENDPOINT_TYPES = {"openai", "hosted_vllm", "anthropic", "gemini"}

# Real OpenAI API hosts; anything else behind endpointType: openai is a
# third-party OpenAI-compatible gateway.
_REAL_OPENAI_HOSTS = {"api.openai.com"}


def _effective_custom_llm_provider(resolved: "ResolvedModel") -> str:
    """Pick the litellm custom_llm_provider actually sent upstream.

    router.yaml's `endpointType: openai` is meant to cover both the real
    OpenAI API and any third-party OpenAI-compatible gateway (vLLM, SGLang,
    NVIDIA NIM, aggregators like agentrouter...). litellm's plain "openai"
    provider silently drops streamed reasoning traces for the latter: any SSE
    delta that carries only `reasoning_content` (no `content`) is treated as
    an empty chunk and discarded, so extended-thinking output vanishes
    mid-stream even though the upstream bytes clearly contain it (confirmed
    directly against NVIDIA NIM). litellm's "hosted_vllm" provider handles the
    identical request/response shape but has a working reasoning_content path.

    Auto-upgrade "openai" to "hosted_vllm" whenever providerBaseUrl isn't
    actually OpenAI's, so router.yaml authors never need to know this litellm
    quirk exists. An explicit `endpointType: hosted_vllm` always wins.
    """
    if resolved.endpointType != "openai":
        return resolved.endpointType
    if not resolved.providerBaseUrl:
        # No override -> litellm's own default api_base, which is OpenAI's.
        return "openai"
    host = urlparse(resolved.providerBaseUrl).netloc.lower().split(":")[0]
    if host in _REAL_OPENAI_HOSTS or host.endswith(".openai.com"):
        return "openai"
    return "hosted_vllm"
REQUIRED_MODEL_FIELDS = ("id", "providerModelName", "endpointType", "providerApiKey")

# ${ENV_VAR} or ${ENV_VAR:-default_value}
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")


def _interpolate_env_string(value: str) -> str:
    def _replace(match: "re.Match") -> str:
        var_name = match.group(1)
        has_default = match.group(2) is not None
        default = match.group(3) if has_default else None
        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        return default if has_default else ""

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _interpolate_env(value: Any) -> Any:
    """Recursively apply ${ENV_VAR} / ${ENV_VAR:-default} interpolation."""
    if isinstance(value, str):
        return _interpolate_env_string(value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


@dataclass
class ModelEntry:
    id: str
    providerModelName: str
    endpointType: str
    providerApiKey: str
    providerBaseUrl: Optional[str] = None
    displayName: Optional[str] = None
    # Explicit override for reasoning-capable OpenAI-compatible models (glm,
    # minimax, nemotron, ...) that REASONING_MODEL_PATTERN's name-based
    # whitelist doesn't recognise. None means "fall back to the pattern".
    reasoning: Optional[bool] = None


@dataclass
class ResolvedModel:
    id: str
    providerModelName: str
    endpointType: str
    providerApiKey: str
    providerBaseUrl: Optional[str] = None
    displayName: Optional[str] = None
    reasoning: Optional[bool] = None

    @property
    def litellm_model(self) -> str:
        return self.providerModelName

    @property
    def litellm_provider(self) -> str:
        return self.endpointType


@dataclass
class RouterConfig:
    preferredModel: str
    models: List[ModelEntry]
    preferredModelHaiku: Optional[str] = None
    preferredModelSonnet: Optional[str] = None
    preferredModelOpus: Optional[str] = None


def _parse_router_config(raw: Dict[str, Any]) -> RouterConfig:
    if not isinstance(raw, dict):
        raise RouterConfigError(
            "router.yaml must contain a YAML mapping at the top level"
        )

    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise RouterConfigError("router.yaml: 'models' must be a non-empty list")

    models: List[ModelEntry] = []
    ids_seen: set = set()
    for idx, raw_entry in enumerate(raw_models):
        if not isinstance(raw_entry, dict):
            raise RouterConfigError(f"Entry {idx}: must be a mapping")
        entry = _interpolate_env(raw_entry)

        for field_name in REQUIRED_MODEL_FIELDS:
            if not entry.get(field_name):
                raise RouterConfigError(
                    f"Entry {idx}: missing required field '{field_name}'"
                )

        endpoint_type = entry["endpointType"]
        if endpoint_type not in VALID_ENDPOINT_TYPES:
            raise RouterConfigError(
                f"Entry {idx}: invalid endpointType '{endpoint_type}' "
                f"(must be one of {sorted(VALID_ENDPOINT_TYPES)})"
            )

        raw_reasoning = entry.get("reasoning")
        models.append(
            ModelEntry(
                id=entry["id"],
                providerModelName=entry["providerModelName"],
                endpointType=endpoint_type,
                providerApiKey=entry["providerApiKey"],
                providerBaseUrl=entry.get("providerBaseUrl") or None,
                displayName=entry.get("displayName") or None,
                reasoning=bool(raw_reasoning) if raw_reasoning is not None else None,
            )
        )
        ids_seen.add(entry["id"])

    preferred_model = _interpolate_env(raw.get("preferredModel"))
    if not preferred_model:
        raise RouterConfigError("router.yaml: 'preferredModel' is required")

    def _tier_override(key: str) -> Optional[str]:
        v = raw.get(key)
        return _interpolate_env(v) if v else None

    preferred_haiku = _tier_override("preferredModelHaiku")
    preferred_sonnet = _tier_override("preferredModelSonnet")
    preferred_opus = _tier_override("preferredModelOpus")

    for label, target in (
        ("preferredModel", preferred_model),
        ("preferredModelHaiku", preferred_haiku),
        ("preferredModelSonnet", preferred_sonnet),
        ("preferredModelOpus", preferred_opus),
    ):
        if target is not None and target not in ids_seen:
            raise RouterConfigError(
                f"router.yaml: {label}='{target}' does not match any models[].id"
            )

    return RouterConfig(
        preferredModel=preferred_model,
        models=models,
        preferredModelHaiku=preferred_haiku,
        preferredModelSonnet=preferred_sonnet,
        preferredModelOpus=preferred_opus,
    )


def load_router_config(path: Path) -> RouterConfig:
    if not path.exists():
        raise RouterConfigError("router.yaml not found. Please create it.")
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RouterConfigError(f"router.yaml: malformed YAML: {e}")
    return _parse_router_config(raw or {})


ROUTER_CONFIG_PATH = Path(__file__).resolve().parent / "router.yaml"

try:
    ROUTER_CONFIG = load_router_config(ROUTER_CONFIG_PATH)
except RouterConfigError as e:
    logger.critical(str(e))
    sys.exit(1)

logger.warning(
    f"Loaded router.yaml: {len(ROUTER_CONFIG.models)} model entries. "
    f"Preferred: {ROUTER_CONFIG.preferredModel}"
)


def getModel(tier: Optional[Literal["haiku", "sonnet", "opus"]] = None) -> ResolvedModel:
    """Resolve a Claude tier onto its configured upstream model."""
    if tier == "haiku":
        target_id = ROUTER_CONFIG.preferredModelHaiku or ROUTER_CONFIG.preferredModel
    elif tier == "sonnet":
        target_id = ROUTER_CONFIG.preferredModelSonnet or ROUTER_CONFIG.preferredModel
    elif tier == "opus":
        target_id = ROUTER_CONFIG.preferredModelOpus or ROUTER_CONFIG.preferredModel
    else:
        target_id = ROUTER_CONFIG.preferredModel

    for entry in ROUTER_CONFIG.models:
        if entry.id == target_id:
            return ResolvedModel(
                id=entry.id,
                providerModelName=entry.providerModelName,
                endpointType=entry.endpointType,
                providerApiKey=entry.providerApiKey,
                providerBaseUrl=entry.providerBaseUrl,
                displayName=entry.displayName,
                reasoning=entry.reasoning,
            )

    # Startup validation guarantees every preferred* id resolves; this only
    # triggers if router.yaml is mutated after the process started.
    raise RouterConfigError(f"No model entry found for id '{target_id}' (tier={tier!r})")


# Output-token ceiling. The old code hardcoded 16384, which silently truncated
# long edits on models that support far more.
MAX_TOKENS_LIMIT = int(os.environ.get("MAX_TOKENS_LIMIT", "0")) or None

# Some OpenAI-compatible gateways only understand `max_tokens` and reject
# `max_completion_tokens`. Set FORCE_MAX_TOKENS_PARAM=max_tokens for those.
MAX_TOKENS_PARAM = os.environ.get("FORCE_MAX_TOKENS_PARAM", "auto").lower()

# The claude->Open text rewriting is destructive for a coding agent (it rewrites
# source code, file paths and tool output). Off unless explicitly enabled.
ENABLE_CONTENT_REPLACEMENTS = (
    os.environ.get("ENABLE_CONTENT_REPLACEMENTS", "false").lower() == "true"
)

# Surface upstream reasoning traces as Anthropic thinking blocks. Default "auto":
# only when the client actually asked for thinking. Emitting unrequested thinking
# blocks (and with no valid signature, which we cannot produce for a non-Anthropic
# backend) makes clients discard the message and end the turn silently. Set to
# "always" to force, "never" to drop reasoning entirely.
EMIT_REASONING = os.environ.get("EMIT_REASONING", "auto").lower()

# Some OpenAI-compatible gateways reject the stream_options field outright.
DISABLE_STREAM_OPTIONS = (
    os.environ.get("DISABLE_STREAM_OPTIONS", "false").lower() == "true"
)

# DUMP_EVENTS=true logs every Anthropic SSE event the proxy emits, plus the
# outgoing upstream payload. This is the fastest way to see what a client is
# actually receiving when a turn ends unexpectedly.
DUMP_EVENTS = os.environ.get("DUMP_EVENTS", "false").lower() == "true"

# Seconds of silence before the proxy emits a keep-alive ping mid-stream. Tool
# calls are buffered until the stream ends, so a large Edit/Write payload can
# otherwise produce tens of seconds with no SSE traffic at all, and clients drop
# the connection. Anthropic's protocol allows ping events anywhere in a stream.
# Set to 0 to disable.
STREAM_KEEPALIVE_SECONDS = float(os.environ.get("STREAM_KEEPALIVE_SECONDS", "3"))

# Abort if the upstream sends no data for this long. A gateway returning 200 and
# then never streaming a body used to hang the request forever with nothing in
# the log after the response headers. 0 disables.
UPSTREAM_IDLE_TIMEOUT = float(os.environ.get("UPSTREAM_IDLE_TIMEOUT", "90"))

# COMPAT_MODE=true drops every request field this proxy added over the older,
# known-working shape: no stream_options, no `user`, and max_tokens capped at
# 16384. Strict or older OpenAI-compatible gateways (one-api / new-api forks and
# similar) can accept a request carrying unknown fields with 200 and then never
# stream a body. This keeps all the response-side and tool-handling fixes while
# sending exactly what the pre-patch proxy sent.
COMPAT_MODE = os.environ.get("COMPAT_MODE", "false").lower() == "true"

# Report an empty assistant turn as an error rather than letting it look like a
# normal finish. An empty completion silently ends the agent loop.
ERROR_ON_EMPTY_RESPONSE = (
    os.environ.get("ERROR_ON_EMPTY_RESPONSE", "true").lower() == "true"
)

# Placeholder signature attached to synthesized thinking blocks. Real Anthropic
# signatures are cryptographic and cannot be produced for a third-party backend;
# this value only has to be non-empty and stable, since the proxy strips thinking
# blocks out of conversation history before forwarding them upstream.
PROXY_THINKING_SIGNATURE = base64.b64encode(b"litellm-proxy-unsigned-thinking").decode()


def _should_emit_thinking(original_request) -> bool:
    if EMIT_REASONING == "never":
        return False
    if EMIT_REASONING == "always":
        return True
    thinking = getattr(original_request, "thinking", None)
    return bool(thinking is not None and thinking.is_enabled)

# Reasoning models reject temperature/top_p/stop and need max_completion_tokens.
REASONING_MODEL_PATTERN = re.compile(
    r"^(o[1-4]|gpt-5|gpt-4\.5-preview|deepseek-r|qwq|grok-.*-reasoning)", re.IGNORECASE
)


def strip_provider_prefix(model: str) -> str:
    for prefix in ("anthropic/", "openai/", "gemini/", "vertex_ai/", "azure/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def is_reasoning_model(model: str) -> bool:
    """True for models that use the restricted reasoning-model parameter set."""
    return bool(REASONING_MODEL_PATTERN.match(strip_provider_prefix(model)))


def supports_reasoning_effort(resolved: "ResolvedModel") -> bool:
    """True when `reasoning_effort` should be forwarded for this upstream model.

    REASONING_MODEL_PATTERN only recognises OpenAI's own reasoning-model
    naming (o1-o4, gpt-5, ...). Reasoning-capable models served through
    OpenAI-compatible gateways (glm, minimax, nemotron, ...) don't match that
    pattern and never got `reasoning_effort`, so they silently never reasoned
    even when the client asked for extended thinking. router.yaml's
    `reasoning: true/false` on a model entry overrides the pattern; when unset
    the pattern is the fallback.
    """
    if resolved.reasoning is not None:
        return resolved.reasoning
    return is_reasoning_model(resolved.litellm_model)


def supports_temperature(model: str) -> bool:
    return not is_reasoning_model(model)


def extract_tier(v: str) -> Optional[Literal["haiku", "sonnet", "opus"]]:
    """Extract the Claude tier (haiku/sonnet/opus) from an incoming model id.

    router.yaml resolves the actual upstream model; this only determines which
    tier's preferredModel* the request belongs to.
    """
    lowered = strip_provider_prefix(v or "").lower()
    if "haiku" in lowered:
        return "haiku"
    if "sonnet" in lowered:
        return "sonnet"
    if "opus" in lowered:
        return "opus"
    return None


# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(
                    f"Removing unsupported format '{schema['format']}' for string type in Gemini schema."
                )
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(
            schema.items()
        ):  # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema


# Models for Anthropic API requests
#
# NOTE: every block model allows extra fields. Real Anthropic clients (Claude Code
# in particular) attach `cache_control`, `citations`, `signature`, etc. to blocks.
# Rejecting those with a 422 is the single most common cause of "the proxy works
# with curl but not with Claude Code".
class ContentBlockText(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockDocument(BaseModel):
    """PDF / plain-text document blocks (Anthropic 'document' content block)."""

    model_config = {"extra": "allow"}
    type: Literal["document"]
    source: Dict[str, Any]
    title: Optional[str] = None
    context: Optional[str] = None


class ContentBlockThinking(BaseModel):
    """Extended-thinking block echoed back in assistant turns."""

    model_config = {"extra": "allow"}
    type: Literal["thinking"]
    thinking: str = ""
    signature: Optional[str] = None


class ContentBlockRedactedThinking(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["redacted_thinking"]
    data: Optional[str] = None


class ContentBlockToolUse(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ContentBlockToolResult(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any] = ""
    # Anthropic signals failed tool calls with is_error=True. Dropping this makes
    # the model think every failed command succeeded.
    is_error: Optional[bool] = None


# Any block type we do not model explicitly (server_tool_use, web_search_tool_result,
# mcp_tool_use, future additions...) falls through to this instead of 422-ing.
class ContentBlockUnknown(BaseModel):
    model_config = {"extra": "allow"}
    type: str


AnyContentBlock = Union[
    ContentBlockText,
    ContentBlockImage,
    ContentBlockDocument,
    ContentBlockThinking,
    ContentBlockRedactedThinking,
    ContentBlockToolUse,
    ContentBlockToolResult,
    ContentBlockUnknown,
]


class SystemContent(BaseModel):
    model_config = {"extra": "allow"}
    type: Literal["text"]
    text: str


class Message(BaseModel):
    model_config = {"extra": "allow"}
    # Anthropic's Messages API only accepts user/assistant here; system goes in the
    # top-level `system` field. We stay permissive on input and normalize later.
    role: str
    content: Union[str, List[AnyContentBlock]]


class Tool(BaseModel):
    model_config = {"extra": "allow"}
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    # Anthropic server-side tools (web_search, text_editor, bash...) carry a `type`
    # instead of a schema. They have no OpenAI equivalent and must be dropped.
    type: Optional[str] = None


class ThinkingConfig(BaseModel):
    """Anthropic sends {"type": "enabled"|"disabled", "budget_tokens": N}."""

    model_config = {"extra": "allow"}
    type: Optional[str] = None
    budget_tokens: Optional[int] = None
    enabled: bool = True

    @property
    def is_enabled(self) -> bool:
        if self.type is not None:
            return self.type != "disabled"
        return self.enabled


class MessagesRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    # Default None (not 1.0): sending an explicit temperature to a reasoning model
    # is a hard 400, and forwarding a default the caller never set is wrong anyway.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Dict[str, Any], str]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    model_tier: Optional[Literal["haiku", "sonnet", "opus"]] = None  # Derived from original_model

    # mode="before" is required: a field_validator cannot populate a *sibling*
    # field. The previous code assigned into `info.data`, which Pydantic v2
    # discards, so original_model was always None and every response echoed the
    # mapped backend id (e.g. "openai/gpt-4.1") instead of the model the client
    # asked for. Claude Code compares that field against what it sent.
    @model_validator(mode="before")
    @classmethod
    def capture_original_model(cls, data):
        if isinstance(data, dict) and data.get("model") is not None:
            data.setdefault("original_model", data["model"])
        return data

    @model_validator(mode="after")
    def compute_model_tier(self):
        self.model_tier = extract_tier(self.original_model or self.model)
        return self


class TokenCountRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Union[Dict[str, Any], str]] = None
    original_model: Optional[str] = None  # Will store the original model name
    model_tier: Optional[Literal["haiku", "sonnet", "opus"]] = None  # Derived from original_model

    @model_validator(mode="before")
    @classmethod
    def capture_original_model(cls, data):
        if isinstance(data, dict) and data.get("model") is not None:
            data.setdefault("original_model", data["model"])
        return data

    @model_validator(mode="after")
    def compute_model_tier(self):
        self.model_tier = extract_tier(self.original_model or self.model)
        return self


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockThinking, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal[
            "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
        ]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


@app.on_event("startup")
async def startup_event():
    if CUSTOM_HEADERS:
        upstream_only = [
            n for n in CUSTOM_HEADERS if n.lower() in RESPONSE_PROTECTED_HEADERS
        ]
        applied = [n for n in CUSTOM_HEADERS if n.lower() not in RESPONSE_PROTECTED_HEADERS]
        logger.warning(
            f"Custom headers: {len(applied)} sent upstream and echoed on responses, "
            f"{len(upstream_only)} sent upstream only"
        )
        for name in applied:
            value = CUSTOM_HEADERS[name]
            display_value = value if len(value) < 20 else value[:6] + "..."
            logger.warning(f"  {name}: {display_value}")
        for name in upstream_only:
            logger.warning(
                f"  {name}: {CUSTOM_HEADERS[name]}  [upstream only - overriding this "
                f"on responses would break SSE streaming]"
            )
    else:
        logger.debug("No CUSTOM_HEADER_* environment variables found.")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    path = request.url.path
    logger.debug(f"Request: {method} {path}")

    response = await call_next(request)

    # Inject custom headers from environment variables.
    #
    # Protocol headers are never overridden. CUSTOM_HEADER_CONTENT_TYPE is meant
    # for the *upstream* request (where application/json is correct), but applying
    # it to the response replaces text/event-stream on streamed replies, so the
    # client stops treating the body as SSE, waits for a complete JSON document
    # that never comes, and silently ends the turn.
    for header_name, header_value in CUSTOM_HEADERS.items():
        if header_name.lower() in RESPONSE_PROTECTED_HEADERS:
            continue
        response.headers[header_name] = header_value

    return response


# Not using validation function as we're using the environment API key


# NOTE: parse_tool_result_content() was removed here. Tool results are now
# normalized by _tool_result_to_text(), which is used by the real tool-message
# conversion path rather than the old stringify-into-prose path.



def _block_get(block, key, default=None):
    """Read a field off either a pydantic block model or a plain dict."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _sanitize_json_schema_for_openai(schema: Any) -> Any:
    """Strip JSON-Schema keywords that OpenAI-compatible backends reject.

    Anthropic tool schemas are full JSON Schema. Many OpenAI-compatible servers
    (vLLM, Ollama, Azure, several gateways) hard-fail on $schema/$id or on exotic
    string formats, which shows up as a blanket 400 on every request that carries
    Claude Code's toolset.
    """
    if isinstance(schema, list):
        return [_sanitize_json_schema_for_openai(i) for i in schema]
    if not isinstance(schema, dict):
        return schema

    drop_keys = {"$schema", "$id", "$comment", "additionalItems", "cache_control"}
    allowed_string_formats = {"date-time", "date", "time", "duration", "email", "uuid"}

    out = {}
    for key, value in schema.items():
        if key in drop_keys:
            continue
        if key == "format" and schema.get("type") == "string":
            if value not in allowed_string_formats:
                continue
        if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
            out[key] = {k: _sanitize_json_schema_for_openai(v) for k, v in value.items()}
        elif key in ("items", "additionalProperties"):
            out[key] = _sanitize_json_schema_for_openai(value)
        elif key in ("anyOf", "oneOf", "allOf", "prefixItems") and isinstance(value, list):
            out[key] = [_sanitize_json_schema_for_openai(v) for v in value]
        else:
            out[key] = _sanitize_json_schema_for_openai(value)

    # An object schema with no `properties` is rejected by strict validators.
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def _anthropic_image_to_openai(source: Any) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic image source into an OpenAI image_url part.

    The old code passed Anthropic's {"type":"image","source":{...}} through
    untouched and then replaced it with the literal string
    '[Image content - not displayed in text format]', so screenshots and pasted
    images silently never reached the model.
    """
    if not isinstance(source, dict):
        return None
    src_type = source.get("type")
    if src_type == "base64":
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        if not data:
            return None
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    if src_type == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _normalize_openai_content(parts: List[Dict[str, Any]]) -> Union[str, List[Dict[str, Any]], None]:
    """Collapse to a plain string when there is no multimodal part.

    Plain strings are accepted by every OpenAI-compatible server; the parts array
    is not (older/self-hosted endpoints often reject it), so only use it when an
    image actually needs to be carried.
    """
    if not parts:
        return None
    if all(p.get("type") == "text" for p in parts):
        text = "\n".join(p.get("text", "") for p in parts if p.get("text"))
        return text
    return parts


def _tool_result_to_text(content: Any) -> str:
    """Flatten Anthropic tool_result content into the string OpenAI expects."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            item_type = _block_get(item, "type")
            if item_type == "text":
                chunks.append(_block_get(item, "text", "") or "")
            elif item_type == "image":
                # OpenAI tool messages cannot carry images; note it instead of
                # dropping it silently so the model knows something was returned.
                chunks.append("[image omitted: tool results cannot carry images]")
            elif isinstance(item, str):
                chunks.append(item)
            else:
                try:
                    chunks.append(
                        json.dumps(item if isinstance(item, dict) else str(item))
                    )
                except Exception:
                    chunks.append(str(item))
        return "\n".join(c for c in chunks if c)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    return str(content)


def _system_role_for(model: str) -> Optional[str]:
    """Pick the role name the target model accepts for system instructions."""
    clean = strip_provider_prefix(model).lower()
    # o1-mini / o1-preview accept no system-ish role at all.
    if clean.startswith("o1-mini") or clean.startswith("o1-preview"):
        return None
    if is_reasoning_model(model):
        return "developer"
    return "system"


def convert_anthropic_to_litellm(
    anthropic_request: MessagesRequest, resolved: ResolvedModel
) -> Dict[str, Any]:
    """Convert an Anthropic Messages request into OpenAI/LiteLLM shape.

    The important part is that Anthropic tool_use / tool_result blocks become real
    OpenAI `tool_calls` and `role:"tool"` messages. The previous implementation
    stringified them ("Tool result for toolu_123: ..."), which destroyed the
    tool-calling contract: the backend never saw a structured call/result pair, so
    multi-turn agentic loops degraded into the model re-describing tools in prose.

    `resolved` is the router.yaml entry chosen for this request's tier: it supplies
    the exact upstream model name/provider/credentials, separate from whatever
    Claude model id the client sent.
    """
    target_model = resolved.litellm_model
    is_anthropic_model = resolved.litellm_provider == "anthropic"
    is_gemini_model = resolved.litellm_provider == "gemini"
    messages: List[Dict[str, Any]] = []

    # ---------- system ----------
    system_text = ""
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            system_text = anthropic_request.system
        elif isinstance(anthropic_request.system, list):
            parts = []
            for block in anthropic_request.system:
                if _block_get(block, "type") == "text":
                    parts.append(_block_get(block, "text", "") or "")
            system_text = "\n\n".join(p for p in parts if p)

    system_role = _system_role_for(target_model)
    pending_system_prefix = ""
    if system_text.strip():
        if system_role is None:
            # Fold into the first user turn for models with no system role.
            pending_system_prefix = system_text.strip() + "\n\n"
        else:
            messages.append({"role": system_role, "content": system_text.strip()})

    # ---------- conversation ----------
    # Track which tool_use ids the assistant actually emitted. OpenAI returns a hard
    # 400 for a tool message whose tool_call_id was never announced, and for an
    # assistant tool_calls entry with no matching tool reply.
    announced_tool_ids: set = set()
    satisfied_tool_ids: set = set()

    for msg in anthropic_request.messages:
        role = msg.role if msg.role in ("user", "assistant") else "user"
        content = msg.content

        if isinstance(content, str):
            text = content
            if pending_system_prefix and role == "user":
                text = pending_system_prefix + text
                pending_system_prefix = ""
            messages.append({"role": role, "content": text})
            continue

        blocks = content or []

        if role == "assistant":
            text_parts: List[Dict[str, Any]] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in blocks:
                btype = _block_get(block, "type")
                if btype == "text":
                    t = _block_get(block, "text", "") or ""
                    if t:
                        text_parts.append({"type": "text", "text": t})
                elif btype == "tool_use":
                    tool_id = _block_get(block, "id") or f"call_{uuid.uuid4().hex[:24]}"
                    tool_input = _block_get(block, "input", {})
                    if not isinstance(tool_input, dict):
                        tool_input = {"value": tool_input}
                    announced_tool_ids.add(tool_id)
                    tool_calls.append(
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": _block_get(block, "name", "") or "unknown",
                                "arguments": json.dumps(tool_input, ensure_ascii=False),
                            },
                        }
                    )
                elif btype in ("thinking", "redacted_thinking"):
                    # Reasoning traces are provider-specific and are not replayable
                    # as assistant content on OpenAI; drop them from history.
                    continue

            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            normalized = _normalize_openai_content(text_parts)
            if tool_calls:
                # content may be null alongside tool_calls, and must be when empty.
                assistant_msg["content"] = normalized if normalized else None
                assistant_msg["tool_calls"] = tool_calls
            else:
                assistant_msg["content"] = normalized if normalized else ""
            messages.append(assistant_msg)
            continue

        # ----- user turn -----
        # Tool results must be emitted as their own `tool` messages, immediately
        # after the assistant call and before any new user text.
        tool_messages: List[Dict[str, Any]] = []
        user_parts: List[Dict[str, Any]] = []

        for block in blocks:
            btype = _block_get(block, "type")
            if btype == "tool_result":
                tool_use_id = _block_get(block, "tool_use_id") or ""
                result_text = _tool_result_to_text(_block_get(block, "content", ""))
                if _block_get(block, "is_error"):
                    result_text = f"Error: {result_text}" if result_text else "Error"
                if not result_text:
                    result_text = "(no output)"
                if tool_use_id and tool_use_id in announced_tool_ids:
                    satisfied_tool_ids.add(tool_use_id)
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_use_id,
                            "content": result_text,
                        }
                    )
                else:
                    # Orphaned result (no matching call in this window, e.g. after
                    # client-side history truncation). Demote to user text rather
                    # than letting the backend 400 the whole request.
                    logger.debug(
                        f"Orphan tool_result {tool_use_id!r}; demoting to user text"
                    )
                    user_parts.append(
                        {"type": "text", "text": f"Tool result:\n{result_text}"}
                    )
            elif btype == "text":
                t = _block_get(block, "text", "") or ""
                if t:
                    user_parts.append({"type": "text", "text": t})
            elif btype == "image":
                part = _anthropic_image_to_openai(_block_get(block, "source"))
                if part:
                    user_parts.append(part)
            elif btype == "document":
                src = _block_get(block, "source") or {}
                if isinstance(src, dict) and src.get("type") == "text":
                    user_parts.append(
                        {"type": "text", "text": src.get("data", "") or ""}
                    )
                else:
                    user_parts.append(
                        {"type": "text", "text": "[document omitted: unsupported by backend]"}
                    )

        messages.extend(tool_messages)

        if pending_system_prefix and user_parts:
            user_parts.insert(0, {"type": "text", "text": pending_system_prefix.strip()})
            pending_system_prefix = ""

        normalized_user = _normalize_openai_content(user_parts)
        if normalized_user:
            messages.append({"role": "user", "content": normalized_user})

    # Any tool_call the assistant announced but that has no reply must be answered,
    # or OpenAI rejects the conversation outright.
    messages = _repair_tool_call_pairs(messages)

    # ---------- parameters ----------
    max_tokens = anthropic_request.max_tokens
    if MAX_TOKENS_LIMIT:
        max_tokens = min(max_tokens, MAX_TOKENS_LIMIT)
    elif COMPAT_MODE and not is_anthropic_model:
        # The pre-patch proxy hardcoded this cap; some gateways reject or stall on
        # larger values than the backing model advertises.
        max_tokens = min(max_tokens, 16384)

    litellm_request: Dict[str, Any] = {
        "model": target_model,
        "custom_llm_provider": _effective_custom_llm_provider(resolved),
        "api_key": resolved.providerApiKey,
        "messages": messages,
        "stream": bool(anthropic_request.stream),
        # Anthropic-only fields (e.g. `thinking`) that leak through to a
        # non-Anthropic target must be dropped by litellm rather than sent
        # upstream as unknown params and rejected by the gateway.
        "drop_params": True,
    }
    if resolved.providerBaseUrl:
        litellm_request["api_base"] = resolved.providerBaseUrl

    # Reasoning models require max_completion_tokens; some gateways only know
    # max_tokens. `auto` picks per-model, and the env var forces one.
    if MAX_TOKENS_PARAM == "max_tokens":
        litellm_request["max_tokens"] = max_tokens
    elif MAX_TOKENS_PARAM == "max_completion_tokens":
        litellm_request["max_completion_tokens"] = max_tokens
    elif is_reasoning_model(target_model):
        litellm_request["max_completion_tokens"] = max_tokens
    else:
        litellm_request["max_tokens"] = max_tokens

    # Reasoning models reject temperature / top_p / top_k / stop outright.
    if supports_temperature(target_model):
        if anthropic_request.temperature is not None:
            litellm_request["temperature"] = anthropic_request.temperature
        if anthropic_request.top_p is not None:
            litellm_request["top_p"] = anthropic_request.top_p
        if anthropic_request.top_k is not None and is_gemini_model:
            # top_k is not an OpenAI parameter; only forward where it is real.
            litellm_request["top_k"] = anthropic_request.top_k
        if anthropic_request.stop_sequences:
            # OpenAI caps `stop` at 4 entries.
            litellm_request["stop"] = anthropic_request.stop_sequences[:4]

    # Map Anthropic's thinking budget onto OpenAI's reasoning_effort.
    if anthropic_request.thinking is not None and anthropic_request.thinking.is_enabled:
        if is_anthropic_model:
            thinking_payload = {"type": "enabled"}
            if anthropic_request.thinking.budget_tokens:
                thinking_payload["budget_tokens"] = anthropic_request.thinking.budget_tokens
            litellm_request["thinking"] = thinking_payload
        elif supports_reasoning_effort(resolved):
            budget = anthropic_request.thinking.budget_tokens or 0
            if budget and budget <= 4096:
                litellm_request["reasoning_effort"] = "low"
            elif budget and budget >= 16384:
                litellm_request["reasoning_effort"] = "high"
            else:
                litellm_request["reasoning_effort"] = "medium"

    # Streaming usage is opt-in on OpenAI. Without this the proxy reported
    # output_tokens: 0 for every streamed reply and Claude Code's context meter
    # never moved. Some third-party gateways reject the unknown field, so it can
    # be turned off with DISABLE_STREAM_OPTIONS=true.
    if anthropic_request.stream and not (DISABLE_STREAM_OPTIONS or COMPAT_MODE):
        litellm_request["stream_options"] = {"include_usage": True}

    # metadata.user_id -> `user` (abuse-tracking / caching hints)
    if (
        anthropic_request.metadata
        and isinstance(anthropic_request.metadata, dict)
        and not COMPAT_MODE
    ):
        user_id = anthropic_request.metadata.get("user_id")
        if isinstance(user_id, str) and user_id:
            litellm_request["user"] = user_id[:128]

    if CUSTOM_HEADERS:
        litellm_request["extra_headers"] = dict(CUSTOM_HEADERS)

    # ---------- tools ----------
    if anthropic_request.tools:
        openai_tools = []
        for tool in anthropic_request.tools:
            tool_dict = (
                tool.model_dump(exclude_none=True)
                if hasattr(tool, "model_dump")
                else dict(tool)
            )
            name = tool_dict.get("name")
            if not name:
                continue
            # Anthropic server-side tools have a `type` and no usable schema.
            if tool_dict.get("type") and not tool_dict.get("input_schema"):
                logger.warning(f"Dropping unsupported server-side tool: {name}")
                continue

            input_schema = tool_dict.get("input_schema") or {
                "type": "object",
                "properties": {},
            }
            if is_gemini_model:
                input_schema = clean_gemini_schema(input_schema)
            else:
                input_schema = _sanitize_json_schema_for_openai(input_schema)

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool_dict.get("description", "") or "",
                        "parameters": input_schema,
                    },
                }
            )
        if openai_tools:
            litellm_request["tools"] = openai_tools

    # ---------- tool_choice ----------
    if anthropic_request.tool_choice is not None and litellm_request.get("tools"):
        tc = anthropic_request.tool_choice
        if isinstance(tc, str):
            litellm_request["tool_choice"] = tc
        else:
            choice_type = tc.get("type")
            if choice_type == "auto":
                litellm_request["tool_choice"] = "auto"
            elif choice_type == "any":
                # Anthropic "any" == must call some tool == OpenAI "required".
                # The old code forwarded the literal "any", which OpenAI rejects.
                litellm_request["tool_choice"] = "required"
            elif choice_type == "none":
                litellm_request["tool_choice"] = "none"
            elif choice_type == "tool" and tc.get("name"):
                litellm_request["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc["name"]},
                }
            else:
                litellm_request["tool_choice"] = "auto"

            # Anthropic disables parallel calls via the tool_choice object.
            if tc.get("disable_parallel_tool_use") is True:
                litellm_request["parallel_tool_calls"] = False

    return litellm_request


def _repair_tool_call_pairs(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Guarantee every assistant tool_call is followed by a matching tool message.

    Claude Code truncates history mid-loop and users hit ESC between a tool call
    and its result, both of which leave dangling calls that OpenAI rejects with
    "messages with role 'tool' must be a response to a preceding message with
    'tool_calls'".
    """
    repaired: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        repaired.append(msg)
        tool_calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
        if not tool_calls:
            continue

        # Collect the tool replies that immediately follow this assistant message.
        replied = set()
        j = idx + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            replied.add(messages[j].get("tool_call_id"))
            j += 1

        for call in tool_calls:
            if call.get("id") not in replied:
                logger.debug(f"Synthesizing missing tool result for {call.get('id')}")
                repaired.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": "(tool call was interrupted; no result returned)",
                    }
                )
    return repaired


OPENAI_MESSAGE_KEYS = {"role", "content", "name", "tool_call_id", "tool_calls"}


def _validate_openai_messages(litellm_request: Dict[str, Any]) -> None:
    """Last-mile sanity pass on the outgoing message array (mutates in place).

    Everything structural is already handled by the converter; this only guards
    against the invariants that make OpenAI-compatible servers return a bare 400.
    """
    messages = litellm_request.get("messages") or []
    cleaned: List[Dict[str, Any]] = []

    for msg in messages:
        # Drop any key OpenAI does not accept on a message object.
        for key in list(msg.keys()):
            if key not in OPENAI_MESSAGE_KEYS:
                logger.debug(f"Removing unsupported message field: {key}")
                del msg[key]

        role = msg.get("role")
        content = msg.get("content")

        # Only an assistant message carrying tool_calls may have null content.
        if content is None and not (role == "assistant" and msg.get("tool_calls")):
            msg["content"] = ""

        # An assistant turn with neither text nor tool_calls is meaningless and is
        # rejected by some backends; skip it entirely.
        if role == "assistant" and not msg.get("tool_calls"):
            body = msg.get("content")
            if isinstance(body, str) and not body.strip():
                continue

        # A tool message with no id cannot be matched to a call.
        if role == "tool" and not msg.get("tool_call_id"):
            logger.warning("Dropping tool message with no tool_call_id")
            continue

        cleaned.append(msg)

    # OpenAI requires at least one message.
    if not cleaned:
        cleaned = [{"role": "user", "content": "(empty request)"}]

    # A trailing assistant message with unanswered tool_calls cannot be completed.
    if cleaned[-1].get("role") == "assistant" and cleaned[-1].get("tool_calls"):
        for call in cleaned[-1]["tool_calls"]:
            cleaned.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": "(no result returned)",
                }
            )

    litellm_request["messages"] = cleaned


def _anthropic_msg_id(raw_id: Any) -> str:
    """Anthropic ids start with msg_; some SDK paths assert on that prefix."""
    if isinstance(raw_id, str) and raw_id.startswith("msg_"):
        return raw_id
    suffix = uuid.uuid4().hex[:24]
    if isinstance(raw_id, str) and raw_id:
        suffix = re.sub(r"[^A-Za-z0-9]", "", raw_id)[-24:] or suffix
    return f"msg_{suffix}"


def _extract_usage(usage_info: Any) -> tuple:
    """Return (prompt_tokens, completion_tokens, cache_read_tokens).

    Prompt-cache hits are reported by OpenAI under
    prompt_tokens_details.cached_tokens; forwarding them lets Claude Code show
    real cache savings instead of a flat zero.
    """
    def _get(obj, key, default=0):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    if usage_info is None:
        return 0, 0, 0

    prompt_tokens = _get(usage_info, "prompt_tokens", 0) or 0
    completion_tokens = _get(usage_info, "completion_tokens", 0) or 0

    cached = 0
    details = _get(usage_info, "prompt_tokens_details", None)
    if details is not None:
        cached = _get(details, "cached_tokens", 0) or 0
    if not cached:
        cached = _get(usage_info, "cache_read_input_tokens", 0) or 0

    # Anthropic counts cache reads separately from input_tokens; OpenAI includes
    # them, so subtract to keep the totals consistent.
    if cached and cached <= prompt_tokens:
        prompt_tokens = prompt_tokens - cached

    return prompt_tokens, completion_tokens, cached


def map_finish_reason(finish_reason: Any, has_tool_use: bool = False) -> str:
    """Map an OpenAI finish_reason onto an Anthropic stop_reason.

    `has_tool_use` is not optional in practice. Many OpenAI-compatible backends
    (LiteLLM, Ollama, vLLM, Gemini-via-compat, and OpenAI itself intermittently)
    report finish_reason "stop" even when the response carries tool_calls. Passing
    that through as "end_turn" tells the client the turn is over, so it delivers
    the tool_use block but never executes it -- which Claude Code records as an
    interrupted tool call. If we emitted a tool_use block, the turn is a tool turn,
    whatever the backend called it.
    """
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
        "content_filter": "refusal",
    }
    reason = mapping.get(finish_reason, "end_turn")

    if has_tool_use and reason not in ("max_tokens", "refusal"):
        if reason != "tool_use":
            logger.debug(
                f"Overriding stop_reason {reason!r} -> 'tool_use' "
                f"(backend reported finish_reason={finish_reason!r} with tool calls)"
            )
        return "tool_use"
    return reason


def convert_litellm_to_anthropic(
    litellm_response: Union[Dict[str, Any], Any], original_request: MessagesRequest
) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""

    # Enhanced response extraction with better error handling
    try:
        reasoning_text = None

        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, "choices") and hasattr(litellm_response, "usage"):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = (
                message.content if message and hasattr(message, "content") else ""
            )
            tool_calls = (
                message.tool_calls
                if message and hasattr(message, "tool_calls")
                else None
            )
            # Reasoning models expose their trace here; surface it as a thinking
            # block instead of discarding it.
            reasoning_text = getattr(message, "reasoning_content", None) or getattr(
                message, "reasoning", None
            )
            finish_reason = (
                choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            )
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, "id", f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = (
                    litellm_response
                    if isinstance(litellm_response, dict)
                    else litellm_response.dict()
                )
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__
                try:
                    response_dict = (
                        litellm_response.model_dump()
                        if hasattr(litellm_response, "model_dump")
                        else litellm_response.__dict__
                    )
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, "id", f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, "choices", [{}]),
                        "usage": getattr(litellm_response, "usage", {}),
                    }

            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = (
                choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            )
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = (
                choices[0].get("finish_reason", "stop")
                if choices and len(choices) > 0
                else "stop"
            )
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        if reasoning_text is None and isinstance(message, dict):
            reasoning_text = message.get("reasoning_content") or message.get("reasoning")

        # Create content list for Anthropic format
        content = []

        # Thinking must precede text in an Anthropic content array.
        if _should_emit_thinking(original_request) and reasoning_text:
            content.append(
                {
                    "type": "thinking",
                    "thinking": reasoning_text,
                    "signature": PROXY_THINKING_SIGNATURE,
                }
            )

        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        # Add tool calls if present (tool_use in Anthropic format)
        # For ALL models, not just Claude models - convert tool_calls to tool_use blocks
        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = (
                        getattr(function, "arguments", "{}") if function else "{}"
                    )

                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse tool arguments as JSON: {arguments}"
                        )
                        arguments = {"raw": arguments}

                logger.debug(
                    f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}"
                )

                content.append(
                    {
                        "type": "tool_use",
                        "id": _client_tool_id(tool_id, name),
                        "name": name,
                        "input": arguments,
                    }
                )

        # Get usage information - extract values safely from object or dict
        prompt_tokens, completion_tokens, cache_read_tokens = _extract_usage(usage_info)

        stop_reason = map_finish_reason(
            finish_reason,
            has_tool_use=any(b.get("type") == "tool_use" for b in content),
        )

        # Anthropic reports which stop sequence fired; OpenAI does not, so detect it.
        stop_sequence = None
        if original_request.stop_sequences and content_text:
            for seq in original_request.stop_sequences:
                if seq and content_text.endswith(seq):
                    stop_reason = "stop_sequence"
                    stop_sequence = seq
                    content_text = content_text[: -len(seq)]
                    for blk in content:
                        if blk.get("type") == "text":
                            blk["text"] = content_text
                    break

        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})

        # Echo the model id the client asked for, not the internal mapped name.
        # Claude Code compares this against what it sent.
        response_model = original_request.original_model or original_request.model

        anthropic_response = MessagesResponse(
            id=_anthropic_msg_id(response_id),
            model=response_model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=stop_sequence,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                cache_read_input_tokens=cache_read_tokens,
            ),
        )

        return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        error_message = (
            f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        )
        logger.error(error_message)

        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[
                {
                    "type": "text",
                    "text": f"Error converting response: {str(e)}. Please check server logs.",
                }
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


def _repair_truncated_json(buf: str) -> Optional[str]:
    """Best-effort completion of argument JSON that was cut off mid-value.

    A truncated tool argument is unrunnable, and reporting it as max_tokens halts
    the agent loop entirely. Closing the open strings/brackets at least yields a
    runnable call the model can correct on the next turn.
    """
    if not buf.strip():
        return "{}"
    stack = []
    in_string = False
    escaped = False
    for ch in buf:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()

    candidate = buf
    if escaped:
        candidate = candidate[:-1]
    if in_string:
        candidate += '"'
    # A dangling key with no value ("foo": ) cannot be closed meaningfully.
    stripped = candidate.rstrip()
    if stripped.endswith(":"):
        candidate = stripped + "null"
    elif stripped.endswith(","):
        candidate = stripped[:-1]
    candidate += "".join(reversed(stack))

    try:
        json.loads(candidate)
        return candidate
    except (json.JSONDecodeError, TypeError):
        return None


async def _iter_with_idle_timeout(generator, timeout: float):
    """Yield from an upstream stream, failing loudly if it goes silent.

    A gateway can return 200 with `content-type: text/event-stream` and then never
    send a body chunk. Awaiting that forever leaves the proxy hung with nothing in
    the log after the response headers, which is indistinguishable from a proxy
    bug. Time-boxing each read turns it into a reportable error.
    """
    iterator = generator.__aiter__()
    first = True
    while True:
        try:
            if timeout and timeout > 0:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
            else:
                chunk = await iterator.__anext__()
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Upstream sent no data for {timeout:g}s after "
                + ("the response headers" if first else "the previous chunk")
                + ". The backend accepted the request but is not streaming; check "
                "the gateway, or try DISABLE_STREAM_OPTIONS=true and a lower "
                "MAX_TOKENS_LIMIT."
            )
        if first:
            first = False
            logger.debug("First upstream chunk received")
        yield chunk


# Characters Anthropic tool_use ids never contain. An id outside this set is
# certainly synthesized by the backend rather than a random handle.
_PLAIN_TOOL_ID = re.compile(r"^[A-Za-z0-9_-]+$")

PRESERVE_UPSTREAM_TOOL_IDS = (
    os.environ.get("PRESERVE_UPSTREAM_TOOL_IDS", "false").lower() == "true"
)


def _client_tool_id(upstream_id: Any, tool_name: Any = None) -> str:
    """Return a tool_use id that is unique across the whole conversation.

    Some backends derive the id from the tool name and its position, e.g. Kimi
    emits "Edit:0". Those repeat on every turn that calls the same tool in the
    same slot. Anthropic requires tool_use ids to be unique within a conversation
    and clients key tool results by id, so a repeat collides with an earlier call
    and gets reported as an interrupted tool use.

    Only ids that look name-derived (or contain illegal characters) are rewritten;
    genuinely random handles like `call_x7fa...` are passed through untouched.
    Rewriting is safe either way: the id is only meaningful to the client, and the
    request sent upstream keeps assistant tool_calls and tool messages internally
    consistent.
    """
    if isinstance(upstream_id, str) and upstream_id:
        if PRESERVE_UPSTREAM_TOOL_IDS:
            return upstream_id
        derived = False
        if isinstance(tool_name, str) and len(tool_name) >= 3:
            # e.g. "Edit:0", "Edit_1", "Read.2", "Bash3" -- tool name plus a
            # positional index. Real tool names are >= 3 chars, so a short name
            # cannot accidentally match a random id.
            derived = bool(
                re.fullmatch(
                    re.escape(tool_name) + r"[-_:.]?\d*", upstream_id, re.IGNORECASE
                )
            )
        if _PLAIN_TOOL_ID.match(upstream_id) and not derived:
            return upstream_id
        logger.debug(
            f"Rewriting tool id {upstream_id!r} for tool {tool_name!r}: "
            f"looks derived from the tool name and will repeat across turns"
        )
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _sse(event_type: str, payload: Dict[str, Any]) -> str:
    if DUMP_EVENTS:
        logger.warning(f"SSE-> {event_type}: {json.dumps(payload, ensure_ascii=False)[:600]}")
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class _BlockTracker:
    """Tracks the open Anthropic content block while translating an OpenAI stream.

    Anthropic requires that a text_delta only ever targets an open `text` block and
    an input_json_delta only ever targets an open `tool_use` block, and that block
    indices match their position in the final content array. The previous
    implementation pinned text to index 0 forever, so any text emitted *after* a
    tool call was dropped on the floor, and interleaved text/tool output produced
    deltas aimed at the wrong block type (which the Anthropic SDK rejects with
    "Content block is not a text block").
    """

    def __init__(self):
        self.index = -1
        self.open_type = None  # "text" | "tool_use" | "thinking"

    def close(self):
        if self.open_type is not None:
            out = ""
            if self.open_type == "thinking":
                # Anthropic closes a thinking block with a signature_delta, and
                # clients that verify block shape drop thinking blocks that never
                # receive one. We cannot mint a real Anthropic signature for a
                # third-party backend, so emit a clearly-marked placeholder: it is
                # only ever consumed by this proxy, which strips thinking blocks
                # back out of conversation history before they reach any backend.
                out += _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": PROXY_THINKING_SIGNATURE,
                        },
                    },
                )
            out += _sse(
                "content_block_stop", {"type": "content_block_stop", "index": self.index}
            )
            self.open_type = None
            return out
        return ""

    def open(self, block_type: str, block: Dict[str, Any]):
        self.index += 1
        self.open_type = block_type
        return _sse(
            "content_block_start",
            {"type": "content_block_start", "index": self.index, "content_block": block},
        )


async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Translate an OpenAI-style stream into Anthropic SSE events."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    response_model = original_request.original_model or original_request.model

    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    stop_reason = "end_turn"
    stop_sequence = None
    accumulated_text = ""

    tracker = _BlockTracker()
    # Per-upstream-tool-call state. Tool calls are buffered in full and emitted as
    # complete blocks once the stream ends, because Anthropic content blocks cannot
    # be reopened once closed.
    tool_states: Dict[Any, Dict[str, Any]] = {}
    emitted_stop = False

    def _args_complete(buf: str) -> bool:
        """True when a tool's accumulated argument JSON is syntactically whole."""
        if not buf.strip():
            return True
        try:
            json.loads(buf)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    def _emit_text(txt: str):
        out = []
        if tracker.open_type != "text":
            closed = tracker.close()
            if closed:
                out.append(closed)
            out.append(tracker.open("text", {"type": "text", "text": ""}))
        out.append(
            _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": tracker.index,
                    "delta": {"type": "text_delta", "text": txt},
                },
            )
        )
        return out

    try:
        # message_start must carry input_tokens; sending 0 here made Claude Code's
        # context/cost accounting read as zero for every streamed turn.
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": response_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_tokens,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": cache_read_tokens,
                        "output_tokens": 0,
                    },
                },
            },
        )
        yield _sse("ping", {"type": "ping"})
        last_emit_at = time.monotonic()
        stream_started_at = last_emit_at
        upstream_chunks = 0
        first_chunk_at = None
        text_delta_count = 0

        async for chunk in _iter_with_idle_timeout(
            response_generator, UPSTREAM_IDLE_TIMEOUT
        ):
            try:
                upstream_chunks += 1
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()

                # While tool arguments accumulate the proxy emits no content at
                # all. On a large Edit/Write that is tens of seconds of dead air,
                # and clients time the connection out mid-generation. Ping to keep
                # it alive; pings are valid anywhere in an Anthropic stream.
                if (
                    STREAM_KEEPALIVE_SECONDS
                    and time.monotonic() - last_emit_at >= STREAM_KEEPALIVE_SECONDS
                ):
                    last_emit_at = time.monotonic()
                    yield _sse("ping", {"type": "ping"})

                # Usage can arrive on any chunk; with stream_options.include_usage
                # it lands on a final chunk that has an empty choices array.
                usage_obj = getattr(chunk, "usage", None)
                if usage_obj is None and isinstance(chunk, dict):
                    usage_obj = chunk.get("usage")
                if usage_obj is not None:
                    p, c, cached = _extract_usage(usage_obj)
                    input_tokens = p or input_tokens
                    output_tokens = c or output_tokens
                    cache_read_tokens = cached or cache_read_tokens

                choices = getattr(chunk, "choices", None)
                if choices is None and isinstance(chunk, dict):
                    choices = chunk.get("choices")
                if not choices:
                    continue

                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is None:
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                if delta is None:
                    delta = getattr(choice, "message", None) or {}

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason is None and isinstance(choice, dict):
                    finish_reason = choice.get("finish_reason")

                # ---- reasoning trace ----
                reasoning = _block_get(delta, "reasoning_content", None) or _block_get(
                    delta, "reasoning", None
                )
                if _should_emit_thinking(original_request) and reasoning:
                    if tracker.open_type != "thinking":
                        closed = tracker.close()
                        if closed:
                            yield closed
                        yield tracker.open(
                            "thinking", {"type": "thinking", "thinking": ""}
                        )
                    yield _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": tracker.index,
                            "delta": {"type": "thinking_delta", "thinking": reasoning},
                        },
                    )

                # ---- text ----
                delta_content = _block_get(delta, "content", None)
                if delta_content:
                    # Tool blocks are no longer opened mid-stream, so text can
                    # always stream immediately without risk of landing on a
                    # tool_use block.
                    for s in _emit_text(delta_content):
                        yield s
                    accumulated_text += delta_content
                    text_delta_count += 1
                    last_emit_at = time.monotonic()

                # ---- tool calls ----
                # Accumulate only. Anthropic content blocks cannot be reopened once
                # closed, so opening a tool block while its arguments are still
                # arriving means any fragment that shows up after another block has
                # opened (parallel calls, interleaved indices) is unrepresentable
                # and gets dropped -- which silently truncates the JSON on exactly
                # the largest payloads, i.e. file edits. Buffer everything and emit
                # complete, well-formed tool_use blocks at the end of the stream.
                delta_tool_calls = _block_get(delta, "tool_calls", None)

                # Some gateways still emit the pre-2023 `function_call` shape
                # instead of `tool_calls`. Ignoring it means the tool call is
                # invisible to us, the turn looks like an empty reply, and the
                # agent loop halts with no error anywhere.
                if not delta_tool_calls:
                    legacy = _block_get(delta, "function_call", None)
                    if legacy:
                        delta_tool_calls = [
                            {
                                "index": 0,
                                "id": _block_get(legacy, "id", None),
                                "function": {
                                    "name": _block_get(legacy, "name", None),
                                    "arguments": _block_get(legacy, "arguments", None),
                                },
                            }
                        ]
                        logger.debug("Translated legacy function_call delta")
                if delta_tool_calls:
                    if not isinstance(delta_tool_calls, list):
                        delta_tool_calls = [delta_tool_calls]

                    for tool_call in delta_tool_calls:
                        upstream_index = _block_get(tool_call, "index", 0)
                        if upstream_index is None:
                            upstream_index = 0
                        function = _block_get(tool_call, "function", None)
                        name = _block_get(function, "name", None) if function else None
                        arguments = (
                            _block_get(function, "arguments", None) if function else None
                        )
                        call_id = _block_get(tool_call, "id", None)

                        st = tool_states.setdefault(
                            upstream_index,
                            {"id": None, "name": None, "args": "", "order": len(tool_states)},
                        )
                        if call_id and not st["id"]:
                            st["id"] = call_id
                        # First non-empty name wins; some backends send the name in
                        # a later chunk than the one that opens the call.
                        if name and not st["name"]:
                            st["name"] = name
                        if arguments:
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            st["args"] += arguments

                # ---- completion ----
                if finish_reason and not emitted_stop:
                    emitted_stop = True
                    stop_reason = map_finish_reason(
                        finish_reason, has_tool_use=bool(tool_states)
                    )

                    if original_request.stop_sequences and accumulated_text:
                        for seq in original_request.stop_sequences:
                            if seq and accumulated_text.endswith(seq):
                                stop_reason = "stop_sequence"
                                stop_sequence = seq
                                break

                    closed = tracker.close()
                    if closed:
                        yield closed
                    # Do not return here: the usage-bearing chunk from
                    # stream_options.include_usage arrives *after* finish_reason.

            except Exception as chunk_error:
                logger.error(f"Error processing chunk: {chunk_error}")
                continue

        if DUMP_EVENTS:
            now = time.monotonic()
            ttfc = (first_chunk_at - stream_started_at) if first_chunk_at else -1
            reasoning_only = upstream_chunks - text_delta_count
            logger.warning(
                "STREAM-PROFILE: upstream_chunks=%d text_deltas=%d "
                "non_text_chunks=%d tool_calls=%d "
                "time_to_first_chunk=%.2fs total=%.2fs\n"
                "  If text_deltas is ~1, the backend sent the whole answer in one "
                "chunk (upstream buffering).\n"
                "  If non_text_chunks is large and text_deltas is small, the time "
                "went into reasoning that is being dropped -- set "
                "EMIT_REASONING=always to stream it.\n"
                "  Tool call arguments are buffered by design and always appear at "
                "once.",
                upstream_chunks, text_delta_count, reasoning_only,
                len(tool_states), ttfc, now - stream_started_at,
            )

        # An assistant turn with no text and no tool calls ends the agent loop with
        # nothing shown. That is almost always a backend problem (context window
        # exceeded, upstream filter, quota, an unrecognised response shape), but as
        # a well-formed empty message it is indistinguishable from the model simply
        # choosing to stop. Surface it.
        produced_nothing = not accumulated_text.strip() and not tool_states
        if produced_nothing:
            logger.warning(
                "EMPTY COMPLETION: backend returned no text and no tool calls "
                "(upstream_chunks=%d, finish_reason=%s, input_tokens=%s). "
                "Common causes: prompt exceeded the backend's context window, "
                "upstream content filter, or an unrecognised response shape. "
                "Run with DUMP_EVENTS=true to see the raw stream.",
                upstream_chunks, stop_reason, input_tokens or "unknown",
            )
            if ERROR_ON_EMPTY_RESPONSE:
                yield _sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": (
                                "Backend returned an empty completion (no text, no "
                                "tool calls) after "
                                f"{upstream_chunks} chunks. This usually means the "
                                "prompt exceeded the backend's context window. Set "
                                "ERROR_ON_EMPTY_RESPONSE=false to suppress."
                            ),
                        },
                    },
                )
                yield _sse("message_stop", {"type": "message_stop"})
                yield "data: [DONE]\n\n"
                return

        # Close whatever text block is still open.
        closed = tracker.close()
        if closed:
            yield closed

        # Emit every buffered tool call as a complete, well-formed block, in the
        # order the backend introduced them.
        for key in sorted(tool_states, key=lambda k: tool_states[k]["order"]):
            st = tool_states[key]
            if not (st["id"] or st["name"] or st["args"]):
                continue
            args = st["args"]
            if not _args_complete(args):
                repaired = _repair_truncated_json(args)
                logger.warning(
                    f"Tool {st['name']!r} arguments were not valid JSON "
                    f"({len(args)} chars); "
                    + ("repaired" if repaired is not None else "sending as-is")
                )
                if repaired is not None:
                    args = repaired
            yield tracker.open(
                "tool_use",
                {
                    "type": "tool_use",
                    "id": _client_tool_id(st["id"], st["name"]),
                    "name": st["name"] or "unknown",
                    "input": {},
                },
            )
            if args:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": tracker.index,
                        "delta": {"type": "input_json_delta", "partial_json": args},
                    },
                )
            closed = tracker.close()
            if closed:
                yield closed

        # Anthropic requires at least one content block in the message.
        if tracker.index == -1:
            yield tracker.open("text", {"type": "text", "text": ""})
            closed = tracker.close()
            if closed:
                yield closed

        # Some backends end the stream without ever sending a finish_reason. If we
        # emitted tool calls, the turn is still a tool turn.
        if tool_states and stop_reason == "end_turn":
            stop_reason = "tool_use"

        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
                "usage": {
                    "input_tokens": input_tokens,
                    "cache_read_input_tokens": cache_read_tokens,
                    "output_tokens": output_tokens,
                },
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback

        logger.error(f"Error in streaming: {e}\n{traceback.format_exc()}")

        # Close whatever is open so the client's parser is not left mid-block.
        try:
            closed = tracker.close()
            if closed:
                yield closed
        except Exception:
            pass

        # Anthropic signals mid-stream failure with an `error` event; the previous
        # code sent stop_reason:"error", which is not a valid Anthropic stop_reason
        # and made SDKs report a malformed response instead of the real cause.
        yield _sse(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    try:
        # print the body here
        body = await raw_request.body()

        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        resolved = getModel(request.model_tier)
        logger.debug(
            f"📌 MODEL ROUTING: tier={request.model_tier!r} -> id={resolved.id!r} "
            f"(provider={resolved.litellm_provider}, upstream={resolved.providerModelName!r})"
        )
        resolved_display = resolved.displayName or resolved.id

        logger.debug(
            f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}"
        )

        # Convert Anthropic request to LiteLLM format. This also fills in the
        # target model/provider/api_key/api_base from `resolved`.
        litellm_request = convert_anthropic_to_litellm(request, resolved)

        # NOTE: the legacy "flatten every content block into a string" pass that
        # used to live here has been removed. convert_anthropic_to_litellm() now
        # emits proper OpenAI messages (tool_calls / role:"tool" / image_url), and
        # re-flattening them here would have thrown that structure away again.
        _validate_openai_messages(litellm_request)

        if DUMP_EVENTS:
            _dump = {
                k: v for k, v in litellm_request.items()
                if k not in ("api_key", "messages", "tools", "extra_headers")
            }
            _dump["message_roles"] = [m.get("role") for m in litellm_request["messages"]]
            _dump["tool_names"] = [
                t["function"]["name"] for t in litellm_request.get("tools", [])
            ]
            logger.warning(f"UPSTREAM-> {json.dumps(_dump, ensure_ascii=False)[:1500]}")

        # Only log basic info about the request, not the full details
        logger.debug(
            f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}"
        )

        litellm_request["messages"] = apply_content_replacements(litellm_request["messages"])

        # Handle streaming mode
        if request.stream:
            # Use LiteLLM for streaming
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                resolved_display,
                len(litellm_request["messages"]),
                num_tools,
                None,  # in-flight; real status is logged on failure
            )
            # Ensure we use the async version for streaming
            response_generator = await litellm.acompletion(**litellm_request)

            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                resolved_display,
                len(litellm_request["messages"]),
                num_tools,
                None,  # in-flight; real status is logged on failure
            )
            start_time = time.time()
            # acompletion, not completion: the sync client blocks the event loop for
            # the whole generation, so one slow request stalled every other client.
            litellm_response = await litellm.acompletion(**litellm_request)
            logger.debug(
                f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s"
            )

            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

            return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()

        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback,
        }

        # Check for LiteLLM-specific attributes
        for attr in ["message", "status_code", "response", "llm_provider", "model"]:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        # Check for additional exception details in dictionaries
        if hasattr(e, "__dict__"):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ["args", "__traceback__"]:
                    error_details[key] = str(value)

        # Helper function to safely serialize objects for JSON
        def sanitize_for_json(obj):
            """递归地清理对象使其可以JSON序列化"""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, "__dict__"):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, "text"):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        # Log all error details with safe serialization
        sanitized_details = sanitize_for_json(error_details)
        logger.error(
            f"Error processing request: {json.dumps(sanitized_details, indent=2)}"
        )

        # Format error for response
        error_message = str(e)
        if error_details.get("message"):
            error_message = str(error_details["message"])

        status_code = error_details.get("status_code", 500)
        try:
            status_code = int(status_code)
        except (TypeError, ValueError):
            status_code = 500

        try:
            error_resolved = getModel(request.model_tier)
            error_resolved_display = error_resolved.displayName or error_resolved.id
        except Exception:
            error_resolved_display = request.model

        log_request_beautifully(
            "POST",
            raw_request.url.path,
            request.original_model or request.model,
            error_resolved_display,
            len(request.messages),
            len(request.tools) if request.tools else 0,
            status_code,
        )

        # Anthropic clients parse {"type":"error","error":{...}}. FastAPI's default
        # {"detail": ...} envelope made every upstream failure surface in Claude Code
        # as an opaque "API Error" with no message.
        return JSONResponse(
            status_code=status_code,
            content={
                "type": "error",
                "error": {
                    "type": _anthropic_error_type(status_code),
                    "message": error_message,
                },
            },
        )


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model

        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        resolved = getModel(request.model_tier)
        resolved_display = resolved.displayName or resolved.id

        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            ),
            resolved,
        )

        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter

            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                resolved_display,
                len(converted_request["messages"]),
                num_tools,
                None,  # in-flight; real status is logged on failure
            )

            # litellm.token_counter() takes no api_base/api_key/custom_llm_provider
            # in this version -- model is the only routing hint it accepts, and it
            # picks a tokenizer from that alone (falling back to a generic one for
            # unrecognized ids, which is the best available for custom gateways).
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )

            # Tool definitions are part of the billed prompt but token_counter
            # ignores them. Claude Code uses this endpoint to decide when to
            # compact, so undercounting the whole toolset by thousands of tokens
            # made it compact far too late and overflow the real context window.
            tools = converted_request.get("tools") or []
            if tools:
                try:
                    tools_text = json.dumps(tools, ensure_ascii=False)
                    token_count += token_counter(
                        model=converted_request["model"],
                        text=tools_text,
                    )
                except Exception as tool_err:
                    logger.debug(f"Tool token estimate failed: {tool_err}")
                    token_count += len(json.dumps(tools)) // 4

            return TokenCountResponse(input_tokens=token_count)

        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # ~4 chars per token is far closer than a flat 1000, which reported the
            # same count for a one-line prompt and a 100k-token conversation.
            approx = len(json.dumps(converted_request.get("messages", []))) // 4
            return TokenCountResponse(input_tokens=max(approx, 1))

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


def _anthropic_error_type(status_code: int) -> str:
    """Map an HTTP status onto Anthropic's error `type` vocabulary."""
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        413: "request_too_large",
        422: "invalid_request_error",
        429: "rate_limit_error",
        500: "api_error",
        529: "overloaded_error",
    }.get(status_code, "api_error")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Report malformed requests in Anthropic's error envelope.

    A 422 with FastAPI's default body is unparseable to the Anthropic SDK, so a
    schema mismatch used to look like a network fault to the client.
    """
    logger.warning(f"Request validation failed: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": f"Invalid request: {exc.errors()}",
            },
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "error": {
                "type": _anthropic_error_type(exc.status_code),
                "message": str(exc.detail),
            },
        },
    )


@app.get("/health")
async def health():
    """Liveness probe plus the effective router.yaml mapping, for debugging setups."""
    resolved = getModel(None)
    base_url_display = None
    if resolved.providerBaseUrl:
        base_url_display = urlparse(resolved.providerBaseUrl).netloc or resolved.providerBaseUrl
    return {
        "status": "ok",
        "preferredModel": resolved.id,
        "endpointType": resolved.endpointType,
        "providerBaseUrl": base_url_display,
        "providerModelName": resolved.providerModelName,
        "modelCount": len(ROUTER_CONFIG.models),
        "source": "router.yaml",
    }


@app.get("/v1/models")
async def list_models():
    """Model ids configured in router.yaml; clients probe this to discover choices."""
    ids: List[str] = []
    for entry in ROUTER_CONFIG.models:
        if entry.id not in ids:
            ids.append(entry.id)
    return {
        "data": [
            {
                "id": mid,
                "type": "model",
                "display_name": mid,
                "created_at": "2025-01-01T00:00:00Z",
            }
            for mid in ids
        ],
        "has_more": False,
    }


@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}


# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"


def log_request_beautifully(
    method, path, claude_model, resolved_model_display, num_messages, num_tools, status_code
):
    """Log requests in a beautiful, twitter-friendly format showing Claude to upstream mapping.

    `resolved_model_display` must already be the router.yaml displayName/id to show
    (not a raw providerModelName), since providerModelName can itself contain
    slashes (e.g. "minimaxai/minimax-m3") that would be mangled by a naive split.
    """
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    openai_display = f"{Colors.GREEN}{resolved_model_display}{Colors.RESET}"

    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    # Format status code. `None` means the upstream call has not returned yet, so
    # do not claim success: the old code hardcoded 200 before the request was even
    # made, which printed a green "✓ 200 OK" for requests that then failed.
    if status_code is None:
        status_str = f"{Colors.DIM}… in flight{Colors.RESET}"
    elif status_code == 200:
        status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
    else:
        status_str = f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")