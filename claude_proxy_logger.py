"""Claude Code Trace Extractor - captures conversations with latency metrics."""

from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import orjson

from cli_utils import (
    build_launch_command,
    seed_claude_append_system_prompt,
    copy_to_clipboard,
    enforce_cli_version,
    get_cert_path,
    get_cli_version,
    get_platform_info,
    is_port_free,
    zip_subagent_transcripts,
    VERSION,
    MODEL_REQUIRED,
    CONFIG_REQUIRED,
    CLAUDE_DEFAULT_MODEL,
    CLAUDE_DEFAULT_EFFORT_LEVEL,
    CLAUDE_DEFAULT_PERMISSION_MODE,
    CLAUDE_PERMISSION_MODES,
)

def json_loads(s): return orjson.loads(s)
def json_dumps(obj): return orjson.dumps(obj, option=orjson.OPT_INDENT_2)

SAVE_EVERY_TURNS = 5
SAVE_EVERY_SECS = 5.0

# Supported API hosts for Claude/Anthropic
CLAUDE_API_HOSTS = ("anthropic.com", "dataannotation.tech")

# Tools whose tool_result (the user's answer, arriving in a later request) is
# attached to the originating tool_use entry in the JSON trace. Kept to
# interactive question tools: capturing results for all tools would embed
# every Bash/Read output and balloon the trace.
ANSWER_CAPTURE_TOOLS = {"AskUserQuestion"}


def _tool_result_text(content) -> str:
    """Normalize a tool_result's content to text. AskUserQuestion answers
    arrive as a plain string; handle the list-of-blocks form too."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return "" if content is None else str(content)

if TYPE_CHECKING:
    from mitmproxy import http

# Response headers copied verbatim into the trace, matched by exact lowercase
# name or prefix: request ids, rate-limit state, organization and workspace
# ids, the retry headers and the x-cc-*/x-is-* fallback and compaction stamps.
_RESPONSE_HEADER_NAMES = {"request-id", "x-request-id", "x-should-retry", "retry-after"}
_RESPONSE_HEADER_PREFIXES = ("anthropic-", "x-cc-", "x-is-")

# Segments marking a credential-bearing header or body key, matched against the
# "-"/"_"/"."-split (body keys also camelCase-split, lowercased) name, never as
# substrings: max_tokens and anthropic-ratelimit-input-tokens-limit pass while
# x-api-key, authorization, set-cookie and access_token do not.
_SECRET_SEGMENTS = {"auth", "authorization", "key", "apikey", "token", "secret", "cookie", "password",
                    "passphrase", "credential", "credentials", "bearer", "apitoken", "authtoken", "accesstoken",
                    "sessiontoken", "apisecret", "secretkey", "privatekey", "oauth", "jwt", "passcode", "pwd",
                    "passwd", "secrets", "keys", "signature"}
_SEGMENT_SEPARATORS = re.compile(r"[-_.]")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

# Body keys that are large or captured in reduced form (system_prompt,
# user_message, tool_names, client_metadata); the rest lands in request.params
# minus credential-shaped keys, so new API parameters need no parser change.
_REQUEST_BODY_BULK_KEYS = ("messages", "system", "tools", "metadata")


def _clean_text(value: str) -> str:
    """Header names and values arrive surrogate-escaped; make them JSON-serialisable."""
    return value.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def _secret_segments(name: str) -> bool:
    return any(segment in _SECRET_SEGMENTS for segment in _SEGMENT_SEPARATORS.split(name))


def _pick_headers(headers, names, prefixes) -> dict:
    """Copy headers whose lowercase name is in `names` or starts with one of
    `prefixes`, never one that looks like a credential. mitmproxy folds
    repeated names into one value."""
    picked = {}
    for name, value in headers.items():
        lname = _clean_text(name.lower())
        if _secret_segments(lname):
            continue
        if lname in names or lname.startswith(prefixes):
            picked[lname] = _clean_text(value)
    return picked


def _int_or_zero(value) -> int:
    """Integer fields (request params, usage counters) the wire may carry as null."""
    return value if isinstance(value, int) else 0


def _int_header(headers, name: str) -> int:
    """Integer header value, tolerating a folded duplicate ("12, 34" -> 12)
    and non-numeric junk (-> 0)."""
    value = (headers.get(name, "") or "").split(",")[0].strip()
    return int(value) if value.isdecimal() else 0


def _secret_body_key(key) -> bool:
    return _secret_segments(_CAMEL_BOUNDARY.sub("_", str(key)).lower())


def _scrub_params(value):
    """Recursively drop credential-shaped keys from a request-body fragment."""
    if isinstance(value, dict):
        return {k: _scrub_params(v) for k, v in value.items() if not _secret_body_key(k)}
    if isinstance(value, list):
        return [_scrub_params(v) for v in value]
    return value


def _bump(counter: dict, key) -> None:
    """Count `key`, coerced to str so the trace stays JSON-serialisable."""
    key = key if isinstance(key, str) else repr(key)
    counter[key] = counter.get(key, 0) + 1


def _parse_client_metadata(metadata) -> dict:
    """Claude Code sends metadata.user_id as a JSON string holding device_id,
    account_uuid and session_id. Return it decoded (credential-shaped keys
    dropped); fall back to the raw string under "user_id" if it is not a JSON
    object."""
    if not isinstance(metadata, dict):
        return {}
    user_id = metadata.get("user_id")
    if isinstance(user_id, str):
        try:
            decoded = json_loads(user_id)
        except Exception:
            decoded = None
        if isinstance(decoded, dict):
            return _scrub_params(decoded)
        return {"user_id": user_id}
    return {}


def _count_cache_control(tools, system, messages) -> dict:
    """Count cache_control breakpoints on tool definitions, system blocks and
    message content blocks, keyed "ttl" or "ttl:scope" (e.g. "1h:global"); a
    breakpoint without an explicit ttl counts under the API default "5m"."""
    counts: dict = {}
    blocks = list(tools) + (list(system) if isinstance(system, list) else [])
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            blocks.extend(content)
    for block in blocks:
        cc = block.get("cache_control") if isinstance(block, dict) else None
        if isinstance(cc, dict):
            key = str(cc.get("ttl") or "5m")
            if cc.get("scope"):
                key = f"{key}:{cc['scope']}"
            _bump(counts, key)
    return counts


def _summarize_transformations(transformations) -> Optional[dict]:
    """Collapse message.input_transformations (one entry per affected block,
    so it grows with history) into counts keyed "type:reason"."""
    if not isinstance(transformations, list):
        return None
    counts: dict = {}
    for entry in transformations:
        if isinstance(entry, dict):
            key = entry.get("type", "")
            if entry.get("reason"):
                key = f"{key}:{entry['reason']}"
            _bump(counts, key)
        else:
            _bump(counts, entry)
    return counts


def _api_error(obj) -> dict:
    """Normalise an API `error` object; a missing one becomes {} and any other
    non-dict payload {message}."""
    if isinstance(obj, dict):
        return obj
    return {} if obj is None else {"message": str(obj)}


def _error_text(error: dict) -> str:
    kind = error.get("type") or "error"
    message = error.get("message")
    return f"{kind}: {message}" if message else kind


def _parse_error_body(content: bytes) -> Optional[dict]:
    """Return the `error` object from an API error body
    ({"type":"error","error":{...}}), or None if the body is not one."""
    try:
        data = json_loads(content)
    except Exception:
        return None
    if isinstance(data, dict) and data.get("type") == "error":
        return _api_error(data.get("error"))
    return None

# mitmproxy's own content decoder (gzip/deflate/br/zstd), used to decode streamed
# SSE bodies. They arrive still-encoded via flow.response.stream, unlike buffered
# flow.response.content (which mitmproxy decodes automatically). Guarded so the
# module imports even without mitmproxy, matching the addon import below.
try:
    from mitmproxy.net import encoding as _mitm_encoding
except Exception:
    _mitm_encoding = None


def _iter_sse_data(text: str):
    # SSE recognises only CR, LF and CRLF; str.splitlines() would also break
    # on U+2028/U+2029/NEL, which JSON may carry raw inside strings.
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    data_parts = []
    for line in text.split("\n"):
        if line == "":
            if data_parts:
                yield "\n".join(data_parts)
                data_parts = []
            continue
        if line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if data_parts:
        yield "\n".join(data_parts)


def _decode_sse_body(raw: bytes, content_encoding: str) -> bytes:
    """Decode a streamed (possibly truncated) SSE body by its Content-Encoding,
    using mitmproxy's own decoder. Returns the raw bytes for identity/unknown
    encodings or on any decode failure, so the turn is still recorded (the SSE
    parser simply finds no events in undecodable bytes)."""
    if not raw:
        return b""
    enc = (content_encoding or "").lower().strip()
    if enc in ("", "identity") or _mitm_encoding is None:
        return bytes(raw)
    try:
        out = _mitm_encoding.decode(bytes(raw), enc)
        return out if isinstance(out, (bytes, bytearray)) else bytes(raw)
    except Exception:
        return bytes(raw)


class TurnMetrics:
    def __init__(self):
        self.turn_id = 0
        self.timestamp = ""
        self.model = ""
        self.request_id = ""
        self.message_id = ""

        self.messages_count = 0
        self.user_message = ""
        self.system_prompt = ""
        # role:system messages inside the messages array (mid-conversation system).
        self.mid_conversation_system = []
        self.request_size_bytes = 0
        self.max_tokens = 0
        self.thinking_budget_tokens = 0
        # output_config.effort from the request body ("" when absent, e.g.
        # utility calls). With adaptive thinking this is the effort knob;
        # thinking_budget_tokens stays 0 on adaptive requests.
        self.effort = ""
        # thinking.type from the request body ("adaptive", "enabled",
        # "disabled", or "" when absent).
        self.thinking_type = ""
        self.tools_count = 0
        # thinking.display from the request body: "" (Claude Code's interactive
        # default), "omitted" (print mode), "summarized" (--show-thinking-summaries),
        # "updates" (first-party endpoint, forced by _CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL=1).
        self.thinking_display = ""
        # anthropic-beta request header, split into individual flags.
        self.betas = []
        # x-claude-code-request-class header ("main", subagent classes, ...).
        self.request_class = ""
        # Decoded metadata.user_id (device_id, account_uuid, session_id).
        self.client_metadata = {}
        self.tool_names = []
        # cache_control breakpoints by "ttl[:scope]" across tools, system and messages.
        self.cache_control_counts = {}
        # Request body minus _REQUEST_BODY_BULK_KEYS and credential-shaped keys.
        self.request_params = {}

        self.request_start = 0.0
        self.first_byte = None
        self.stream_end = None
        self.server_time_ms = 0

        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.cache_creation_5m_tokens = 0
        self.cache_creation_1h_tokens = 0

        self.ratelimit_input_tokens_limit = 0
        self.ratelimit_input_tokens_remaining = 0
        self.ratelimit_requests_limit = 0
        self.ratelimit_requests_remaining = 0

        self.thinking_parts = []
        self.text_parts = []
        self.tool_uses = []
        self.stop_reason = ""
        self.response_size_bytes = 0
        self.service_tier = ""
        self.context_mgmt_edits = []

        # HTTP status and the allowlisted response headers.
        self.status_code = None
        self.response_headers = {}
        # Model that served the response (the fallback block's target on a server-side fallback).
        self.response_model = ""
        # message_start.message minus content/usage/input_transformations, overlaid with the final message_delta.
        self.message_meta = {}
        # message.input_transformations as counts per "type:reason".
        self.input_transformations = None
        # message_start.usage overlaid with message_delta.usage, verbatim.
        self.usage = {}
        # usage.output_tokens_details.thinking_tokens (None when not reported).
        self.thinking_tokens = None
        # fallback content blocks ({from, to}), one per server-side hop.
        self.fallbacks = []
        # Per-type tallies of SSE events, blocks and deltas, so unhandled types stay visible.
        self.event_types = {}
        self.block_types = {}
        self.delta_types = {}
        # error SSE events and API error bodies, verbatim.
        self.api_errors = []
        # SSE payloads that failed to parse: count and first error text.
        self.parse_errors = 0
        self.parse_error_sample = None

        self.total_events = 0
        self.first_thinking_idx = None
        self.first_text_idx = None

        # Streaming-passthrough state. stream_buf accumulates the still-encoded
        # response body forwarded via flow.response.stream. aborted marks a
        # failed turn (ended mid-flight, API error, HTTP status >= 400 or no
        # stop_reason): recorded so failures stay visible, kept out of averages.
        self.stream_buf = None
        self.content_encoding = ""
        self.was_streamed = False
        self.aborted = False
        self.error_message = ""

    def _estimate_output(self, thinking_text: str, response_text: str) -> dict:
        """Thinking/text/tool token breakdown. The thinking share is the API's
        own count when it reported one (thinking_source "api"), or that count
        replaced ("api_adjusted") when it had to be clamped to the output
        count or raised to it because nothing visible was produced, else a
        character-proportion estimate ("estimate"). The remainder is split
        between text and tool input by character count; the three parts always
        sum to the output count."""
        empty = {"thinking": None, "thinking_source": None,
                 "non_thinking": {"total": None, "text": None, "tool": None}}
        if self.output_tokens <= 0:
            return empty

        thinking_chars = len(thinking_text)
        text_chars = len(response_text)
        tool_chars = 0
        for tool in self.tool_uses:
            tool_chars += len(tool.get("name", ""))
            tool_input = tool.get("input", "")
            tool_chars += len(tool_input) if isinstance(tool_input, str) else len(orjson.dumps(tool_input))
        visible_chars = text_chars + tool_chars
        total_content_chars = thinking_chars + visible_chars

        if self.thinking_tokens is not None:
            if visible_chars == 0:
                thinking = self.output_tokens
            else:
                thinking = min(self.thinking_tokens, self.output_tokens)
            source = "api" if thinking == self.thinking_tokens else "api_adjusted"
        elif total_content_chars > 0:
            thinking = int(self.output_tokens * thinking_chars / total_content_chars)
            source = "estimate"
        else:
            return empty

        non_thinking = self.output_tokens - thinking
        text = int(non_thinking * text_chars / visible_chars) if visible_chars else 0
        return {
            "thinking": thinking,
            "thinking_source": source,
            "non_thinking": {"total": non_thinking, "text": text, "tool": non_thinking - text},
        }

    def to_dict(self) -> dict:
        fb, rs, se = self.first_byte, self.request_start, self.stream_end

        ttfb = round((fb - rs) * 1000, 1) if fb and rs else None
        ttlt = round((se - rs) * 1000, 1) if se and rs else None
        stream_dur = (se - fb) if se and fb else 0

        ttf_thinking = ttf_answer = None
        if self.total_events > 0 and ttfb:
            if self.first_thinking_idx is not None:
                ttf_thinking = round(ttfb + (stream_dur * self.first_thinking_idx / self.total_events * 1000), 1)
            if self.first_text_idx is not None:
                ttf_answer = round(ttfb + (stream_dur * self.first_text_idx / self.total_events * 1000), 1)

        otps = round(self.output_tokens / (ttlt / 1000), 1) if ttlt and ttlt > 0 and self.output_tokens > 0 else None
        total_input = self.input_tokens

        thinking_text = "".join(self.thinking_parts)
        response_text = "".join(self.text_parts)

        estimated_output = self._estimate_output(thinking_text, response_text)

        meta = self.message_meta
        diagnostics = meta.get("diagnostics")

        return {
            "turn": self.turn_id,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "message_id": self.message_id,
            "model": self.model,
            "service_tier": self.service_tier,
            "request": {
                "messages_count": self.messages_count,
                "user_message": self.user_message,
                "system_prompt": self.system_prompt,
                "mid_conversation_system": self.mid_conversation_system,
                "size_bytes": self.request_size_bytes,
                "max_tokens": self.max_tokens,
                "thinking_budget_tokens": self.thinking_budget_tokens,
                "effort": self.effort,
                "thinking_type": self.thinking_type,
                "thinking_display": self.thinking_display,
                "tools_count": self.tools_count,
                "tool_names": self.tool_names,
                "cache_control_counts": self.cache_control_counts,
                "betas": self.betas,
                "request_class": self.request_class,
                "client_metadata": self.client_metadata,
                "params": self.request_params,
            },
            "response": {
                "model": self.response_model,
                "status_code": self.status_code,
                "thinking": thinking_text,
                "text": response_text,
                "tool_uses": self.tool_uses,
                "stop_reason": self.stop_reason,
                "stop_details": meta.get("stop_details"),
                "fallbacks": self.fallbacks,
                "input_transformations": self.input_transformations,
                "cache_miss_reason": diagnostics.get("cache_miss_reason") if isinstance(diagnostics, dict) else None,
                "size_bytes": self.response_size_bytes,
                "context_mgmt_edits": self.context_mgmt_edits,
                "event_types": self.event_types,
                "block_types": self.block_types,
                "delta_types": self.delta_types,
                "api_errors": self.api_errors,
                "parse_errors": self.parse_errors,
                "parse_error_sample": self.parse_error_sample,
                "meta": meta,
                "headers": self.response_headers,
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "thinking": self.thinking_tokens,
                "estimated_output": estimated_output,
                "cache_creation": self.cache_creation_input_tokens,
                "cache_read": self.cache_read_input_tokens,
                "cache_creation_5m": self.cache_creation_5m_tokens,
                "cache_creation_1h": self.cache_creation_1h_tokens,
                "total_input": total_input,
                "iterations": self.usage.get("iterations"),
                "usage": self.usage,
            },
            "latency": {
                "server_time_ms": self.server_time_ms,
                "ttfb_ms": ttfb,
                "ttf_thinking_ms": ttf_thinking,
                "ttf_answer_ms": ttf_answer,
                "ttlt_ms": ttlt,
            },
            "tool_call_count": len(self.tool_uses),
            "aborted": self.aborted,
            "error": self.error_message or None,
            "throughput": {"output_tokens_per_sec": otps},
            "rate_limits": {
                "input_tokens_limit": self.ratelimit_input_tokens_limit,
                "input_tokens_remaining": self.ratelimit_input_tokens_remaining,
                "requests_limit": self.ratelimit_requests_limit,
                "requests_remaining": self.ratelimit_requests_remaining,
            },
            "events_count": self.total_events,
        }



class ConversationLog:
    __slots__ = ('log_dir', 'run_id', 'session_id', 'session_start', 'log_file', 'turns', 'turn_counter',
                 'total_input_tokens', 'total_output_tokens', 'total_cache_read',
                 'total_cache_creation', 'total_latency_ms', 'models_used',
                 '_sum_server', '_sum_ttfb', '_sum_ttlt', '_sum_otps',
                 '_sum_ttf_thinking', '_sum_ttf_answer',
                 '_cnt_server', '_cnt_ttfb', '_cnt_ttlt', '_cnt_otps',
                 '_cnt_ttf_thinking', '_cnt_ttf_answer',
                 '_last_save_ts', '_last_save_turn', 'total_tool_calls',
                 'total_estimated_thinking_tokens', 'total_estimated_text_tokens',
                 'total_estimated_tool_tokens', 'claude_code_metrics',
                 'cli_version', 'claude_config_dir', 'thinking_disabled', 'aborted_turns',
                 'pending_answers', 'models_responded', 'fallback_turns', 'thinking_source_counts')

    def __init__(self, log_dir: Path, run_id: str = "", claude_config_dir: str = "", thinking_disabled: bool = False):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.session_id: Optional[str] = None
        self.session_start: Optional[str] = None
        self.log_file: Optional[Path] = None
        self.turns: list[dict] = []
        self.turn_counter = 0

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_read = 0
        self.total_cache_creation = 0
        self.total_latency_ms = 0.0
        self.models_used: set = set()
        # Models that served responses; differs from models_used on a server-side fallback.
        self.models_responded: set = set()
        self.fallback_turns = 0
        # Turns whose thinking share came from the API vs an estimate.
        self.thinking_source_counts: dict = {"api": 0, "api_adjusted": 0, "estimate": 0}

        self._sum_server = self._sum_ttfb = self._sum_ttlt = self._sum_otps = 0.0
        self._sum_ttf_thinking = self._sum_ttf_answer = 0.0
        self._cnt_server = self._cnt_ttfb = self._cnt_ttlt = self._cnt_otps = 0
        self._cnt_ttf_thinking = self._cnt_ttf_answer = 0
        self._last_save_ts = 0.0
        self._last_save_turn = 0
        self.total_tool_calls = 0
        self.aborted_turns = 0
        self.total_estimated_thinking_tokens = 0
        self.total_estimated_text_tokens = 0
        self.total_estimated_tool_tokens = 0
        self.claude_code_metrics: Optional[dict] = None
        # None when claude is not on PATH or `claude --version` is unparseable.
        self.cli_version: Optional[str] = get_cli_version("claude")
        # CLAUDE_CONFIG_DIR set on the launch command for this run. None when
        # --no-claude-config was set or the option was not passed.
        self.claude_config_dir: Optional[str] = claude_config_dir or None
        # Whether the extractor launched Claude Code with --disable-thinking
        # (alwaysThinkingEnabled:false + MAX_THINKING_TOKENS=0). This records the
        # LAUNCH CONFIGURATION the extractor applied, not a wire-level confirmation
        # that the API disabled thinking. Sourced from the --disable-thinking flag.
        self.thinking_disabled: bool = thinking_disabled
        # Outstanding ANSWER_CAPTURE_TOOLS tool_uses (e.g. AskUserQuestion)
        # whose answer hasn't been seen yet: tool_use_id -> the live tool_use
        # dict inside self.turns, so filling the answer mutates the recorded
        # turn in place. Doubles as the watch flag: requests are only scanned
        # for tool_results while this is non-empty.
        self.pending_answers: dict[str, dict] = {}

    def _init_session(self, request_id: str):
        # The session id waits for the first turn that carried a request id.
        if self.log_file is None:
            self.session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / "trace.json"
        if not self.session_id and request_id:
            self.session_id = request_id[4:12] if request_id.startswith("req_") else request_id[:8]

    def add_turn(self, metrics: TurnMetrics) -> tuple[bool, str]:
        """Add a turn and maybe save. Returns (saved, filename) if intermediate save occurred."""
        self._init_session(metrics.request_id)
        self.turn_counter += 1
        metrics.turn_id = self.turn_counter

        turn_data = metrics.to_dict()
        self.turns.append(turn_data)
        self.models_used.add(turn_data["model"])
        if turn_data["response"]["model"]:
            self.models_responded.add(turn_data["response"]["model"])
        if turn_data["response"]["fallbacks"]:
            self.fallback_turns += 1

        # Register question tool_uses so later requests are watched for their
        # tool_result answer. result:None distinguishes "asked, never answered"
        # from traces predating answer capture (key absent). Registered even on
        # aborted turns: a retried question gets a fresh tool_use id, so a
        # stale pending entry just keeps the scan on until shutdown (harmless).
        for tool_use in turn_data["response"]["tool_uses"]:
            if tool_use.get("name") in ANSWER_CAPTURE_TOOLS and tool_use.get("id"):
                tool_use["result"] = None
                self.pending_answers[tool_use["id"]] = tool_use

        # Aborted turns are recorded for visibility but excluded from token/
        # latency/throughput aggregates (their partial counts would skew averages).
        if turn_data.get("aborted"):
            self.aborted_turns += 1
            return self._maybe_save()

        tokens = turn_data["tokens"]
        latency = turn_data["latency"]
        throughput = turn_data["throughput"]

        self.total_input_tokens += tokens["input"]
        self.total_output_tokens += tokens["output"]
        self.total_cache_read += tokens["cache_read"]
        self.total_cache_creation += tokens["cache_creation"]
        self.total_tool_calls += turn_data["tool_call_count"]
        est = tokens.get("estimated_output", {})
        if est.get("thinking") is not None:
            self.total_estimated_thinking_tokens += est["thinking"]
            self.total_estimated_text_tokens += est["non_thinking"]["text"]
            self.total_estimated_tool_tokens += est["non_thinking"]["tool"]
            source = est.get("thinking_source")
            if source in self.thinking_source_counts:
                self.thinking_source_counts[source] += 1

        if latency["server_time_ms"]:
            self._sum_server += latency["server_time_ms"]
            self._cnt_server += 1
        if latency["ttfb_ms"]:
            self._sum_ttfb += latency["ttfb_ms"]
            self._cnt_ttfb += 1
        if latency["ttf_thinking_ms"]:
            self._sum_ttf_thinking += latency["ttf_thinking_ms"]
            self._cnt_ttf_thinking += 1
        if latency["ttf_answer_ms"]:
            self._sum_ttf_answer += latency["ttf_answer_ms"]
            self._cnt_ttf_answer += 1
        if latency["ttlt_ms"]:
            self._sum_ttlt += latency["ttlt_ms"]
            self._cnt_ttlt += 1
            self.total_latency_ms += latency["ttlt_ms"]
        if throughput["output_tokens_per_sec"]:
            self._sum_otps += throughput["output_tokens_per_sec"]
            self._cnt_otps += 1

        return self._maybe_save()

    def record_answers(self, answers: list[tuple[str, object, bool]]) -> int:
        """Fill pending question tool_uses with their tool_result answers.
        Each answer is (tool_use_id, content, is_error). First sighting wins;
        the id then leaves pending_answers so request scanning stops once no
        questions are outstanding. Returns how many were filled."""
        filled = 0
        for tool_use_id, content, is_error in answers:
            tool_use = self.pending_answers.pop(tool_use_id, None)
            if tool_use is None:
                continue
            tool_use["result"] = {
                "content": _tool_result_text(content),
                "is_error": bool(is_error),
                "received_at": datetime.now().isoformat(),
            }
            filled += 1
        if filled:
            # Answers mutate an already-recorded turn, so nudge the periodic
            # saver; the final save at shutdown persists them regardless.
            self._maybe_save()
        return filled

    def _parse_session_cost(self) -> dict:
        """Extract cost from claude_code_metrics. Returns empty dict if unavailable."""
        if not self.claude_code_metrics:
            return {"total_cost_usd": None, "cost_by_model": {}}

        for m in self.claude_code_metrics.get("metrics", []):
            if m.get("name") == "claude_code.cost.usage":
                data_points = m.get("data_points", [])
                if not data_points:
                    break
                total_cost = 0.0
                cost_by_model = {}
                for dp in data_points:
                    val = dp.get("value", 0)
                    total_cost += val
                    model = dp.get("attributes", {}).get("model", "unknown")
                    cost_by_model[model] = round(cost_by_model.get(model, 0) + val, 6)
                return {"total_cost_usd": round(total_cost, 6), "cost_by_model": cost_by_model}

        return {"total_cost_usd": None, "cost_by_model": {}}

    def _compute_summary(self) -> dict:
        # Rate limits from the last turn whose typed block is non-zero (a 429 carries
        # the headers; a trailing header-less failure must not blank them).
        last_rl = next((t["rate_limits"] for t in reversed(self.turns) if any(t["rate_limits"].values())),
                       self.turns[-1]["rate_limits"] if self.turns else {})

        return {
            "total_turns": len(self.turns),
            "aborted_turns": self.aborted_turns,
            "total_tool_calls": self.total_tool_calls,
            "models_used": sorted(self.models_used),
            "models_responded": sorted(self.models_responded),
            "fallback_turns": self.fallback_turns,
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "estimated_output": {
                    "thinking": self.total_estimated_thinking_tokens,
                    "thinking_source_counts": dict(self.thinking_source_counts),
                    "non_thinking": {
                        "total": self.total_estimated_text_tokens + self.total_estimated_tool_tokens,
                        "text": self.total_estimated_text_tokens,
                        "tool": self.total_estimated_tool_tokens,
                    },
                },
                "total_cache_read": self.total_cache_read,
                "total_cache_creation": self.total_cache_creation,
            },
            "cost": self._parse_session_cost(),
            "latency": {
                "avg_server_time_ms": round(self._sum_server / self._cnt_server, 1) if self._cnt_server else 0.0,
                "avg_ttfb_ms": round(self._sum_ttfb / self._cnt_ttfb, 1) if self._cnt_ttfb else 0.0,
                "avg_ttf_thinking_ms": round(self._sum_ttf_thinking / self._cnt_ttf_thinking, 1) if self._cnt_ttf_thinking else 0.0,
                "avg_ttf_answer_ms": round(self._sum_ttf_answer / self._cnt_ttf_answer, 1) if self._cnt_ttf_answer else 0.0,
                "avg_ttlt_ms": round(self._sum_ttlt / self._cnt_ttlt, 1) if self._cnt_ttlt else 0.0,
                "total_ms": round(self.total_latency_ms, 1),
            },
            "throughput": {
                "avg_output_tokens_per_sec": round(self._sum_otps / self._cnt_otps, 1) if self._cnt_otps else 0.0,
            },
            "rate_limits": last_rl,
        }

    def _save(self) -> bool:
        if not self.log_file:
            return False
        try:
            data = {
                "session": {
                    "id": self.session_id or "unknown",
                    "started_at": self.session_start,
                    "exported_at": datetime.now().isoformat(),
                    "extractor_version": VERSION,
                    "cli_version": self.cli_version,
                    "claude_config_dir": self.claude_config_dir,
                    "thinking_disabled": self.thinking_disabled,
                },
                "summary": self._compute_summary(),
                "turns": self.turns,
            }
            if self.claude_code_metrics:
                data["claude_code_metrics"] = self.claude_code_metrics
            self.log_file.write_bytes(json_dumps(data))
        except Exception as e:
            print(f"[LOG] Save failed: {e}", file=sys.stderr)
            return False
        return True

    def _maybe_save(self) -> tuple[bool, str]:
        """Returns (saved, filename) if a save occurred."""
        now = time.monotonic()
        if (
            self.turn_counter - self._last_save_turn >= SAVE_EVERY_TURNS
            or now - self._last_save_ts >= SAVE_EVERY_SECS
        ):
            if self._save():
                self._last_save_ts = now
                self._last_save_turn = self.turn_counter
                return True, self.log_file.name
        return False, ""

    def save(self) -> tuple[bool, str]:
        """Final save. Returns (saved, filename)."""
        if self._save():
            return True, self.log_file.name if self.log_file else ""
        return False, ""

    def get_final_summary(self) -> str:
        if not self.turns:
            return "No conversation data captured."

        summary = self._compute_summary()
        tok = summary["tokens"]
        lat = summary["latency"]

        lines = [
            f"Session: {self.session_id or 'unknown'}",
            f"  Turns: {summary['total_turns']}",
            f"  Models: {', '.join(summary['models_used'])}",
            f"  Tokens: {tok['total_input']:,} in / {tok['total_output']:,} out",
            f"  Avg TTFB: {lat['avg_ttfb_ms']:.0f}ms",
            f"  Total time: {lat['total_ms']/1000:.1f}s",
        ]

        if self.claude_code_metrics:
            metrics = self.claude_code_metrics.get("metrics", [])
            for m in metrics:
                if m.get("name") == "claude_code.cost.usage":
                    total_cost = sum(dp.get("value", 0) for dp in m.get("data_points", []))
                    lines.append(f"  Cost: ${total_cost:.4f}")
                    break

        return "\n".join(lines)


try:
    from mitmproxy import http, ctx

    class ClaudeLatencyAddon:
        __slots__ = ('active_requests', 'conversation_log', '_session_started', '_shutdown_flag')

        def __init__(self):
            self.active_requests: dict[str, TurnMetrics] = {}
            self.conversation_log: Optional[ConversationLog] = None
            self._session_started: bool = False
            self._shutdown_flag: bool = False

        def load(self, loader):
            loader.add_option(name="latency_log_dir", typespec=str, default="",
                              help="Directory for logging conversations")
            loader.add_option(name="run_id", typespec=str, default="",
                              help="Run identifier embedded in the JSON trace filename for cross-file matching")
            loader.add_option(name="claude_config_dir", typespec=str, default="",
                              help="Per-run path under claude_config/ where Claude Code's raw state ends up after the shutdown rename (claude_config/<run_id>/). Recorded as session.claude_config_dir in the JSON trace. Note: differs from the CLAUDE_CONFIG_DIR literal in the launch command, which uses the stable runtime path claude_config/active_<port>/.")
            loader.add_option(name="thinking_disabled", typespec=bool, default=False,
                              help="Whether the launch command was built with --disable-thinking (alwaysThinkingEnabled:false + MAX_THINKING_TOKENS=0). Recorded as session.thinking_disabled in the JSON trace. Reflects the launch configuration applied, not a wire-level confirmation that the API disabled thinking.")

        def configure(self, updates):
            # Only initialize once: a later runtime option change must not discard
            # the in-flight session's accumulated turns.
            if self.conversation_log is not None:
                return
            if "latency_log_dir" in updates or "run_id" in updates or "claude_config_dir" in updates:
                log_dir = ctx.options.latency_log_dir
                if log_dir:
                    self.conversation_log = ConversationLog(
                        Path(log_dir),
                        run_id=ctx.options.run_id,
                        claude_config_dir=ctx.options.claude_config_dir,
                        thinking_disabled=ctx.options.thinking_disabled,
                    )

        def done(self):
            """Called when proxy shuts down (Ctrl+C)."""
            if self._shutdown_flag:
                return
            self._shutdown_flag = True

            # Tracked turns still in flight at shutdown were never recorded (e.g. a
            # mid-turn Ctrl+C). Surface the count rather than dropping them silently.
            if self.active_requests:
                ctx.log.warn(f"[LOG] {len(self.active_requests)} request(s) still in flight at shutdown (not recorded)")

            saved = False
            if self.conversation_log:
                saved, filename = self.conversation_log.save()
                if saved:
                    ctx.log.info(f"[LOG] Final save completed → {filename}")

            print()
            log = self.conversation_log
            if log and log.log_file and log.turns:
                if saved:
                    print(f"Trace saved to: {log.log_file}")
                else:
                    print(f"WARNING: final save failed, {log.log_file} holds the last checkpoint if any")
            else:
                print("No conversation data captured.")

        def _sanitize_metrics(self, metrics: dict) -> dict:
            sensitive_keys = {"user.id", "user.email", "user.account_uuid", "organization.id"}

            def sanitize_attrs(attrs: dict) -> dict:
                return {k: v for k, v in attrs.items() if k not in sensitive_keys}

            sanitized = {}
            if "resource_attributes" in metrics:
                sanitized["resource_attributes"] = sanitize_attrs(metrics["resource_attributes"])

            if "metrics" in metrics:
                sanitized["metrics"] = []
                for m in metrics["metrics"]:
                    sanitized_metric = {
                        "name": m.get("name"),
                        "description": m.get("description"),
                        "unit": m.get("unit"),
                        "data_points": []
                    }
                    for dp in m.get("data_points", []):
                        sanitized_dp = {"value": dp.get("value"), "timestamp": dp.get("timestamp")}
                        if "attributes" in dp:
                            sanitized_dp["attributes"] = sanitize_attrs(dp["attributes"])
                        sanitized_metric["data_points"].append(sanitized_dp)
                    sanitized["metrics"].append(sanitized_metric)

            return sanitized

        def _find_answers(self, messages) -> list[tuple[str, object, bool]]:
            """Collect tool_result blocks answering a pending question tool_use
            from a request's messages: (tool_use_id, content, is_error)."""
            pending = self.conversation_log.pending_answers
            found = []
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if (isinstance(block, dict) and block.get("type") == "tool_result"
                            and block.get("tool_use_id") in pending):
                        found.append((block["tool_use_id"], block.get("content"), bool(block.get("is_error"))))
            return found

        def request(self, flow: http.HTTPFlow):
            req = flow.request
            if not any(host in req.host for host in CLAUDE_API_HOSTS):
                return

            # Capture metrics when session sends them (but don't auto-shutdown)
            if req.path.endswith("/claude_code/metrics"):
                if req.content and self.conversation_log:
                    try:
                        self.conversation_log.claude_code_metrics = self._sanitize_metrics(json_loads(req.content))
                        ctx.log.info("Captured session metrics")
                    except Exception:
                        pass
                return

            if req.method != "POST" or "/v1/messages" not in req.path:
                return
            if "batch" in req.path or "count" in req.path:
                return

            try:
                body = json_loads(req.content)

                # Answers to an outstanding question tool call (AskUserQuestion)
                # come back as a tool_result block in a later request. Scanned
                # only while a question is pending — questions are rare, so the
                # steady-state cost is this emptiness check. Runs before the
                # stream/user-message early-returns in case the answer rides on
                # a request that isn't tracked as a turn.
                if self.conversation_log and self.conversation_log.pending_answers:
                    recorded = self.conversation_log.record_answers(self._find_answers(body.get("messages") or ()))
                    if recorded:
                        ctx.log.info(f"[ASK] {recorded} question answer(s) recorded")

                if not body.get("stream"):
                    return

                messages = body.get("messages")
                # Match on any user message, not just a trailing one (Claude can
                # append a system message after the user turn, mid-conversation).
                if not messages or not any(m.get("role") == "user" for m in messages):
                    return

                if not self._session_started:
                    self._session_started = True
                    ctx.log.info("Claude Code session detected!")

                thinking = body.get("thinking", {})
                tools = body.get("tools") or []

                metrics = TurnMetrics()
                metrics.timestamp = datetime.now().isoformat()
                metrics.model = str(body.get("model") or "unknown")
                metrics.request_start = time.perf_counter()
                metrics.messages_count = len(messages)
                metrics.request_size_bytes = len(req.content) if req.content else 0
                metrics.max_tokens = _int_or_zero(body.get("max_tokens"))
                metrics.thinking_budget_tokens = _int_or_zero(thinking.get("budget_tokens")) if isinstance(thinking, dict) else 0
                metrics.thinking_type = (thinking.get("type") or "") if isinstance(thinking, dict) else ""
                output_config = body.get("output_config")
                metrics.effort = (output_config.get("effort") or "") if isinstance(output_config, dict) else ""
                metrics.thinking_display = (thinking.get("display") or "") if isinstance(thinking, dict) else ""
                metrics.tools_count = len(tools)
                metrics.tool_names = [t.get("name") or "" for t in tools if isinstance(t, dict)]
                metrics.request_params = _scrub_params({k: v for k, v in body.items() if k not in _REQUEST_BODY_BULK_KEYS})
                metrics.betas = [b.strip() for b in _clean_text(req.headers.get("anthropic-beta", "")).split(",") if b.strip()]
                metrics.request_class = _clean_text(req.headers.get("x-claude-code-request-class", ""))
                metrics.client_metadata = _parse_client_metadata(body.get("metadata"))

                user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
                user_content = user_msg.get("content", "") if user_msg else ""
                if isinstance(user_content, list):
                    user_content = " ".join(b.get("text", "") for b in user_content if b.get("type") == "text")
                metrics.user_message = str(user_content)

                mid_system = []
                for m in messages:
                    if m.get("role") == "system":
                        sys_content = m.get("content", "")
                        if isinstance(sys_content, list):
                            sys_content = " ".join(b.get("text", "") for b in sys_content if b.get("type") == "text")
                        mid_system.append(str(sys_content))
                metrics.mid_conversation_system = mid_system

                system = body.get("system", "")
                if isinstance(system, list):
                    system = " ".join(b.get("text", "") for b in system if b.get("type") == "text")
                metrics.system_prompt = str(system)
                metrics.cache_control_counts = _count_cache_control(tools, body.get("system"), messages)

                self.active_requests[flow.id] = metrics
                ctx.log.info(f"[TURN] Started: {metrics.model}")

            except Exception as e:
                ctx.log.warn(f"[TURN] Parse error: {e}")

        def responseheaders(self, flow: http.HTTPFlow):
            metrics = self.active_requests.get(flow.id)
            if not metrics:
                return

            metrics.first_byte = time.perf_counter()
            h = flow.response.headers

            # Forward SSE chunks to the client as they arrive instead of mitmproxy's
            # default buffer-until-complete, so the client is not blocked until the
            # upstream response finishes. The tap copies each chunk for metric
            # parsing while passing it through unchanged. Non-SSE responses stay
            # buffered. Installed before any header parsing so a malformed header
            # can never leave the response buffered.
            if "text/event-stream" in h.get("content-type", ""):
                metrics.content_encoding = h.get("content-encoding", "")
                metrics.was_streamed = True
                metrics.stream_buf = bytearray()
                buf = metrics.stream_buf

                def stream_handler(chunk: bytes) -> bytes:
                    if chunk:
                        buf.extend(chunk)
                    return chunk

                flow.response.stream = stream_handler

            metrics.status_code = flow.response.status_code
            metrics.response_headers = _pick_headers(h, _RESPONSE_HEADER_NAMES, _RESPONSE_HEADER_PREFIXES)

            # Direct API uses request-id; proxies may use x-request-id
            metrics.request_id = _clean_text(h.get("request-id", h.get("x-request-id", "")))
            metrics.server_time_ms = _int_header(h, "x-envoy-upstream-service-time")
            metrics.ratelimit_input_tokens_limit = _int_header(h, "anthropic-ratelimit-input-tokens-limit")
            metrics.ratelimit_input_tokens_remaining = _int_header(h, "anthropic-ratelimit-input-tokens-remaining")
            metrics.ratelimit_requests_limit = _int_header(h, "anthropic-ratelimit-requests-limit")
            metrics.ratelimit_requests_remaining = _int_header(h, "anthropic-ratelimit-requests-remaining")

        def response(self, flow: http.HTTPFlow):
            metrics = self.active_requests.pop(flow.id, None)
            if not metrics:
                return

            metrics.stream_end = time.perf_counter()
            # Streamed flows leave flow.response.content empty, so parse the bytes
            # the tap accumulated (decoded by the captured Content-Encoding).
            if metrics.was_streamed:
                content = _decode_sse_body(metrics.stream_buf or b"", metrics.content_encoding)
            else:
                # strict=False returns an undecodable body raw instead of
                # raising, so the turn is still recorded.
                content = flow.response.get_content(strict=False)
            self._record_body(metrics, content)

            # An API error (in-stream error event or error body) or an HTTP error
            # status is a failed turn, even with a non-API body such as an HTML 502.
            if metrics.api_errors or (metrics.status_code is not None and metrics.status_code >= 400):
                metrics.aborted = True
                if not metrics.error_message:
                    metrics.error_message = f"HTTP {metrics.status_code}"

            # No terminal stop_reason: a stream closed cleanly before message_delta, an
            # empty or non-SSE 2xx body, or a redirect. Every tracked request records
            # exactly one turn, here or in the error hook.
            if not metrics.stop_reason and not metrics.aborted:
                metrics.aborted = True
                metrics.error_message = "response ended before completion (no stop_reason)"

            if self.conversation_log:
                saved, filename = self.conversation_log.add_turn(metrics)
                if saved:
                    ctx.log.info(f"[LOG] Checkpoint saved → {filename}")

        def error(self, flow: http.HTTPFlow):
            # A flow that errors mid-stream (client disconnect, upstream cut, etc.)
            # never reaches the response hook, so this hook records the partial turn
            # from whatever the tap salvaged, keeping the failure visible. pop()
            # keeps this safe against mitmproxy versions that fire both error and
            # response for one flow.
            metrics = self.active_requests.pop(flow.id, None)
            if not metrics:
                return

            metrics.stream_end = time.perf_counter()
            metrics.aborted = True
            if metrics.was_streamed and metrics.stream_buf:
                self._record_body(metrics, _decode_sse_body(metrics.stream_buf, metrics.content_encoding))
            # An API error salvaged from the body is the more specific cause; the
            # transport error is the fallback, prefixed with any HTTP error status.
            if not metrics.error_message:
                transport = str(flow.error) if flow.error else "stream aborted"
                if metrics.status_code is not None and metrics.status_code >= 400:
                    metrics.error_message = f"HTTP {metrics.status_code}: {transport}"
                else:
                    metrics.error_message = transport

            if self.conversation_log:
                saved, filename = self.conversation_log.add_turn(metrics)
                if saved:
                    ctx.log.info(f"[LOG] Aborted turn recorded → {filename}")

        def _record_body(self, metrics: TurnMetrics, content: bytes) -> None:
            """Parse a response body as an API error object or as SSE (a streamed
            body with no SSE event is retried as an error object). Error objects
            land in api_errors; the caller decides whether the turn is aborted."""
            if not content:
                return
            metrics.response_size_bytes = len(content)
            error = None if metrics.was_streamed else _parse_error_body(content)
            if error is None:
                self._parse_sse(content, metrics)
                if metrics.was_streamed and metrics.total_events == 0:
                    error = _parse_error_body(content)
            if error is not None:
                metrics.api_errors.append(error)
                metrics.error_message = f"HTTP {metrics.status_code}: {_error_text(error)}"

        def _parse_sse(self, content: bytes, m: TurnMetrics):
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                return

            current_tool_use = None
            tool_input_parts = []
            event_idx = 0

            for data_str in _iter_sse_data(text):
                if not data_str or data_str == "[DONE]":
                    continue

                m.total_events += 1
                etype = ""

                try:
                    data = json_loads(data_str)
                    etype = data.get("type", "")
                    _bump(m.event_types, etype)

                    if etype == "message_start":
                        msg = data.get("message", {})
                        m.message_id = msg.get("id") or ""
                        m.response_model = str(msg.get("model") or "")
                        m.message_meta = {k: v for k, v in msg.items() if k not in ("content", "usage", "input_transformations")}
                        m.input_transformations = _summarize_transformations(msg.get("input_transformations"))
                        usage = msg.get("usage")
                        usage = usage if isinstance(usage, dict) else {}
                        m.usage = dict(usage)
                        m.input_tokens = _int_or_zero(usage.get("input_tokens"))
                        m.cache_creation_input_tokens = _int_or_zero(usage.get("cache_creation_input_tokens"))
                        m.cache_read_input_tokens = _int_or_zero(usage.get("cache_read_input_tokens"))
                        m.service_tier = usage.get("service_tier") or ""
                        cc = usage.get("cache_creation", {})
                        if isinstance(cc, dict):
                            m.cache_creation_5m_tokens = _int_or_zero(cc.get("ephemeral_5m_input_tokens"))
                            m.cache_creation_1h_tokens = _int_or_zero(cc.get("ephemeral_1h_input_tokens"))

                    elif etype == "content_block_start":
                        cb = data.get("content_block", {})
                        cb_type = cb.get("type", "")
                        _bump(m.block_types, cb_type)
                        if cb_type == "thinking" and m.first_thinking_idx is None:
                            m.first_thinking_idx = event_idx
                        elif cb_type == "tool_use":
                            current_tool_use = {"name": cb.get("name") or "", "id": cb.get("id") or "", "input": ""}
                            tool_input_parts = []
                        elif cb_type == "fallback":
                            # message_start names the model the request started
                            # on; the fallback block announces the one serving it.
                            m.fallbacks.append({k: v for k, v in cb.items() if k != "type"})
                            served = cb.get("to")
                            if isinstance(served, dict) and isinstance(served.get("model"), str):
                                m.response_model = served["model"]

                    elif etype == "content_block_delta":
                        delta = data.get("delta", {})
                        dtype = delta.get("type", "")
                        _bump(m.delta_types, dtype)
                        if dtype == "thinking_delta":
                            if m.first_thinking_idx is None:
                                m.first_thinking_idx = event_idx
                            m.thinking_parts.append(delta.get("thinking") or "")
                        elif dtype == "text_delta":
                            if m.first_text_idx is None:
                                m.first_text_idx = event_idx
                            m.text_parts.append(delta.get("text") or "")
                        elif dtype == "input_json_delta" and current_tool_use is not None:
                            tool_input_parts.append(delta.get("partial_json") or "")

                    elif etype == "content_block_stop":
                        if current_tool_use is not None:
                            input_str = "".join(tool_input_parts)
                            try:
                                current_tool_use["input"] = json_loads(input_str)
                            except Exception:
                                current_tool_use["input"] = input_str
                            m.tool_uses.append(current_tool_use)
                            current_tool_use = None
                            tool_input_parts = []

                    elif etype == "message_delta":
                        delta = data.get("delta")
                        # The final delta carries the closing fields (stop_details, container, ...)
                        # and siblings such as context_management; overlay them on the snapshot.
                        if isinstance(delta, dict):
                            if "stop_reason" in delta:
                                m.stop_reason = delta["stop_reason"]
                            m.message_meta.update(delta)
                        for key, value in data.items():
                            if key not in ("type", "delta", "usage"):
                                m.message_meta[key] = value
                        # Final token counts override message_start's when present;
                        # a null counter (allowed by the API) keeps the earlier value.
                        usage = data.get("usage")
                        usage = usage if isinstance(usage, dict) else {}
                        m.usage.update(usage)
                        if isinstance(usage.get("output_tokens"), int):
                            m.output_tokens = usage["output_tokens"]
                        if isinstance(usage.get("input_tokens"), int):
                            m.input_tokens = usage["input_tokens"]
                        if isinstance(usage.get("cache_creation_input_tokens"), int):
                            m.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
                        if isinstance(usage.get("cache_read_input_tokens"), int):
                            m.cache_read_input_tokens = usage["cache_read_input_tokens"]
                        details = usage.get("output_tokens_details")
                        if isinstance(details, dict) and isinstance(details.get("thinking_tokens"), int):
                            m.thinking_tokens = details["thinking_tokens"]
                        ctx_mgmt = data.get("context_management", {})
                        if isinstance(ctx_mgmt, dict) and ctx_mgmt.get("applied_edits"):
                            m.context_mgmt_edits = ctx_mgmt["applied_edits"]

                    elif etype == "error":
                        # In-stream API error (overloaded, invalid_request, ...).
                        error = _api_error(data.get("error"))
                        m.api_errors.append(error)
                        if not m.error_message:
                            m.error_message = _error_text(error)

                except Exception as e:
                    m.parse_errors += 1
                    if m.parse_error_sample is None:
                        m.parse_error_sample = f"{type(e).__name__}: {e}"
                        ctx.log.warn(f"[SSE] parse error on {etype or 'payload'}: {m.parse_error_sample}")

                event_idx += 1

    addons = [ClaudeLatencyAddon()]

except ImportError:
    addons = []


def main():
    import argparse
    import subprocess
    import shutil

    parser = argparse.ArgumentParser(description="Claude Trace Extractor")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Proxy port (default: 8080)")
    if MODEL_REQUIRED:
        parser.add_argument("--model", "-m", required=True, help="Model name to use (e.g., claude-opus-4-7)")
    else:
        parser.add_argument("--model", "-m", default=CLAUDE_DEFAULT_MODEL, help=f"Model name to use (default: {CLAUDE_DEFAULT_MODEL})")
    permission_help = (
        "Permission mode "
        "(skip = --dangerously-skip-permissions, auto = --permission-mode auto)"
    )
    if CONFIG_REQUIRED:
        parser.add_argument("--effort-level", "-e", required=True, help="Thinking effort level (e.g., medium, high, xhigh)")
        parser.add_argument("--permission-mode", choices=CLAUDE_PERMISSION_MODES, required=True, help=permission_help)
    else:
        parser.add_argument("--effort-level", "-e", default=CLAUDE_DEFAULT_EFFORT_LEVEL, help=f"Thinking effort level (default: {CLAUDE_DEFAULT_EFFORT_LEVEL})")
        parser.add_argument("--permission-mode", choices=CLAUDE_PERMISSION_MODES, default=CLAUDE_DEFAULT_PERMISSION_MODE, help=f"{permission_help} (default: {CLAUDE_DEFAULT_PERMISSION_MODE})")
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to Claude (shows full tool calls, results, and thinking inline).")
    parser.add_argument("--disable-autoupdate", action="store_true", help="Set DISABLE_AUTOUPDATER=1 on the Claude launch command so Claude Code does not auto-update.")
    parser.add_argument("--show-thinking-summaries", action="store_true", help="Add showThinkingSummaries:true to the Claude --settings JSON so summarized thinking text is returned in API responses (and captured in the JSON trace).")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable extended thinking for the session: sets alwaysThinkingEnabled:false and env.MAX_THINKING_TOKENS:0 in the Claude --settings JSON. Default is thinking enabled.")
    parser.add_argument("--require-cli-version", metavar="X.Y.Z", default=None, help="Require an exact installed version of claude before running. Prints an install command and exits 1 if the installed version doesn't match or claude isn't on PATH. Detection runs `claude --version`.")
    parser.add_argument("--no-claude-config", action="store_true", help="Disable the CLAUDE_CONFIG_DIR redirect. By default the launch command sets CLAUDE_CONFIG_DIR to ./claude_config/active_<port>/ (a stable path keyed by proxy port so reusing a previously-printed launch command still works). Claude Code's session JSONL and debug log are written there during the run, then copied into ./logs/<run_id>/ at extractor shutdown (per-subagent transcripts are bundled into subagents.zip); the runtime dir is renamed to ./claude_config/<run_id>/ at the same time so the next run starts with an empty active_<port>/. With this flag, no redirect happens and Claude Code uses the worker's regular ~/.claude/ instead.")
    args = parser.parse_args()

    if not is_port_free(args.port):
        print(f"ERROR: Port {args.port} is in use. Try --port {args.port + 1}")
        sys.exit(1)

    cert_path = get_cert_path()
    if not cert_path.exists():
        print("ERROR: mitmproxy certificate not found.")
        print("Run 'mitmproxy' once to generate certificates, then try again.")
        sys.exit(1)

    if args.require_cli_version is not None:
        err = enforce_cli_version("claude", args.require_cli_version)
        if err:
            print(err)
            sys.exit(1)

    installed_cli_version = get_cli_version("claude")
    installed_cli_str = f"claude {installed_cli_version}" if installed_cli_version else "claude (not detected)"

    script_dir = Path(__file__).parent.resolve()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_dir = script_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # See main.py for the rationale on the two-path design. Briefly: the
    # runtime path is what goes in the launch command (stable per port so
    # stale launch commands still work); the final path is what the run's raw
    # state is renamed to at shutdown and what the JSON trace records.
    claude_config_runtime: Optional[Path] = None
    claude_config_final: Optional[Path] = None
    claude_config_row = ""
    if not args.no_claude_config:
        claude_config_runtime = script_dir / "claude_config" / f"active_{args.port}"
        claude_config_runtime.mkdir(parents=True, exist_ok=True)
        claude_config_final = script_dir / "claude_config" / run_id
        claude_config_row = f"  Claude config:  {claude_config_runtime}\n"
    append_system_prompt_file = seed_claude_append_system_prompt(claude_config_runtime or run_dir, args.model)

    system, is_wsl = get_platform_info()
    platform_str = f"{system}" + (" (WSL)" if is_wsl else "")

    launch_cmd = build_launch_command(
        cert_path, args.port, "claude", args.model,
        effort_level=args.effort_level,
        permission_mode=args.permission_mode,
        verbose=args.verbose,
        disable_autoupdate=args.disable_autoupdate,
        show_thinking_summaries=args.show_thinking_summaries,
        disable_thinking=args.disable_thinking,
        claude_config_dir=claude_config_runtime,
        append_system_prompt_file=append_system_prompt_file,
    )

    copied = copy_to_clipboard(launch_cmd)
    clipboard_msg = "(copied to clipboard)" if copied else "(copy the command below)"

    sep = "=" * 70
    print(f"""
{sep}
  CLAUDE CODE TRACE EXTRACTOR v{VERSION}
{sep}
  Platform:       {platform_str}
  Installed CLI:  {installed_cli_str}
  Proxy:          http://127.0.0.1:{args.port}
  Run dir:        {run_dir}
{claude_config_row}{sep}

  Run this in your project directory {clipboard_msg}:

  {launch_cmd}

{sep}
  Waiting for Claude Code to connect...
  When done: exit CLI first, then Ctrl+C here to save.
{sep}
""")

    mitmdump_cmd = shutil.which("mitmdump")
    if not mitmdump_cmd:
        print("ERROR: mitmdump not found. Install mitmproxy: pip install mitmproxy")
        sys.exit(1)

    mitmdump_args = [
        mitmdump_cmd, "-p", str(args.port),
        "-s", str(Path(__file__).resolve()),
        "--set", f"latency_log_dir={run_dir}",
        "--set", f"run_id={run_id}",
    ]
    if claude_config_final is not None:
        mitmdump_args += ["--set", f"claude_config_dir={claude_config_final}"]
    mitmdump_args += ["--set", f"thinking_disabled={'true' if args.disable_thinking else 'false'}"]
    mitmdump_args.append("--ssl-insecure")
    proc = subprocess.Popen(mitmdump_args)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        proc.terminate()
        proc.wait()

    print("\nProxy stopped.")

    preserved_dir: Optional[Path] = claude_config_final
    if claude_config_runtime is not None:
        def _copy_set(srcs, stem, ext):
            if len(srcs) == 1:
                shutil.copy2(srcs[0], run_dir / f"{stem}{ext}")
            elif len(srcs) > 1:
                for i, p in enumerate(srcs, 1):
                    shutil.copy2(p, run_dir / f"{stem}_{i}{ext}")
        _copy_set(sorted(claude_config_runtime.glob("projects/*/*.jsonl")), "session", ".jsonl")
        _copy_set(sorted(claude_config_runtime.glob("debug/*.txt")), "debug", ".txt")
        try:
            zip_subagent_transcripts(claude_config_runtime, run_dir)
        except Exception as e:
            print(f"WARNING: could not bundle subagent transcripts: {e}")
        try:
            claude_config_runtime.rename(claude_config_final)
        except OSError as e:
            print(f"WARNING: could not rename {claude_config_runtime} -> {claude_config_final}: {e}")
            preserved_dir = claude_config_runtime

    def _order_key(p):
        name = p.name
        if name == "trace.json": return (0, name)
        if name == "transcript.txt": return (1, name)
        if name.startswith("session") and name.endswith(".jsonl"): return (2, name)
        if name == "subagents.zip": return (3, name)
        if name.startswith("debug") and name.endswith(".txt"): return (4, name)
        return (5, name)
    files = sorted(
        (p for p in run_dir.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=_order_key,
    )
    if files:
        print(f"\nFiles to share for this run ({run_dir}):")
        for p in files:
            print(f"  {p.name}")
        if preserved_dir is not None and preserved_dir.exists():
            print(f"\n(Originals and additional Claude Code state preserved at {preserved_dir})")


if __name__ == "__main__":
    main()
