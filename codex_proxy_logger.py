"""Codex CLI Trace Extractor - captures conversations with latency metrics."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import orjson

from cli_utils import (
    build_launch_command,
    copy_to_clipboard,
    enforce_cli_version,
    get_cert_path,
    get_cli_version,
    get_platform_info,
    is_port_free,
    seed_codex_config,
    VERSION,
    MODEL_REQUIRED,
    CONFIG_REQUIRED,
    CODEX_DEFAULT_MODEL,
    CODEX_DEFAULT_REASONING_EFFORT,
)

def json_loads(s): return orjson.loads(s)
def json_dumps(obj): return orjson.dumps(obj, option=orjson.OPT_INDENT_2)

SAVE_EVERY_TURNS = 5
SAVE_EVERY_SECS = 5.0

# Supported API hosts for Codex/OpenAI
CODEX_API_HOSTS = ("chatgpt.com", "dataannotation.tech")

if TYPE_CHECKING:
    from mitmproxy import http

# mitmproxy's own content decoder (gzip/deflate/br/zstd), used to decode streamed
# SSE bodies. They arrive still-encoded via flow.response.stream, unlike buffered
# flow.response.content (which mitmproxy decodes automatically). Guarded so the
# module imports even without mitmproxy, matching the addon import below.
try:
    from mitmproxy.net import encoding as _mitm_encoding
except Exception:
    _mitm_encoding = None


def _bump(counter: dict, key) -> None:
    """Count `key`, coerced to str so the trace stays JSON-serialisable."""
    key = key if isinstance(key, str) else repr(key)
    counter[key] = counter.get(key, 0) + 1


def _iter_sse_data(text: str):
    data_parts = []
    for line in text.splitlines():
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
        self.response_id = ""
        # x-oai-request-id response header.
        self.request_id = ""

        self.messages_count = 0
        self.user_message = ""
        self.instructions = ""
        self.request_size_bytes = 0
        self.reasoning_effort = ""
        self.tools_count = 0

        self.request_start = 0.0
        self.first_byte = None
        self.stream_end = None
        self.created_at = None
        self.completed_at = None

        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.cached_tokens = 0
        self.cache_write_tokens = 0
        self.total_tokens = 0

        self.reasoning_summary = ""
        self.text_parts = []
        self.tool_calls = []
        self.status = ""
        # response.completed incomplete_details (why the model stopped early), else None.
        self.incomplete_details = None
        self.response_size_bytes = 0
        self.service_tier = ""
        self.prompt_cache_key = ""

        self.total_events = 0
        # Per-type tallies of SSE events and output items, so an unhandled type stays visible.
        self.event_types = {}
        self.item_types = {}
        self.first_reasoning_seq = None
        self.first_text_seq = None
        self.last_reasoning_seq = None

        # Streaming-passthrough state. stream_buf accumulates the still-encoded
        # response body forwarded via flow.response.stream. aborted marks a turn
        # that ended mid-flight (recorded so failures stay visible).
        self.stream_buf = None
        self.content_encoding = ""
        self.was_streamed = False
        self.aborted = False
        self.error_message = ""

    def to_dict(self) -> dict:
        fb, rs, se = self.first_byte, self.request_start, self.stream_end

        ttfb = round((fb - rs) * 1000, 1) if fb and rs else None
        ttlt = round((se - rs) * 1000, 1) if se and rs else None
        stream_dur = (se - fb) if se and fb else 0

        # Calculate server time from response timestamps if available
        server_time_ms = None
        if self.created_at and self.completed_at:
            server_time_ms = (self.completed_at - self.created_at) * 1000

        # Estimate timing based on sequence numbers
        ttf_thinking = ttf_answer = ttlt_thinking = None
        if self.total_events > 0 and ttfb and stream_dur > 0:
            ms_per_event = (stream_dur * 1000) / self.total_events
            if self.first_reasoning_seq is not None:
                ttf_thinking = round(ttfb + (self.first_reasoning_seq * ms_per_event), 1)
            if self.first_text_seq is not None:
                ttf_answer = round(ttfb + (self.first_text_seq * ms_per_event), 1)
            if self.last_reasoning_seq is not None:
                ttlt_thinking = round(ttfb + (self.last_reasoning_seq * ms_per_event), 1)

        otps = round(self.output_tokens / (ttlt / 1000), 1) if ttlt and ttlt > 0 and self.output_tokens > 0 else None
        total_input = self.input_tokens + self.cached_tokens

        return {
            "turn": self.turn_id,
            "timestamp": self.timestamp,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "model": self.model,
            "service_tier": self.service_tier,
            "request": {
                "messages_count": self.messages_count,
                "user_message": self.user_message,
                "instructions": self.instructions,
                "size_bytes": self.request_size_bytes,
                "reasoning_effort": self.reasoning_effort,
                "tools_count": self.tools_count,
                "prompt_cache_key": self.prompt_cache_key,
            },
            "response": {
                "reasoning_summary": self.reasoning_summary,
                "text": "".join(self.text_parts),
                "tool_calls": self.tool_calls,
                "status": self.status,
                "incomplete_details": self.incomplete_details,
                "size_bytes": self.response_size_bytes,
                "event_types": self.event_types,
                "item_types": self.item_types,
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "reasoning": self.reasoning_tokens,
                "cached": self.cached_tokens,
                "cache_write": self.cache_write_tokens,
                "total": self.total_tokens,
                "total_input": total_input,
            },
            "latency": {
                "server_time_ms": server_time_ms,
                "ttfb_ms": ttfb,
                "ttf_thinking_ms": ttf_thinking,
                "ttf_answer_ms": ttf_answer,
                "ttlt_ms": ttlt_thinking,
                "total_ms": ttlt,
            },
            "tool_call_count": len(self.tool_calls),
            "aborted": self.aborted,
            "error": self.error_message or None,
            "throughput": {"output_tokens_per_sec": otps},
            "events_count": self.total_events,
        }



class ConversationLog:
    __slots__ = ('log_dir', 'run_id', 'session_id', 'session_start', 'log_file', 'turns', 'turn_counter',
                 'total_input_tokens', 'total_output_tokens', 'total_cached',
                 'total_reasoning', 'total_latency_ms', 'models_used',
                 '_sum_server', '_sum_ttfb', '_sum_ttlt', '_sum_otps',
                 '_sum_ttf_thinking', '_sum_ttf_answer',
                 '_cnt_server', '_cnt_ttfb', '_cnt_ttlt', '_cnt_otps',
                 '_cnt_ttf_thinking', '_cnt_ttf_answer',
                 '_last_save_ts', '_last_save_turn', 'total_tool_calls',
                 'cli_version', 'codex_home', 'aborted_turns')

    def __init__(self, log_dir: Path, run_id: str = "", codex_home: str = ""):
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
        self.total_cached = 0
        self.total_reasoning = 0
        self.total_latency_ms = 0.0
        self.models_used: set = set()

        self._sum_server = self._sum_ttfb = self._sum_ttlt = self._sum_otps = 0.0
        self._sum_ttf_thinking = self._sum_ttf_answer = 0.0
        self._cnt_server = self._cnt_ttfb = self._cnt_ttlt = self._cnt_otps = 0
        self._cnt_ttf_thinking = self._cnt_ttf_answer = 0
        self._last_save_ts = 0.0
        self._last_save_turn = 0
        self.total_tool_calls = 0
        self.aborted_turns = 0
        # None when codex is not on PATH or `codex --version` is unparseable.
        self.cli_version: Optional[str] = get_cli_version("codex")
        # CODEX_HOME's post-rename per-run path (codex_home/<run_id>/), where the
        # native session rollout transcripts are preserved after shutdown. None
        # when --no-codex-home was set or the option was not passed.
        self.codex_home: Optional[str] = codex_home or None

    def _init_session(self, response_id: str):
        if self.session_id:
            return
        self.session_id = response_id[5:13] if response_id.startswith("resp_") else (response_id[:8] or "unknown")
        self.session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / "trace.json"

    def add_turn(self, metrics: TurnMetrics) -> tuple[bool, str]:
        """Add a turn and maybe save. Returns (saved, filename) if intermediate save occurred."""
        self._init_session(metrics.response_id)
        self.turn_counter += 1
        metrics.turn_id = self.turn_counter

        turn_data = metrics.to_dict()
        self.turns.append(turn_data)
        self.models_used.add(turn_data["model"])

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
        self.total_cached += tokens["cached"]
        self.total_reasoning += tokens["reasoning"]
        self.total_tool_calls += turn_data["tool_call_count"]

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
        if latency["total_ms"]:
            self._sum_ttlt += latency["total_ms"]
            self._cnt_ttlt += 1
            self.total_latency_ms += latency["total_ms"]
        if throughput["output_tokens_per_sec"]:
            self._sum_otps += throughput["output_tokens_per_sec"]
            self._cnt_otps += 1

        return self._maybe_save()

    def _compute_summary(self) -> dict:
        return {
            "total_turns": len(self.turns),
            "aborted_turns": self.aborted_turns,
            "total_tool_calls": self.total_tool_calls,
            "models_used": sorted(self.models_used),
            "tokens": {
                "total_input": self.total_input_tokens,
                "total_output": self.total_output_tokens,
                "total_cached": self.total_cached,
                "total_reasoning": self.total_reasoning,
            },
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
        }

    def _save(self) -> bool:
        if not self.log_file:
            return False
        try:
            data = {
                "session": {
                    "id": self.session_id,
                    "started_at": self.session_start,
                    "exported_at": datetime.now().isoformat(),
                    "extractor_version": VERSION,
                    "cli_version": self.cli_version,
                    "codex_home": self.codex_home,
                },
                "summary": self._compute_summary(),
                "turns": self.turns,
            }
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
            f"Session: {self.session_id}",
            f"  Turns: {summary['total_turns']}",
            f"  Models: {', '.join(summary['models_used'])}",
            f"  Tokens: {tok['total_input']:,} in / {tok['total_output']:,} out",
            f"  Avg TTFB: {lat['avg_ttfb_ms']:.0f}ms",
            f"  Total time: {lat['total_ms']/1000:.1f}s",
        ]

        return "\n".join(lines)


try:
    from mitmproxy import http, ctx

    class CodexLatencyAddon:
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
            loader.add_option(name="codex_home", typespec=str, default="",
                              help="Per-run path under codex_home/ where Codex's raw state (including session rollout transcripts) ends up after the shutdown rename (codex_home/<run_id>/). Recorded as session.codex_home in the JSON trace. Differs from the CODEX_HOME literal in the launch command, which uses the stable runtime path codex_home/active_<port>/.")

        def configure(self, updates):
            # Only initialize once: a later runtime option change must not discard
            # the in-flight session's accumulated turns.
            if self.conversation_log is not None:
                return
            if "latency_log_dir" in updates or "run_id" in updates or "codex_home" in updates:
                log_dir = ctx.options.latency_log_dir
                if log_dir:
                    self.conversation_log = ConversationLog(Path(log_dir), run_id=ctx.options.run_id, codex_home=ctx.options.codex_home)

        def done(self):
            """Called when proxy shuts down (Ctrl+C)."""
            if self._shutdown_flag:
                return
            self._shutdown_flag = True

            # Tracked turns still in flight at shutdown were never recorded (e.g. a
            # mid-turn Ctrl+C). Surface the count rather than dropping them silently.
            if self.active_requests:
                ctx.log.warn(f"[LOG] {len(self.active_requests)} request(s) still in flight at shutdown (not recorded)")

            if self.conversation_log:
                saved, filename = self.conversation_log.save()
                if saved:
                    ctx.log.info(f"[LOG] Final save completed → {filename}")

            print()
            if self.conversation_log and self.conversation_log.log_file and self.conversation_log.turns:
                print(f"Trace saved to: {self.conversation_log.log_file}")
            else:
                print("No conversation data captured.")

        def request(self, flow: http.HTTPFlow):
            req = flow.request
            if not any(host in req.host for host in CODEX_API_HOSTS):
                return

            # Recent Codex versions default to WebSocket for /responses; force HTTP instead.
            # 426 on the WebSocket upgrade triggers Codex's built-in fallback to HTTP POST + SSE.
            if (req.method == "GET" and req.path.endswith("/responses")
                    and req.headers.get("upgrade", "").lower() == "websocket"):
                flow.response = http.Response.make(426, b"", {})
                return

            if req.method != "POST" or not req.path.endswith("/responses"):
                return

            try:
                body = json_loads(req.content)
                if not body.get("stream"):
                    return

                if not self._session_started:
                    self._session_started = True
                    ctx.log.info("Codex CLI session detected!")

                reasoning = body.get("reasoning", {})
                input_messages = body.get("input") or []
                # Tool definitions arrive top-level or, in the responses-lite
                # request shape, as an additional_tools input item of namespaces.
                tools = body.get("tools") or [
                    tool
                    for item in input_messages
                    if isinstance(item, dict) and item.get("type") == "additional_tools"
                    for namespace in item.get("tools") or []
                    if isinstance(namespace, dict)
                    for tool in namespace.get("tools") or [namespace]
                ]

                metrics = TurnMetrics()
                metrics.timestamp = datetime.now().isoformat()
                metrics.model = body.get("model", "unknown")
                metrics.request_start = time.perf_counter()
                metrics.messages_count = len(input_messages)
                metrics.request_size_bytes = len(req.content) if req.content else 0
                metrics.reasoning_effort = reasoning.get("effort", "") if isinstance(reasoning, dict) else ""
                metrics.tools_count = len(tools)
                metrics.prompt_cache_key = body.get("prompt_cache_key", "")

                # Extract user message from input array
                for msg in reversed(input_messages):
                    if msg.get("type") == "message" and msg.get("role") == "user":
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for c in content:
                                if c.get("type") == "input_text":
                                    metrics.user_message = c.get("text", "")
                                    break
                        break

                # System prompt: top-level instructions, or the developer
                # messages the responses-lite request shape carries inside input.
                metrics.instructions = body.get("instructions") or "\n\n".join(
                    " ".join(c.get("text", "") for c in msg.get("content", []) if isinstance(c, dict) and c.get("type") == "input_text")
                    for msg in input_messages
                    if isinstance(msg, dict) and msg.get("type") == "message" and msg.get("role") in ("developer", "system")
                )

                self.active_requests[flow.id] = metrics
                ctx.log.info(f"[TURN] Started: {metrics.model}")

            except Exception as e:
                ctx.log.warn(f"[TURN] Parse error: {e}")

        def responseheaders(self, flow: http.HTTPFlow):
            metrics = self.active_requests.get(flow.id)
            if not metrics:
                return

            metrics.first_byte = time.perf_counter()
            metrics.request_id = flow.response.headers.get("x-oai-request-id", "")

            # Forward SSE chunks to the client as they arrive instead of mitmproxy's
            # default buffer-until-complete, so the client is not blocked until the
            # upstream response finishes. The tap copies each chunk for metric
            # parsing while passing it through unchanged. Non-SSE responses stay
            # buffered.
            h = flow.response.headers
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
                content = flow.response.content
            if content:
                metrics.response_size_bytes = len(content)
                self._parse_sse(content, metrics)

            # A streamed flow that reached the response hook but never produced a
            # terminal response.completed (status stays empty) ended early (e.g.
            # upstream closed mid-stream, delivered here rather than to the error
            # hook). Flag it aborted so it is recorded but kept out of the averages.
            if metrics.was_streamed and (metrics.total_events > 0 or metrics.response_size_bytes > 0) and not metrics.status:
                metrics.aborted = True
                metrics.error_message = "stream ended before completion (no response.completed)"

            if self.conversation_log and (
                metrics.total_events > 0 or metrics.response_size_bytes > 0 or metrics.output_tokens > 0
            ):
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
            metrics.error_message = str(flow.error) if flow.error else "stream aborted"
            if metrics.was_streamed and metrics.stream_buf:
                content = _decode_sse_body(metrics.stream_buf, metrics.content_encoding)
                if content:
                    metrics.response_size_bytes = len(content)
                    self._parse_sse(content, metrics)

            if self.conversation_log:
                saved, filename = self.conversation_log.add_turn(metrics)
                if saved:
                    ctx.log.info(f"[LOG] Aborted turn recorded → {filename}")

        def _parse_sse(self, content: bytes, m: TurnMetrics):
            try:
                text = content.decode("utf-8", errors="ignore")
            except Exception:
                return

            current_tool_call = None
            current_tool_args = []

            for data_str in _iter_sse_data(text):
                if not data_str or data_str == "[DONE]":
                    continue

                m.total_events += 1

                try:
                    data = json_loads(data_str)
                    etype = data.get("type", "")
                    seq = data.get("sequence_number", 0)
                    _bump(m.event_types, etype)

                    if etype == "response.created":
                        resp = data.get("response", {})
                        m.response_id = resp.get("id", "")
                        m.model = resp.get("model", m.model)
                        m.service_tier = resp.get("service_tier", "")
                        m.created_at = resp.get("created_at")

                    elif etype == "response.output_item.added":
                        item = data.get("item", {})
                        item_type = item.get("type", "")
                        _bump(m.item_types, item_type)
                        if item_type == "reasoning":
                            if m.first_reasoning_seq is None:
                                m.first_reasoning_seq = seq
                        elif item_type == "message":
                            if m.first_text_seq is None:
                                m.first_text_seq = seq
                        elif item_type in ("function_call", "custom_tool_call"):
                            current_tool_call = {
                                "name": item.get("name", ""),
                                "call_id": item.get("call_id", ""),
                                "arguments": ""
                            }
                            current_tool_args = []
                        elif item_type == "local_shell_call":
                            current_tool_call = {
                                "name": "local_shell",
                                "call_id": item.get("call_id", ""),
                                "arguments": item.get("action", {})
                            }
                            current_tool_args = []

                    elif etype == "response.output_item.done":
                        item = data.get("item", {})
                        item_type = item.get("type", "")
                        if item_type == "reasoning":
                            m.last_reasoning_seq = seq
                            # Extract reasoning summary
                            summary_parts = item.get("summary", [])
                            if summary_parts:
                                m.reasoning_summary = " ".join(
                                    s.get("text", "") for s in summary_parts if s.get("type") == "summary_text"
                                )
                        elif item_type == "function_call" and current_tool_call:
                            current_tool_call["arguments"] = "".join(current_tool_args)
                            try:
                                current_tool_call["arguments"] = json_loads(current_tool_call["arguments"])
                            except Exception:
                                pass
                            m.tool_calls.append(current_tool_call)
                            current_tool_call = None
                            current_tool_args = []
                        elif item_type == "custom_tool_call" and current_tool_call:
                            # Freeform tools (exec, apply_patch) carry their input as
                            # plain text; the done item has the full string.
                            current_tool_call["arguments"] = item.get("input") or "".join(current_tool_args)
                            m.tool_calls.append(current_tool_call)
                            current_tool_call = None
                            current_tool_args = []
                        elif item_type == "local_shell_call" and current_tool_call:
                            current_tool_call["arguments"] = item.get("action", current_tool_call["arguments"])
                            m.tool_calls.append(current_tool_call)
                            current_tool_call = None
                            current_tool_args = []

                    elif etype == "response.reasoning_summary_text.delta":
                        # We'll get the full summary in output_item.done
                        pass

                    elif etype == "response.output_text.delta":
                        delta = data.get("delta", "")
                        m.text_parts.append(delta)

                    elif etype in ("response.function_call_arguments.delta", "response.custom_tool_call_input.delta"):
                        delta = data.get("delta", "")
                        current_tool_args.append(delta)

                    elif etype == "response.completed":
                        resp = data.get("response", {})
                        m.status = resp.get("status", "")
                        m.incomplete_details = resp.get("incomplete_details")
                        m.completed_at = resp.get("completed_at")
                        m.service_tier = resp.get("service_tier", m.service_tier)

                        # Extract usage data
                        usage = resp.get("usage", {})
                        if usage:
                            m.input_tokens = usage.get("input_tokens", 0)
                            m.output_tokens = usage.get("output_tokens", 0)
                            m.total_tokens = usage.get("total_tokens", 0)

                            input_details = usage.get("input_tokens_details") or {}
                            m.cached_tokens = input_details.get("cached_tokens", 0)
                            m.cache_write_tokens = input_details.get("cache_write_tokens", 0)

                            output_details = usage.get("output_tokens_details") or {}
                            m.reasoning_tokens = output_details.get("reasoning_tokens", 0)

                except Exception:
                    pass

    addons = [CodexLatencyAddon()]

except ImportError:
    addons = []


def main():
    import argparse
    import subprocess
    import shutil

    parser = argparse.ArgumentParser(description="Codex Trace Extractor")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Proxy port (default: 8080)")
    if MODEL_REQUIRED:
        parser.add_argument("--model", "-m", required=True, help="Model name to use (e.g., gpt-5.5)")
    else:
        parser.add_argument("--model", "-m", default=CODEX_DEFAULT_MODEL, help=f"Model name to use (default: {CODEX_DEFAULT_MODEL})")
    if CONFIG_REQUIRED:
        parser.add_argument("--reasoning-effort", "-r", required=True, help="Reasoning effort (e.g., medium, high, xhigh)")
    else:
        parser.add_argument("--reasoning-effort", "-r", default=CODEX_DEFAULT_REASONING_EFFORT, help=f"Reasoning effort (default: {CODEX_DEFAULT_REASONING_EFFORT})")
    parser.add_argument("--require-cli-version", metavar="X.Y.Z", default=None, help="Require an exact installed version of codex before running. Prints an install command and exits 1 if the installed version doesn't match or codex isn't on PATH. Detection runs `codex --version`.")
    parser.add_argument("--no-codex-home", action="store_true", help="Disable the CODEX_HOME redirect. By default the launch command sets CODEX_HOME to ./codex_home/active_<port>/ so Codex's native session rollout transcripts are written there during the run, then copied into ./logs/<run_id>/ at extractor shutdown; the runtime dir is renamed to ./codex_home/<run_id>/ so the next run starts with an empty active_<port>/. The redirected home is seeded with a minimal config.toml (the LLM-proxy openai_base_url). With this flag, no redirect happens and Codex uses the worker's regular ~/.codex/ instead (no transcript collection).")
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
        err = enforce_cli_version("codex", args.require_cli_version)
        if err:
            print(err)
            sys.exit(1)

    installed_cli_version = get_cli_version("codex")
    installed_cli_str = f"codex {installed_cli_version}" if installed_cli_version else "codex (not detected)"

    script_dir = Path(__file__).parent.resolve()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_dir = script_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    system, is_wsl = get_platform_info()
    platform_str = f"{system}" + (" (WSL)" if is_wsl else "")

    # Two-path design (see main.py): the runtime path goes in the launch command
    # (stable per port); the final path is the rename target at shutdown and what
    # the JSON trace records as session.codex_home.
    codex_home_runtime: Optional[Path] = None
    codex_home_final: Optional[Path] = None
    codex_home_row = ""
    if not args.no_codex_home:
        codex_home_runtime = script_dir / "codex_home" / f"active_{args.port}"
        codex_home_runtime.mkdir(parents=True, exist_ok=True)
        seed_codex_config(codex_home_runtime)
        codex_home_final = script_dir / "codex_home" / run_id
        codex_home_row = f"  Codex home:     {codex_home_runtime}\n"

    launch_cmd = build_launch_command(cert_path, args.port, "codex", args.model, reasoning_effort=args.reasoning_effort, codex_home=codex_home_runtime)

    copied = copy_to_clipboard(launch_cmd)
    clipboard_msg = "(copied to clipboard)" if copied else "(copy the command below)"

    sep = "=" * 70
    print(f"""
{sep}
  CODEX CLI TRACE EXTRACTOR v{VERSION}
{sep}
  Platform:       {platform_str}
  Installed CLI:  {installed_cli_str}
  Proxy:          http://127.0.0.1:{args.port}
  Run dir:        {run_dir}
{codex_home_row}{sep}

  Run this in your project directory {clipboard_msg}:

  {launch_cmd}

{sep}
  Waiting for Codex CLI to connect...
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
    if codex_home_final is not None:
        mitmdump_args += ["--set", f"codex_home={codex_home_final}"]
    mitmdump_args.append("--ssl-insecure")
    proc = subprocess.Popen(mitmdump_args)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        proc.terminate()
        proc.wait()

    print("\nProxy stopped.")

    preserved_dir: Optional[Path] = codex_home_final
    if codex_home_runtime is not None:
        srcs = sorted(codex_home_runtime.glob("sessions/**/rollout-*.jsonl"))
        if len(srcs) == 1:
            shutil.copy2(srcs[0], run_dir / "rollout.jsonl")
        elif len(srcs) > 1:
            for i, p in enumerate(srcs, 1):
                shutil.copy2(p, run_dir / f"rollout_{i}.jsonl")
        try:
            codex_home_runtime.rename(codex_home_final)
        except OSError as e:
            print(f"WARNING: could not rename {codex_home_runtime} -> {codex_home_final}: {e}")
            preserved_dir = codex_home_runtime

    def _order_key(p):
        name = p.name
        if name == "trace.json": return (0, name)
        if name == "transcript.txt": return (1, name)
        if name.startswith("rollout") and name.endswith(".jsonl"): return (2, name)
        return (3, name)
    files = sorted(
        (p for p in run_dir.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=_order_key,
    )
    if files:
        print(f"\nFiles to share for this run ({run_dir}):")
        for p in files:
            print(f"  {p.name}")
        if preserved_dir is not None and preserved_dir.exists():
            print(f"\n(Originals and additional Codex state preserved at {preserved_dir})")


if __name__ == "__main__":
    main()
