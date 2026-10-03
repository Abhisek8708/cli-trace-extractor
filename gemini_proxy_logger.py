"""Gemini CLI Trace Extractor - captures conversations with latency metrics."""

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
    VERSION,
    MODEL_REQUIRED,
    GEMINI_DEFAULT_MODEL,
)

def json_loads(s): return orjson.loads(s)
def json_dumps(obj): return orjson.dumps(obj, option=orjson.OPT_INDENT_2)

SAVE_EVERY_TURNS = 5
SAVE_EVERY_SECS = 5.0

# Supported API hosts for Gemini
GEMINI_API_HOSTS = ("generativelanguage.googleapis.com", "dataannotation.tech", "cloudcode-pa.googleapis.com")

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


def _iter_sse_data(text: str):
    """Iterate over SSE data chunks, handling multi-line data."""
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

        # Request info
        self.contents_count = 0
        self.user_message = ""
        self.system_instruction = ""
        self.request_size_bytes = 0
        self.max_output_tokens = 0
        self.thinking_budget = 0
        self.tools_count = 0

        # Timing
        self.request_start = 0.0
        self.first_byte = None
        self.stream_end = None

        # Token counts
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.thoughts_tokens = 0
        self.cached_tokens = 0
        self.tool_use_tokens = 0

        # Response content
        self.thinking_parts = []
        self.text_parts = []
        self.function_calls = []
        self.finish_reason = ""
        self.response_size_bytes = 0
        self.model_version = ""
        self.traffic_type = ""  # ON_DEMAND or PROVISIONED_THROUGHPUT

        # Event tracking
        self.total_events = 0
        self.first_thinking_idx = None
        self.first_text_idx = None

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

        # Estimate timing based on event indices
        ttf_thinking = ttf_answer = None
        if self.total_events > 0 and ttfb and stream_dur > 0:
            ms_per_event = (stream_dur * 1000) / self.total_events
            if self.first_thinking_idx is not None:
                ttf_thinking = round(ttfb + (self.first_thinking_idx * ms_per_event), 1)
            if self.first_text_idx is not None:
                ttf_answer = round(ttfb + (self.first_text_idx * ms_per_event), 1)

        output_tokens = self.output_tokens
        otps = round(output_tokens / (ttlt / 1000), 1) if ttlt and ttlt > 0 and output_tokens > 0 else None
        total_input = self.input_tokens + self.cached_tokens

        return {
            "turn": self.turn_id,
            "timestamp": self.timestamp,
            "response_id": self.response_id,
            "model": self.model,
            "model_version": self.model_version,
            "traffic_type": self.traffic_type,
            "request": {
                "contents_count": self.contents_count,
                "user_message": self.user_message,
                "system_instruction": self.system_instruction,
                "size_bytes": self.request_size_bytes,
                "max_output_tokens": self.max_output_tokens,
                "thinking_budget": self.thinking_budget,
                "tools_count": self.tools_count,
            },
            "response": {
                "thinking": "".join(self.thinking_parts),
                "text": "".join(self.text_parts),
                "function_calls": self.function_calls,
                "finish_reason": self.finish_reason,
                "size_bytes": self.response_size_bytes,
            },
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "thoughts": self.thoughts_tokens,
                "cached": self.cached_tokens,
                "tool_use": self.tool_use_tokens,
                "total": self.total_tokens,
                "total_input": total_input,
            },
            "latency": {
                "ttfb_ms": ttfb,
                "ttf_thinking_ms": ttf_thinking,
                "ttf_answer_ms": ttf_answer,
                "ttlt_ms": ttlt,
            },
            "tool_call_count": len(self.function_calls),
            "aborted": self.aborted,
            "error": self.error_message or None,
            "throughput": {"output_tokens_per_sec": otps},
            "events_count": self.total_events,
        }


class ConversationLog:
    __slots__ = ('log_dir', 'run_id', 'session_id', 'session_start', 'log_file', 'turns', 'turn_counter',
                 'total_input_tokens', 'total_output_tokens', 'total_cached',
                 'total_thoughts', 'total_latency_ms', 'models_used',
                 '_sum_ttfb', '_sum_ttlt', '_sum_otps',
                 '_sum_ttf_thinking', '_sum_ttf_answer',
                 '_cnt_ttfb', '_cnt_ttlt', '_cnt_otps',
                 '_cnt_ttf_thinking', '_cnt_ttf_answer',
                 '_last_save_ts', '_last_save_turn', 'total_tool_calls',
                 'cli_version', 'aborted_turns')

    def __init__(self, log_dir: Path, run_id: str = ""):
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
        self.total_thoughts = 0
        self.total_latency_ms = 0.0
        self.models_used: set = set()

        self._sum_ttfb = self._sum_ttlt = self._sum_otps = 0.0
        self._sum_ttf_thinking = self._sum_ttf_answer = 0.0
        self._cnt_ttfb = self._cnt_ttlt = self._cnt_otps = 0
        self._cnt_ttf_thinking = self._cnt_ttf_answer = 0
        self._last_save_ts = 0.0
        self._last_save_turn = 0
        self.total_tool_calls = 0
        self.aborted_turns = 0
        # None when gemini is not on PATH or `gemini --version` is unparseable.
        self.cli_version: Optional[str] = get_cli_version("gemini")

    def _init_session(self, response_id: str):
        if self.session_id:
            return
        # Extract a short session ID from response_id or generate one
        self.session_id = response_id[:12] if response_id else datetime.now().strftime("%H%M%S%f")[:8]
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
        self.total_thoughts += tokens["thoughts"]
        self.total_tool_calls += turn_data["tool_call_count"]

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
                "total_thoughts": self.total_thoughts,
            },
            "latency": {
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

    class GeminiLatencyAddon:
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

        def configure(self, updates):
            # Only initialize once: a later runtime option change must not discard
            # the in-flight session's accumulated turns.
            if self.conversation_log is not None:
                return
            if "latency_log_dir" in updates or "run_id" in updates:
                log_dir = ctx.options.latency_log_dir
                if log_dir:
                    self.conversation_log = ConversationLog(Path(log_dir), run_id=ctx.options.run_id)

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

            # Filter for Gemini API requests
            if not any(host in req.host for host in GEMINI_API_HOSTS):
                return

            # Only capture streaming generateContent requests
            if req.method != "POST" or ":streamGenerateContent" not in req.path:
                return

            try:
                body = json_loads(req.content)

                if not self._session_started:
                    self._session_started = True
                    ctx.log.info("Gemini CLI session detected!")

                # Request data location: top-level (standard API) or nested in "request" (OAuth)
                request_data = body.get("request", body)
                contents = request_data.get("contents", [])
                tools = request_data.get("tools", [])
                gen_config = request_data.get("generationConfig", {})
                thinking_config = gen_config.get("thinkingConfig", {})

                metrics = TurnMetrics()
                metrics.timestamp = datetime.now().isoformat()
                # Model in URL path (standard API) or request body (OAuth)
                if "/models/" in req.path:
                    metrics.model = req.path.split("/models/")[-1].split(":")[0]
                else:
                    metrics.model = body.get("model", "unknown")
                metrics.request_start = time.perf_counter()
                metrics.contents_count = len(contents)
                metrics.request_size_bytes = len(req.content) if req.content else 0
                metrics.max_output_tokens = gen_config.get("maxOutputTokens", 0)
                metrics.thinking_budget = thinking_config.get("thinkingBudget", 0)

                # Count tools (function declarations)
                for tool in tools:
                    if "functionDeclarations" in tool:
                        metrics.tools_count += len(tool["functionDeclarations"])

                # Extract user message from contents
                for content in reversed(contents):
                    if content.get("role") == "user":
                        parts = content.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                metrics.user_message = part["text"]
                                break
                        if metrics.user_message:
                            break

                # Extract system instruction
                system_inst = request_data.get("systemInstruction", {})
                if system_inst:
                    parts = system_inst.get("parts", [])
                    for part in parts:
                        if "text" in part:
                            metrics.system_instruction = part["text"]
                            break

                self.active_requests[flow.id] = metrics
                ctx.log.info(f"[TURN] Started: {metrics.model}")

            except Exception as e:
                ctx.log.warn(f"[TURN] Parse error: {e}")

        def responseheaders(self, flow: http.HTTPFlow):
            metrics = self.active_requests.get(flow.id)
            if not metrics:
                return

            metrics.first_byte = time.perf_counter()

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
            # terminal finishReason ended early (e.g. upstream closed mid-stream,
            # delivered here rather than to the error hook). Flag it aborted so it
            # is recorded but kept out of the averages.
            if metrics.was_streamed and (metrics.total_events > 0 or metrics.response_size_bytes > 0) and not metrics.finish_reason:
                metrics.aborted = True
                metrics.error_message = "stream ended before completion (no finishReason)"

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

            event_idx = 0

            for data_str in _iter_sse_data(text):
                if not data_str or data_str == "[DONE]":
                    continue

                m.total_events += 1

                try:
                    data = json_loads(data_str)

                    # Response data location: top-level (standard API) or nested in "response" (OAuth)
                    response_data = data.get("response", data)

                    # Extract response ID
                    if not m.response_id and "responseId" in response_data:
                        m.response_id = response_data["responseId"]

                    # Extract model version
                    if not m.model_version and "modelVersion" in response_data:
                        m.model_version = response_data["modelVersion"]

                    # Parse candidates
                    candidates = response_data.get("candidates", [])
                    for candidate in candidates:
                        # Extract finish reason
                        if "finishReason" in candidate:
                            m.finish_reason = candidate["finishReason"]

                        # Parse content parts
                        content_obj = candidate.get("content", {})
                        parts = content_obj.get("parts", [])

                        for part in parts:
                            # Check for thinking content (thought: true)
                            if part.get("thought") is True:
                                if m.first_thinking_idx is None:
                                    m.first_thinking_idx = event_idx
                                if "text" in part:
                                    m.thinking_parts.append(part["text"])

                            # Check for regular text content
                            elif "text" in part:
                                if m.first_text_idx is None:
                                    m.first_text_idx = event_idx
                                m.text_parts.append(part["text"])

                            # Check for function calls
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                m.function_calls.append({
                                    "name": fc.get("name", ""),
                                    "args": fc.get("args", {})
                                })

                    # Parse usage metadata (typically in final event)
                    usage = response_data.get("usageMetadata", {})
                    if usage:
                        m.input_tokens = usage.get("promptTokenCount", m.input_tokens)
                        m.output_tokens = usage.get("candidatesTokenCount", m.output_tokens)
                        m.total_tokens = usage.get("totalTokenCount", m.total_tokens)
                        m.thoughts_tokens = usage.get("thoughtsTokenCount", m.thoughts_tokens)
                        m.cached_tokens = usage.get("cachedContentTokenCount", m.cached_tokens)
                        m.tool_use_tokens = usage.get("toolUsePromptTokenCount", m.tool_use_tokens)
                        if "trafficType" in usage:
                            m.traffic_type = usage["trafficType"]

                except Exception:
                    pass

                event_idx += 1

    addons = [GeminiLatencyAddon()]

except ImportError:
    addons = []


def main():
    import argparse
    import subprocess
    import shutil

    parser = argparse.ArgumentParser(description="Gemini Trace Extractor")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Proxy port (default: 8080)")
    if MODEL_REQUIRED:
        parser.add_argument("--model", "-m", required=True, help="Model name to use (e.g., gemini-3.1-pro-preview)")
    else:
        parser.add_argument("--model", "-m", default=GEMINI_DEFAULT_MODEL, help=f"Model name to use (default: {GEMINI_DEFAULT_MODEL})")
    parser.add_argument("--require-cli-version", metavar="X.Y.Z", default=None, help="Require an exact installed version of gemini before running. Prints an install command and exits 1 if the installed version doesn't match or gemini isn't on PATH. Detection runs `gemini --version`.")
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
        err = enforce_cli_version("gemini", args.require_cli_version)
        if err:
            print(err)
            sys.exit(1)

    installed_cli_version = get_cli_version("gemini")
    installed_cli_str = f"gemini {installed_cli_version}" if installed_cli_version else "gemini (not detected)"

    script_dir = Path(__file__).parent.resolve()
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    run_dir = script_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    system, is_wsl = get_platform_info()
    platform_str = f"{system}" + (" (WSL)" if is_wsl else "")

    launch_cmd = build_launch_command(cert_path, args.port, "gemini", args.model)

    copied = copy_to_clipboard(launch_cmd)
    clipboard_msg = "(copied to clipboard)" if copied else "(copy the command below)"

    sep = "=" * 70
    print(f"""
{sep}
  GEMINI CLI TRACE EXTRACTOR v{VERSION}
{sep}
  Platform:       {platform_str}
  Installed CLI:  {installed_cli_str}
  Proxy:          http://127.0.0.1:{args.port}
  Run dir:        {run_dir}
{sep}

  Run this in your project directory {clipboard_msg}:

  {launch_cmd}

{sep}
  Waiting for Gemini CLI to connect...
  When done: exit CLI first, then Ctrl+C here to save.
{sep}
""")

    mitmdump_cmd = shutil.which("mitmdump")
    if not mitmdump_cmd:
        print("ERROR: mitmdump not found. Install mitmproxy: pip install mitmproxy")
        sys.exit(1)

    proc = subprocess.Popen([
        mitmdump_cmd, "-p", str(args.port),
        "-s", str(Path(__file__).resolve()),
        "--set", f"latency_log_dir={run_dir}",
        "--set", f"run_id={run_id}",
        "--ssl-insecure",
    ])

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        proc.terminate()
        proc.wait()

    print("\nProxy stopped.")

    if (run_dir / "trace.json").exists():
        print(f"\nFiles to share for this run ({run_dir}):\n  trace.json")


if __name__ == "__main__":
    main()
