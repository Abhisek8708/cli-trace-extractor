"""Tmux-based terminal transcript capture.

Polls a tmux pane periodically and merges new content into a transcript file,
producing a chronological human-readable record of what was rendered to the terminal.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, NamedTuple


# How often to snapshot the tmux pane (seconds)
DEFAULT_CAPTURE_INTERVAL = 30

# Minimum number of paragraph matches required to consider an overlap valid.
# A single common word/line could falsely match across captures, so require at least 2.
MIN_OVERLAP_PARAGRAPHS = 2

# Characters used in CLI UI chrome (separators, prompt symbols).
# A paragraph consisting only of these (after whitespace strip) is treated as chrome.
CHROME_CHARS = "─═━-—_~❯●"

# Status hints / shortcut labels that appear in CLI UI chrome alongside separators.
# A paragraph that, after stripping CHROME_CHARS, equals or starts with one of these is chrome.
CHROME_PHRASES = ("? for", "esc to", "? for shortcuts", "esc to interrupt")

# Characters that indicate a spinner/in-progress status line.
# Combined with a length cap and an ellipsis to avoid catching real content.
SPINNER_SYMBOLS = "✳✶✻✽✢·"
SPINNER_LINE_MAX_LEN = 30

# Upper bound on how long a single tmux invocation may take before we treat it as failed.
# Prevents the capture thread from hanging if the tmux server is unresponsive.
TMUX_COMMAND_TIMEOUT_SECONDS = 10


# --- tmux helpers ---


def tmux_session_exists(session_name: str) -> bool:
    """Return True if the named tmux session is currently live."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
            timeout=TMUX_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def tmux_capture_pane(session_name: str) -> str | None:
    """Capture the full scrollback + visible screen from a tmux pane as clean text.

    Returns the captured text, or None if the tmux command fails or times out.
    """
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session_name, "-pS", "-"],
            capture_output=True,
            text=True,
            timeout=TMUX_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode == 0:
        return result.stdout
    return None


def tmux_enable_remain_on_exit(session_name: str) -> bool:
    """Set the session's window to stay alive after its program exits.

    With this set, the wrapped CLI's exit leaves the pane in a "dead" state
    instead of tearing down the session, so a final pane capture can include
    content that appeared between the last poll and the exit.
    Best-effort: returns False if tmux fails or times out.
    """
    try:
        result = subprocess.run(
            ["tmux", "set-window-option", "-t", session_name, "remain-on-exit", "on"],
            capture_output=True,
            timeout=TMUX_COMMAND_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def tmux_kill_session(session_name: str) -> bool:
    """Terminate the named session. Best-effort; returns False if tmux fails."""
    try:
        result = subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            capture_output=True,
            timeout=TMUX_COMMAND_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


# --- text normalization & chrome detection ---


def _normalize_text(text: str) -> str:
    """Collapse whitespace and replace non-breaking space (tmux emits \\xa0)."""
    return " ".join(text.replace("\xa0", " ").split())


def _is_chrome(paragraph: str) -> bool:
    """True if a paragraph is transient UI chrome rather than conversation content."""
    s = _normalize_text(paragraph)
    if not s:
        return True

    # Strip chrome characters; if nothing meaningful remains, it's chrome.
    # Handles separator+prompt+separator joined into a single paragraph (no blank line between).
    stripped = s
    for c in CHROME_CHARS:
        stripped = stripped.replace(c, "")
    stripped = stripped.strip()
    if not stripped:
        return True

    remaining = stripped.lower()
    if any(remaining == phrase or remaining.startswith(phrase) for phrase in CHROME_PHRASES):
        return True

    # Spinner/status line: short, leading spinner glyph, contains an ellipsis.
    if len(s) <= SPINNER_LINE_MAX_LEN and any(s.startswith(c) for c in SPINNER_SYMBOLS) and "…" in s:
        return True

    return False


def _is_content_line(line: str) -> bool:
    """True if a single line is conversation content (not chrome or blank)."""
    return not _is_chrome(line)


def _strip_trailing_chrome(lines: list[str]) -> list[str]:
    """Trim trailing chrome/blank lines so the idle prompt block doesn't get persisted."""
    end = len(lines)
    while end > 0 and not _is_content_line(lines[end - 1]):
        end -= 1
    return lines[:end]


def _to_paragraphs(lines: list[str]) -> list[tuple[int, str]]:
    """Group consecutive non-blank lines into paragraphs, dropping chrome.

    Returns list of (start_line_index_in_lines, normalized_paragraph_text).
    Paragraph-level matching is wrap-agnostic: same content at different terminal
    widths normalizes to the same paragraph string.
    """
    paragraphs: list[tuple[int, str]] = []
    current: list[str] = []
    start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            if current:
                text = _normalize_text(" ".join(current))
                if not _is_chrome(text):
                    paragraphs.append((start_idx, text))
                current = []
        else:
            if not current:
                start_idx = i
            current.append(stripped)

    if current:
        text = _normalize_text(" ".join(current))
        if not _is_chrome(text):
            paragraphs.append((start_idx, text))

    return paragraphs


# --- overlap detection ---


OverlapState = Literal["new", "no_new", "none"]


class Overlap(NamedTuple):
    """Result of comparing a fresh capture against the existing transcript.

    state:
        "new"     - capture has new content; new_start_idx is where it begins in capture_lines
        "no_new"  - overlap matched the transcript but capture has nothing new beyond it
        "none"    - no overlap found at all (treat capture as a fresh section)
    """
    state: OverlapState
    new_start_idx: int  # only meaningful when state == "new"
    matched_count: int


def find_overlap(transcript_lines: list[str], capture_lines: list[str]) -> Overlap:
    """Find where capture_lines has new content beyond what's already in the transcript.

    Walks both sequences as paragraphs from the top. For each transcript paragraph,
    searches forward in the capture for a match. Skipping transcript paragraphs that
    don't match (rather than breaking) is what makes this resilient to header/banner
    changes from terminal resizes.
    """
    if not transcript_lines or not capture_lines:
        return Overlap(state="none", new_start_idx=0, matched_count=0)

    transcript_paragraphs = _to_paragraphs(transcript_lines)
    capture_paragraphs = _to_paragraphs(capture_lines)

    if not transcript_paragraphs or not capture_paragraphs:
        return Overlap(state="none", new_start_idx=0, matched_count=0)

    last_matched_line_idx = -1
    capture_search_start = 0
    matched_count = 0
    for _, transcript_text in transcript_paragraphs:
        for j in range(capture_search_start, len(capture_paragraphs)):
            capture_idx, capture_text = capture_paragraphs[j]
            if capture_text == transcript_text:
                last_matched_line_idx = capture_idx
                capture_search_start = j + 1
                matched_count += 1
                break
        # No break on miss: header/banner can change on resize while
        # conversation content remains stable. Keep walking.

    if matched_count < MIN_OVERLAP_PARAGRAPHS or last_matched_line_idx < 0:
        return Overlap(state="none", new_start_idx=0, matched_count=matched_count)

    # Advance past the last matched paragraph (its content lines + trailing blanks).
    new_start = last_matched_line_idx
    while new_start < len(capture_lines) and capture_lines[new_start].strip():
        new_start += 1
    while new_start < len(capture_lines) and not capture_lines[new_start].strip():
        new_start += 1

    if new_start >= len(capture_lines):
        return Overlap(state="no_new", new_start_idx=0, matched_count=matched_count)

    return Overlap(state="new", new_start_idx=new_start, matched_count=matched_count)


# --- merge & I/O ---


MergeKind = Literal["first", "append", "section", "skip", "skip_chrome_only"]


class MergeAction(NamedTuple):
    """What _decide_merge resolved to."""
    kind: MergeKind
    lines_to_write: list[str]
    overlap: Overlap | None  # None for "first"; the overlap result otherwise


def _read_transcript(path: Path) -> list[str]:
    """Read the transcript file and return its lines, or [] if missing or empty."""
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    return []


def _decide_merge(transcript_lines: list[str], capture_lines: list[str]) -> MergeAction:
    """Decide what action to take given the existing transcript and a fresh capture."""
    if not transcript_lines:
        return MergeAction(
            kind="first",
            lines_to_write=_strip_trailing_chrome(capture_lines),
            overlap=None,
        )

    overlap = find_overlap(transcript_lines, capture_lines)

    if overlap.state == "no_new":
        return MergeAction(kind="skip", lines_to_write=[], overlap=overlap)

    if overlap.state == "new":
        new_lines = _strip_trailing_chrome(capture_lines[overlap.new_start_idx:])
        if new_lines and any(_is_content_line(line) for line in new_lines):
            return MergeAction(kind="append", lines_to_write=new_lines, overlap=overlap)
        return MergeAction(kind="skip_chrome_only", lines_to_write=[], overlap=overlap)

    # overlap.state == "none"
    return MergeAction(
        kind="section",
        lines_to_write=_strip_trailing_chrome(capture_lines),
        overlap=overlap,
    )


def _apply_merge(transcript_path: Path, action: MergeAction, ts: str) -> None:
    """Apply a merge action by writing or appending to the transcript file."""
    if action.kind == "first":
        text = "\n".join(action.lines_to_write) + "\n" if action.lines_to_write else ""
        transcript_path.write_text(text, encoding="utf-8")
    elif action.kind == "append":
        with transcript_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(action.lines_to_write) + "\n")
    elif action.kind == "section":
        with transcript_path.open("a", encoding="utf-8") as f:
            f.write(f"\n--- new section {ts} (no overlap with previous capture) ---\n")
            if action.lines_to_write:
                f.write("\n".join(action.lines_to_write) + "\n")
    # "skip" / "skip_chrome_only" -> no file write


def _log_merge(action: MergeAction, transcript_path: Path, ts: str) -> None:
    """Emit one [TRANSCRIPT] log line for actions that wrote to the transcript file."""
    if action.kind in ("first", "append", "section"):
        print(f"[TRANSCRIPT] {ts} Checkpoint saved → {transcript_path.name}")
    # "skip" / "skip_chrome_only" -> no log (no file write happened)


def merge_capture(transcript_path: Path, capture_lines: list[str]) -> None:
    """Decide what to write, apply it, and log if it wrote."""
    transcript_lines = _read_transcript(transcript_path)
    action = _decide_merge(transcript_lines, capture_lines)
    ts = datetime.now().strftime("%H:%M:%S")
    _apply_merge(transcript_path, action, ts)
    _log_merge(action, transcript_path, ts)


# --- capture loop ---


def transcript_capture_loop(
    session_name: str,
    transcript_path: Path,
    stop_event: threading.Event,
    interval: int = DEFAULT_CAPTURE_INTERVAL,
) -> None:
    """Background-thread entry: poll the tmux pane and merge captures into the transcript."""
    prev_raw = ""

    # Wait for the tmux session to be created. Poll quickly so we can enable
    # remain-on-exit before the user has a chance to type /exit (otherwise
    # the session would tear down before our final capture can run).
    SESSION_DETECT_INTERVAL = 0.5
    while not stop_event.is_set():
        if tmux_session_exists(session_name):
            break
        stop_event.wait(SESSION_DETECT_INTERVAL)

    if stop_event.is_set():
        return

    # Keep the session alive past the wrapped command's exit so the final
    # capture can include the post-/exit pane content. Best-effort.
    tmux_enable_remain_on_exit(session_name)

    print(f"[TRANSCRIPT] Capture started (tmux session: {session_name})")

    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break

        if not tmux_session_exists(session_name):
            break

        content = tmux_capture_pane(session_name)
        if not content:
            continue
        if content == prev_raw:
            continue

        try:
            merge_capture(transcript_path, content.splitlines())
        except OSError as e:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[TRANSCRIPT] {ts} Merge failed: {e}")
            continue
        prev_raw = content

    # Final capture on shutdown.
    if tmux_session_exists(session_name):
        content = tmux_capture_pane(session_name)
        if content and content != prev_raw:
            try:
                merge_capture(transcript_path, content.splitlines())
            except OSError as e:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[TRANSCRIPT] {ts} Final merge failed: {e}")


@dataclass
class TranscriptOrchestrator:
    """Owns the lifecycle and metadata for one transcript-capture run.

    Created when transcript capture is enabled; main.py treats this as an opaque
    handle and calls start/stop/report at the right points in the proxy lifecycle.
    """

    run_id: str
    tmux_session: str
    path: Path
    interval: int
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    @classmethod
    def create(cls, run_dir: Path, run_id: str, interval: int | None = None) -> "TranscriptOrchestrator":
        """Validate prerequisites and build an orchestrator for the given run.

        Raises RuntimeError if tmux is not on PATH (transcript capture requires it).
        """
        if not shutil.which("tmux"):
            raise RuntimeError(
                "tmux not found. tmux is required for transcript capture; install it and retry."
            )
        tmux_session = f"trace_{run_id}"
        return cls(
            run_id=run_id,
            tmux_session=tmux_session,
            path=run_dir / "transcript.txt",
            interval=interval if interval is not None else DEFAULT_CAPTURE_INTERVAL,
        )

    def start(self) -> None:
        """Spawn the daemon capture thread."""
        self._thread = threading.Thread(
            target=transcript_capture_loop,
            args=(self.tmux_session, self.path, self._stop_event, self.interval),
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the capture thread to do a final snapshot and exit, then
        terminate the tmux session (which `remain-on-exit` would otherwise
        keep alive after the wrapped CLI exits)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        tmux_kill_session(self.tmux_session)

    def banner_rows(self) -> list[str]:
        """Metadata rows to insert in the runtime banner."""
        return [
            f"  Transcript:       {self.path.name}",
            f"  Tmux session:     {self.tmux_session}",
            f"  Capture interval: {self.interval}s",
        ]
