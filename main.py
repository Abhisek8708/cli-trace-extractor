"""Unified CLI Trace Extractor - captures conversations as a JSON trace, with optional tmux-based terminal transcript (--capture-transcript)."""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from cli_utils import (
    build_launch_command,
    copy_to_clipboard,
    enforce_cli_version,
    get_cert_path,
    get_cli_version,
    get_platform_info,
    is_port_free,
    seed_claude_append_system_prompt,
    seed_codex_config,
    zip_subagent_transcripts,
    VERSION,
    MODEL_REQUIRED,
    CONFIG_REQUIRED,
    CLAUDE_DEFAULT_MODEL,
    CODEX_DEFAULT_MODEL,
    GEMINI_DEFAULT_MODEL,
    CLAUDE_DEFAULT_EFFORT_LEVEL,
    CODEX_DEFAULT_REASONING_EFFORT,
    CLAUDE_DEFAULT_PERMISSION_MODE,
    CLAUDE_PERMISSION_MODES,
)
from transcript_capture import DEFAULT_CAPTURE_INTERVAL, TranscriptOrchestrator


def _positive_int(s: str) -> int:
    """argparse type that accepts only integers >= 1."""
    n = int(s)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _print_banner(
    title: str,
    platform_str: str,
    cli_name: str,
    cli_version: str | None,
    port: int,
    run_dir: Path,
    launch_cmd: str,
    clipboard_msg: str,
    transcript: TranscriptOrchestrator | None,
    claude_config_dir: Path | None,
    codex_home: Path | None = None,
) -> None:
    sep = "=" * 70
    transcript_rows = ""
    transcript_note = ""
    if transcript is not None:
        transcript_rows = "\n".join(transcript.banner_rows()) + "\n"
        transcript_note = "  Transcript capture starts automatically when the tmux session is detected.\n"
    config_row = ""
    if claude_config_dir is not None:
        config_row = f"  Claude config:    {claude_config_dir}\n"
    if codex_home is not None:
        config_row = f"  Codex home:       {codex_home}\n"
    installed_cli_str = f"{cli_name} {cli_version}" if cli_version else f"{cli_name} (not detected)"
    print(f"""
{sep}
  {title} TRACE EXTRACTOR v{VERSION}
{sep}
  Platform:         {platform_str}
  Installed CLI:    {installed_cli_str}
  Proxy:            http://127.0.0.1:{port}
  Run dir:          {run_dir}
{config_row}{transcript_rows}{sep}

  Run this in your project directory {clipboard_msg}:

  {launch_cmd}

{sep}
  Waiting for {title} to connect...
{transcript_note}  When done: exit CLI first, then Ctrl+C here to save.
{sep}
""")


def _copy_claude_artifacts(claude_config_dir: Path, run_dir: Path) -> None:
    """Copy Claude Code's session JSONLs and debug logs out of claude_config_dir
    into run_dir as session.jsonl and debug.txt. If multiple of either kind
    exist (e.g., when the worker ran multiple claude invocations within the
    same extractor run, or used `/clear`), each is copied with a numbered
    suffix (session_1.jsonl, session_2.jsonl, ...; debug_1.txt, ...). Silent
    if none are present."""
    def _copy_set(srcs: list[Path], stem: str, ext: str) -> None:
        if len(srcs) == 1:
            shutil.copy2(srcs[0], run_dir / f"{stem}{ext}")
        elif len(srcs) > 1:
            for i, p in enumerate(srcs, 1):
                shutil.copy2(p, run_dir / f"{stem}_{i}{ext}")
    _copy_set(sorted(claude_config_dir.glob("projects/*/*.jsonl")), "session", ".jsonl")
    _copy_set(sorted(claude_config_dir.glob("debug/*.txt")), "debug", ".txt")


def _copy_codex_artifacts(codex_home: Path, run_dir: Path) -> None:
    """Copy Codex's native session rollout transcripts out of codex_home into
    run_dir as rollout.jsonl. Rollouts live at sessions/YYYY/MM/DD/rollout-*.jsonl;
    if multiple exist (e.g., the worker ran codex more than once or resumed within
    the same extractor run), each is copied with a numbered suffix (rollout_1.jsonl,
    rollout_2.jsonl, ...). Silent if none are present."""
    srcs = sorted(codex_home.glob("sessions/**/rollout-*.jsonl"))
    if len(srcs) == 1:
        shutil.copy2(srcs[0], run_dir / "rollout.jsonl")
    elif len(srcs) > 1:
        for i, p in enumerate(srcs, 1):
            shutil.copy2(p, run_dir / f"rollout_{i}.jsonl")


def _print_run_artifacts(run_dir: Path, preserved_dir: Path | None, state_label: str = "state") -> None:
    """Print the run artifacts header workers use to identify files to share.

    Files are ordered most-prominent first: trace.json, transcript.txt,
    session.jsonl and rollout.jsonl (then their numbered variants),
    subagents.zip, debug.txt (then debug_1.txt, ...). Anything unrecognized is
    sorted to the end.
    """
    def _order_key(p: Path) -> tuple[int, str]:
        name = p.name
        if name == "trace.json": return (0, name)
        if name == "transcript.txt": return (1, name)
        if name.startswith("session") and name.endswith(".jsonl"): return (2, name)
        if name.startswith("rollout") and name.endswith(".jsonl"): return (2, name)
        if name == "subagents.zip": return (3, name)
        if name.startswith("debug") and name.endswith(".txt"): return (4, name)
        return (5, name)
    files = sorted(
        (p for p in run_dir.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=_order_key,
    )
    if not files:
        return
    print(f"\nFiles to share for this run ({run_dir}):")
    for p in files:
        print(f"  {p.name}")
    if preserved_dir is not None and preserved_dir.exists():
        print(f"\n(Originals and additional {state_label} preserved at {preserved_dir})")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Trace Extractor - captures Claude Code, Codex CLI, or Gemini CLI conversations as a JSON trace, with optional tmux-based terminal transcript",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto
  python main.py --codex --model gpt-5.5 --reasoning-effort xhigh
  python main.py --gemini --model gemini-3.1-pro-preview
  python main.py --claude --model claude-opus-4-7 --effort-level medium --permission-mode skip -p 8081 --verbose --disable-autoupdate --show-thinking-summaries
  python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto --require-cli-version 2.1.138
  python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto --no-claude-config
  python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto --capture-transcript
  python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto --capture-transcript --capture-interval 60
"""
    )
    parser.add_argument("--version", "-v", action="version", version=f"cli-trace-extractor v{VERSION}")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--claude", action="store_true", help="Log Claude Code conversations")
    mode_group.add_argument("--codex", action="store_true", help="Log Codex CLI conversations")
    mode_group.add_argument("--gemini", action="store_true", help="Log Gemini CLI conversations")

    parser.add_argument("--port", "-p", type=int, default=8080, help="Proxy port (default: 8080)")
    if MODEL_REQUIRED:
        parser.add_argument("--model", "-m", required=True, help="Model name to use")
    else:
        parser.add_argument("--model", "-m", help="Model name to use (uses CLI-specific default if not provided)")
    permission_help = (
        "Claude launch permission mode "
        "(skip = --dangerously-skip-permissions, auto = --permission-mode auto)"
    )
    if CONFIG_REQUIRED:
        parser.add_argument("--effort-level", "-e", help="Thinking effort level for Claude (e.g., medium, high, xhigh)")
        parser.add_argument("--reasoning-effort", "-r", help="Reasoning effort for Codex (e.g., medium, high, xhigh)")
        parser.add_argument("--permission-mode", choices=CLAUDE_PERMISSION_MODES, help=permission_help)
    else:
        parser.add_argument("--effort-level", "-e", default=CLAUDE_DEFAULT_EFFORT_LEVEL, help=f"Thinking effort level for Claude (default: {CLAUDE_DEFAULT_EFFORT_LEVEL})")
        parser.add_argument("--reasoning-effort", "-r", default=CODEX_DEFAULT_REASONING_EFFORT, help=f"Reasoning effort for Codex (default: {CODEX_DEFAULT_REASONING_EFFORT})")
        parser.add_argument("--permission-mode", choices=CLAUDE_PERMISSION_MODES, default=CLAUDE_DEFAULT_PERMISSION_MODE, help=f"{permission_help} (default: {CLAUDE_DEFAULT_PERMISSION_MODE})")
    parser.add_argument(
        "--capture-transcript", action="store_true",
        help="Capture a tmux-based terminal transcript alongside the JSON trace. Requires tmux on PATH.",
    )
    parser.add_argument(
        "--capture-interval", type=_positive_int, default=None,
        help=f"Tmux transcript capture interval in seconds (default: {DEFAULT_CAPTURE_INTERVAL}; requires --capture-transcript).",
    )
    parser.add_argument("--verbose", action="store_true", help="Pass --verbose to Claude (shows full tool calls, results, and thinking inline). Claude only.")
    parser.add_argument("--disable-autoupdate", action="store_true", help="Set DISABLE_AUTOUPDATER=1 on the Claude launch command so Claude Code does not auto-update. Claude only.")
    parser.add_argument("--show-thinking-summaries", action="store_true", help="Add showThinkingSummaries:true to the Claude --settings JSON so summarized thinking text is returned in API responses (and captured in the JSON trace). Claude only.")
    parser.add_argument("--disable-thinking", action="store_true", help="Disable extended thinking for the session: sets alwaysThinkingEnabled:false and env.MAX_THINKING_TOKENS:0 in the Claude --settings JSON (the latter is the authoritative API-level force-off). Default is thinking enabled. Claude only.")
    parser.add_argument("--require-cli-version", metavar="X.Y.Z", default=None, help="Require an exact installed version of the selected CLI (claude/codex/gemini) before running. Prints an install command and exits 1 if the installed version doesn't match or the CLI isn't on PATH. Detection runs `<cli> --version`.")
    parser.add_argument("--no-claude-config", action="store_true", help="Disable the CLAUDE_CONFIG_DIR redirect. By default with --claude, the launch command sets CLAUDE_CONFIG_DIR to ./claude_config/active_<port>/ (a stable path keyed by proxy port so reusing a previously-printed launch command still works). Claude Code's session JSONL and debug log are written there during the run, then copied into ./logs/<run_id>/ at extractor shutdown (per-subagent transcripts are bundled into subagents.zip); the runtime dir is renamed to ./claude_config/<run_id>/ at the same time so the next run starts with an empty active_<port>/. With this flag, no redirect happens and Claude Code uses the worker's regular ~/.claude/ instead. Claude only.")
    parser.add_argument("--no-codex-home", action="store_true", help="Disable the CODEX_HOME redirect. By default with --codex, the launch command sets CODEX_HOME to ./codex_home/active_<port>/ (a stable path keyed by proxy port). Codex's native session rollout transcripts (sessions/**/rollout-*.jsonl) are written there during the run, then copied into ./logs/<run_id>/ at extractor shutdown; the runtime dir is renamed to ./codex_home/<run_id>/ so the next run starts with an empty active_<port>/. The redirected home is seeded with a minimal config.toml (the LLM-proxy openai_base_url). With this flag, no redirect happens and Codex uses the worker's regular ~/.codex/ instead (no transcript collection). Codex only.")
    args = parser.parse_args()

    # Validate required config flags based on selected CLI
    if CONFIG_REQUIRED:
        if args.claude and args.effort_level is None:
            parser.error("--effort-level is required for Claude")
        if args.claude and args.permission_mode is None:
            parser.error("--permission-mode is required for Claude")
        if args.codex and args.reasoning_effort is None:
            parser.error("--reasoning-effort is required for Codex")

    if args.verbose and not args.claude:
        parser.error("--verbose is only supported with --claude")

    if args.disable_autoupdate and not args.claude:
        parser.error("--disable-autoupdate is only supported with --claude")

    if args.show_thinking_summaries and not args.claude:
        parser.error("--show-thinking-summaries is only supported with --claude")

    if args.disable_thinking and not args.claude:
        parser.error("--disable-thinking is only supported with --claude")

    if args.no_claude_config and not args.claude:
        parser.error("--no-claude-config is only supported with --claude")

    if args.no_codex_home and not args.codex:
        parser.error("--no-codex-home is only supported with --codex")

    if args.capture_interval is not None and not args.capture_transcript:
        parser.error("--capture-interval requires --capture-transcript")

    # Determine which logger to use
    if args.claude:
        script_name = "claude_proxy_logger.py"
        cli_name = "claude"
        title = "CLAUDE CODE"
        default_model = CLAUDE_DEFAULT_MODEL
    elif args.codex:
        script_name = "codex_proxy_logger.py"
        cli_name = "codex"
        title = "CODEX CLI"
        default_model = CODEX_DEFAULT_MODEL
    else:
        script_name = "gemini_proxy_logger.py"
        cli_name = "gemini"
        title = "GEMINI CLI"
        default_model = GEMINI_DEFAULT_MODEL

    # Use provided model or fall back to default
    model = args.model if args.model else default_model

    script_dir = Path(__file__).parent.resolve()
    script_path = script_dir / script_name

    if not script_path.exists():
        print(f"ERROR: {script_name} not found in {script_dir}")
        sys.exit(1)

    if not is_port_free(args.port):
        print(f"ERROR: Port {args.port} is in use. Try --port {args.port + 1}")
        sys.exit(1)

    cert_path = get_cert_path()
    if not cert_path.exists():
        print("ERROR: mitmproxy certificate not found.")
        print("Run 'mitmproxy' once to generate certificates, then try again.")
        sys.exit(1)

    mitmdump_cmd = shutil.which("mitmdump")
    if not mitmdump_cmd:
        print("ERROR: mitmdump not found. Install mitmproxy: pip install mitmproxy")
        sys.exit(1)

    # Gate before banner/clipboard so a failed check leaves no side effects.
    if args.require_cli_version is not None:
        err = enforce_cli_version(cli_name, args.require_cli_version)
        if err:
            print(err)
            sys.exit(1)

    # For the banner; each proxy logger detects independently for the JSON trace.
    installed_cli_version = get_cli_version(cli_name)

    # Stable identifier for this extractor run; used as the per-run subdir name
    # under logs/ (and, post-shutdown, under claude_config/ when the Claude
    # redirect is on) so all artifacts from the same run live together and
    # concurrent runs don't collide.
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    # Per-run subdir under logs/. The proxy logger writes trace.json directly
    # here; transcript capture writes transcript.txt here; Claude Code's
    # session JSONL and debug log are copied here at shutdown.
    run_dir = script_dir / "logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # CLAUDE_CONFIG_DIR for Claude. Two paths:
    #   runtime: claude_config/active_<port>/ — stable across runs on the same
    #     port, so reusing a previously-printed launch command still lands
    #     files in the right place. This is the path baked into the launch
    #     command text.
    #   final:   claude_config/<run_id>/ — where the runtime dir is renamed at
    #     shutdown so each run's raw state is preserved at a per-run path. The
    #     proxy logger records this path as session.claude_config_dir in JSON.
    # Default-on for --claude; --no-claude-config opts out (no redirect at all).
    claude_config_runtime: Path | None = None
    claude_config_final: Path | None = None
    if args.claude and not args.no_claude_config:
        claude_config_runtime = script_dir / "claude_config" / f"active_{args.port}"
        claude_config_runtime.mkdir(parents=True, exist_ok=True)
        claude_config_final = script_dir / "claude_config" / run_id

    # Per-model appended system prompt, written next to the run's other
    # per-run state so the launch command can point at it.
    append_system_prompt_file: Path | None = None
    if args.claude:
        append_system_prompt_file = seed_claude_append_system_prompt(claude_config_runtime or run_dir, model)

    # Redirect CODEX_HOME to codex_home/active_<port>/ so Codex's native session
    # rollout transcripts are collected from there at shutdown, then renamed to
    # codex_home/<run_id>/. Default-on for --codex; --no-codex-home opts out.
    codex_home_runtime: Path | None = None
    codex_home_final: Path | None = None
    if args.codex and not args.no_codex_home:
        codex_home_runtime = script_dir / "codex_home" / f"active_{args.port}"
        codex_home_runtime.mkdir(parents=True, exist_ok=True)
        seed_codex_config(codex_home_runtime)
        codex_home_final = script_dir / "codex_home" / run_id

    transcript: TranscriptOrchestrator | None = None
    if args.capture_transcript:
        try:
            transcript = TranscriptOrchestrator.create(run_dir, run_id, args.capture_interval)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    system, is_wsl = get_platform_info()
    platform_str = f"{system}" + (" (WSL)" if is_wsl else "")

    launch_cmd = build_launch_command(
        cert_path, args.port, cli_name, model,
        effort_level=args.effort_level,
        reasoning_effort=args.reasoning_effort,
        permission_mode=args.permission_mode,
        tmux_session=transcript.tmux_session if transcript is not None else None,
        verbose=args.verbose,
        disable_autoupdate=args.disable_autoupdate,
        show_thinking_summaries=args.show_thinking_summaries,
        disable_thinking=args.disable_thinking,
        claude_config_dir=claude_config_runtime,
        codex_home=codex_home_runtime,
        append_system_prompt_file=append_system_prompt_file,
    )

    # Try to copy to clipboard
    copied = copy_to_clipboard(launch_cmd)
    clipboard_msg = "(copied to clipboard)" if copied else "(copy the command below)"

    # Banner shows the runtime path because that's where Claude Code is
    # actually writing during the session. The rename to the final path
    # happens at shutdown.
    _print_banner(title, platform_str, cli_name, installed_cli_version, args.port, run_dir, launch_cmd, clipboard_msg, transcript, claude_config_runtime, codex_home_runtime)

    # Start the capture thread before launching the proxy so it observes the tmux
    # session from the moment it appears.
    if transcript is not None:
        transcript.start()

    mitmdump_args = [
        mitmdump_cmd, "-p", str(args.port),
        "-s", str(script_path),
        "--set", f"latency_log_dir={run_dir}",
        "--set", f"run_id={run_id}",
    ]
    # The JSON field session.claude_config_dir records the post-rename path
    # (claude_config/<run_id>/), which is where the raw state lives after a
    # graceful shutdown — what auditors looking at the trace later care about.
    if claude_config_final is not None:
        mitmdump_args += ["--set", f"claude_config_dir={claude_config_final}"]
    # Record the --disable-thinking launch choice in the trace (session.thinking_disabled).
    # Only the Claude logger defines this option, so passing it to codex/gemini would error.
    if args.claude:
        mitmdump_args += ["--set", f"thinking_disabled={'true' if args.disable_thinking else 'false'}"]
    # Records the post-rename CODEX_HOME path as session.codex_home in the trace.
    # Only the Codex logger defines this option.
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
    finally:
        if transcript is not None:
            transcript.stop()

    print("\nProxy stopped.")

    # Copy session.jsonl/debug.txt and bundle subagents.zip out of the runtime
    # dir, then rename the runtime dir to its per-run name so the next run starts
    # with an empty active_<port>/. Rename is best-effort; on failure, files
    # remain at the runtime path and the shutdown summary points there.
    preserved_dir: Path | None = claude_config_final
    if claude_config_runtime is not None:
        _copy_claude_artifacts(claude_config_runtime, run_dir)
        try:
            zip_subagent_transcripts(claude_config_runtime, run_dir)
        except Exception as e:
            print(f"WARNING: could not bundle subagent transcripts: {e}")
        try:
            claude_config_runtime.rename(claude_config_final)
        except OSError as e:
            print(f"WARNING: could not rename {claude_config_runtime} -> {claude_config_final}: {e}")
            preserved_dir = claude_config_runtime
    if codex_home_runtime is not None:
        preserved_dir = codex_home_final
        _copy_codex_artifacts(codex_home_runtime, run_dir)
        try:
            codex_home_runtime.rename(codex_home_final)
        except OSError as e:
            print(f"WARNING: could not rename {codex_home_runtime} -> {codex_home_final}: {e}")
            preserved_dir = codex_home_runtime

    state_label = "state"
    if claude_config_runtime is not None:
        state_label = "Claude Code state"
    elif codex_home_runtime is not None:
        state_label = "Codex state"
    _print_run_artifacts(run_dir, preserved_dir, state_label)


if __name__ == "__main__":
    main()
