# CLI Trace Extractor

Captures Claude Code, Codex CLI, or Gemini CLI conversations via MITM proxy. Always produces a machine-readable JSON trace; can optionally produce a human-readable terminal transcript when run with `--capture-transcript`.

## Setup

```bash
uv sync
uv run mitmproxy  # then Ctrl+C (one-time cert setup)
```

`tmux` on PATH is required only when running with `--capture-transcript`. Without that flag, no tmux dependency.

## Usage

```bash
uv run python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto
uv run python main.py --codex --model gpt-5.5 --reasoning-effort xhigh
uv run python main.py --gemini --model gemini-3.1-pro-preview

# With terminal transcript capture (opt-in)
uv run python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto --capture-transcript

# With transcript and a custom capture interval
uv run python main.py --claude --model claude-opus-4-7 --effort-level xhigh --permission-mode auto --capture-transcript --capture-interval 60
```

Copy the command shown and run it in your project directory. With `--capture-transcript`, the launch command starts a tmux session running the CLI and the extractor polls that session in the background to build the transcript.

### Options

- `--version`, `-v` — print extractor version and exit
- `--permission-mode {skip,auto}` — Claude launch permission mode, required (`skip` → `--dangerously-skip-permissions`, `auto` → `--permission-mode auto`).
- `--capture-transcript` — opt in to terminal transcript capture. Wraps the CLI launch in a tmux session and writes a transcript file alongside the JSON trace. Requires `tmux` on PATH.
- `--capture-interval <seconds>` — how often to poll the tmux pane for transcript updates (default 30). Requires `--capture-transcript`.
- `--verbose` — Claude only. Passes `--verbose` to Claude Code so the TUI shows full tool calls, full tool results, and thinking text inline (otherwise these are collapsed).
- `--disable-autoupdate` — Claude only. Sets `DISABLE_AUTOUPDATER=1` on the launch command so Claude Code does not auto-update during the session.
- `--show-thinking-summaries` — Claude only. Adds `showThinkingSummaries:true` to the `--settings` JSON in the launch command so summarized thinking text is returned in API responses and captured in the JSON trace (`turns[].response.thinking`). Without this flag Claude Code does not request summarized thinking text and, in its default modes, that field stays empty; the API's own thinking token count, when it reports one, is recorded as `turns[].tokens.thinking`. The display mode Claude Code actually requested is recorded per turn as `turns[].request.thinking_display`.
- `--disable-thinking` — Claude only. Disable extended thinking for the session. Sets `alwaysThinkingEnabled:false` and `env.MAX_THINKING_TOKENS:0` in the `--settings` JSON. `MAX_THINKING_TOKENS=0` is the authoritative force-off (the bool alone is only a session default the in-session toggle can override); it disables thinking on the Anthropic API regardless of `--effort`. Default (flag absent) is thinking enabled. Note: thinking cannot be disabled on Fable 5 (always on by design).
- `--require-cli-version X.Y.Z` — Require an exact installed version of the selected CLI (claude/codex/gemini). The extractor runs `<cli> --version` on PATH before launching and exits 1 with an install command if the installed version doesn't match or the CLI isn't on PATH. The installed version is also recorded as `session.cli_version` in every JSON trace and shown in the startup banner regardless of whether this flag is set.
- `--no-claude-config` — Claude only. Disable the `CLAUDE_CONFIG_DIR` redirect. Default behavior with `--claude` is to set `CLAUDE_CONFIG_DIR=./claude_config/active_<port>/` on the launch command (and pass `--debug` to Claude) so Claude Code's session JSONL and debug log are written there during the run. At extractor shutdown those files are copied into `./logs/<run_id>/` for ease of collection (per-subagent transcripts are bundled into `subagents.zip`), and the runtime dir is renamed to `./claude_config/<run_id>/` so the next run starts with an empty `active_<port>/`. The `active_<port>/` path is stable across runs on the same proxy port, so reusing a previously-printed launch command still lands files in the right place. With this flag, no redirect is set up and no copy or rename happens; Claude Code uses the worker's regular `~/.claude/` instead.
- `--no-codex-home` — Codex only. Disable the `CODEX_HOME` redirect. Default behavior with `--codex` is to set `CODEX_HOME=./codex_home/active_<port>/` on the launch command so Codex's native session rollout transcripts (`sessions/**/rollout-*.jsonl`) are written there during the run. At extractor shutdown they are copied into `./logs/<run_id>/` as `rollout.jsonl` (numbered if multiple), and the runtime dir is renamed to `./codex_home/<run_id>/` so the next run starts with an empty `active_<port>/`. The redirected home is seeded with a minimal `config.toml` (the LLM-proxy `openai_base_url`). With this flag, no redirect or seeding happens, no transcripts are collected, and Codex uses the worker's regular `~/.codex/` instead.

### Claude config redirect

By default with `--claude`, the launch command sets `CLAUDE_CONFIG_DIR=<extractor>/claude_config/active_<port>/` and passes `--debug` to `claude`. Claude Code writes its session JSONL and debug log under that runtime dir during the session.

The `active_<port>/` path is stable across runs on the same proxy port: every launch command for a given port produces byte-identical `CLAUDE_CONFIG_DIR=...` text, so workers can paste a previously-printed launch command and still have files land in the right place.

At extractor shutdown:

1. The session JSONL is copied to `./logs/<run_id>/session.jsonl` and the debug log to `./logs/<run_id>/debug.txt`, alongside the proxy JSON trace. Per-subagent transcripts (which Claude Code writes under `<session>/subagents/`) are bundled into `./logs/<run_id>/subagents.zip`, always written so the artifact set stays uniform.
2. The runtime dir `./claude_config/active_<port>/` is renamed to `./claude_config/<run_id>/`, preserving the full Claude Code per-run state (`.claude.json`, `backups/`, and the originals of the copied files) at a per-run path for diagnostics. The next extractor run on the same port starts with an empty `active_<port>/`.

The worker's `~/.claude/` is untouched throughout.

Side effects worth knowing:

- User-level plugins, skills, agents, and `~/.claude/settings.json` don't apply during the session. Project-level `.claude/settings.json` and `.claude/settings.local.json` in the worker's cwd still do.
- Each run requires fresh authentication.
- One extractor at a time per proxy port. Two concurrent runs on different ports each get their own `active_<port>/` and don't interfere.
- On a hard-kill of the extractor (terminal closed, `kill -9`, system crash), the shutdown copy + rename do not run. Files remain at `./claude_config/active_<port>/`. Before the next run on that port, either delete `./claude_config/active_<port>/` (to start clean) or accept that its existing contents will be bundled into the next run's output.

Pass `--no-claude-config` to opt out and run against the worker's regular `~/.claude/`.

### Codex session transcripts

By default with `--codex`, the launch command sets `CODEX_HOME=<extractor>/codex_home/active_<port>/`. Codex writes its native per-session **rollout transcript** to `sessions/YYYY/MM/DD/rollout-<iso>-<uuid>.jsonl` under that home, automatically (no flag needed).

Because the redirected home starts empty, the extractor seeds a minimal `config.toml` into it containing only `openai_base_url` (pointing Codex at the LLM proxy). The worker's `OPENAI_API_KEY` must be exported in the shell that runs the launch command.

At extractor shutdown the rollout(s) are copied into `./logs/<run_id>/rollout.jsonl` (numbered `rollout_1.jsonl`, `rollout_2.jsonl`, ... if the worker ran or resumed Codex more than once), and the runtime dir is renamed to `./codex_home/<run_id>/`, preserving the full Codex per-run state. The `active_<port>/` path is stable across runs on the same proxy port.

On a hard-kill of the extractor, the copy and rename don't run; rollouts remain at `./codex_home/active_<port>/` and would be bundled into the next run on that port unless that dir is cleared first (the same caveat as the Claude config redirect).

Pass `--no-codex-home` to opt out (no redirect, no transcript collection) and run against the worker's regular `~/.codex/`.

### Extended request timeout

All Claude launch commands set `API_TIMEOUT_MS=1800000` (30 minutes), tripling Claude Code's default of 10 minutes. This accommodates long hidden-thinking responses that would otherwise be aborted by the total-request timer.

### Idle timeout opt-out

All Claude launch commands set `API_FORCE_IDLE_TIMEOUT=0`, disabling the 5-minute no-bytes idle timer that Claude Code 2.1.169+ arms on gateway connections (it aborts any request that goes 5 minutes without receiving bytes). Long hidden-thinking turns through the proxy would otherwise abort with `API error: undefined Request timed out` at ~300s and retry from scratch. With the idle timer off, the total-request timer (`API_TIMEOUT_MS`) is the only client-side timeout in effect.

All Claude launch commands also set `CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000`, raising Claude Code's no-bytes streaming watchdog to 10 minutes (its default is 3 minutes on first-party endpoints). The watchdog only arms when Claude Code treats the endpoint as first-party; elsewhere the variable has no effect.

### Disabling thinking

By default thinking is forced on (`alwaysThinkingEnabled:true`). Pass `--disable-thinking` to turn it off for the session: the `--settings` JSON carries `alwaysThinkingEnabled:false` plus `env.MAX_THINKING_TOKENS:0`. `MAX_THINKING_TOKENS=0` is what actually forces thinking off at the Anthropic API (the boolean alone is only a session default that the in-session `Option+T` toggle can override); it applies regardless of `--effort`, which is left untouched. Useful for running a subset of sessions without extended thinking (assign it per task from the orchestration side for a controlled split). Thinking cannot be disabled on Fable 5.

The JSON trace records `session.thinking_disabled` (`true`/`false`) so a reviewer can tell which runs were launched with the flag. This reflects the launch configuration the extractor applied — it is not a wire-level confirmation that the API disabled thinking.

### Output token cap

## Response streaming

For streamed responses (`text/event-stream`), the proxy forwards each chunk to the CLI as it arrives, so the CLI receives the response headers and output immediately and long turns aren't cut off by its connection/idle timeouts (the timeout settings above are a backstop). This applies to all three CLIs. Non-streaming responses are forwarded whole. The full stream is parsed for metrics.

A turn that ends before its terminal event (an upstream cut mid-stream, or the client disconnecting) is recorded with `aborted: true` and an `error` message so failed turns stay visible. Claude runs also record a response that never carried one (such as an empty body) and any turn the API rejected the same way: an in-stream `error` event, a non-streamed error body (for example a classifier block, or an overload before streaming began) or any HTTP status of 400 or above sets `aborted: true` and records the status in `turns[].response.status_code`; the API's error object lands in `turns[].response.api_errors` when the response carried one; `error` starts with `HTTP <status>` on any error status, followed by the API's error text when there was one (an in-stream `error` event records the API's text alone). Aborted turns are counted in `summary.total_turns` and `summary.aborted_turns` but excluded from the token, latency, and throughput averages so partial data doesn't skew them.

## When Done

1. Exit the CLI first (`/exit` or close terminal)
2. Then Ctrl+C the proxy to save

All artifacts for a run live under `./logs/<run_id>/`, where `<run_id>` is `{ts}_{pid}` (set by `main.py`; concurrent runs don't collide):

- `trace.json` — Always. Proxy JSON trace (tokens, latency, request/response, effort, etc.).
- `transcript.txt` — Only with `--capture-transcript`. Terminal transcript (rendered text from tmux).
- `session.jsonl` — Only on `--claude` runs without `--no-claude-config`. Claude Code's native per-session log (copied from `claude_config/active_<port>/` at shutdown). If multiple sessions ran in the same extractor run (e.g., the worker invoked claude more than once or used `/clear`), each is copied with a numbered suffix: `session_1.jsonl`, `session_2.jsonl`, etc.
- `debug.txt` — Only on `--claude` runs without `--no-claude-config`. Claude Code's debug log (the launch command also passes `--debug`). Same numbering applies if multiple sessions ran (`debug_1.txt`, `debug_2.txt`, ...).
- `rollout.jsonl` — Only on `--codex` runs without `--no-codex-home`. Codex's native session rollout transcript (copied from `codex_home/active_<port>/sessions/**/` at shutdown). If multiple sessions ran in the same extractor run, each is copied with a numbered suffix: `rollout_1.jsonl`, `rollout_2.jsonl`, etc.
- `subagents.zip` — Only on `--claude` runs without `--no-claude-config`. Claude Code's native per-subagent transcripts (`agent-<id>.jsonl` with their `agent-<id>.meta.json` sidecars, whose `toolUseId` ties each subagent to the `Agent` tool call in `session.jsonl`), plus a `manifest.json` listing them by session and meta. Inside the zip, entries are grouped one level deep by session (`<session>/agent-<id>.jsonl`), so a run with more than one session stays separate. Always written: a run that spawned no subagents gets a manifest with an empty `subagents` list.

The extractor prints "Files to share for this run:" with the resolved paths at shutdown.

For `--claude` runs without `--no-claude-config`, the full Claude Code per-run state (`.claude.json`, `backups/`, etc.) is preserved at `./claude_config/<run_id>/` after the shutdown rename. See the "Claude config redirect" section above for details and crash-recovery notes.

JSON trace metadata: `session.extractor_version` (extractor version), `session.cli_version` (wrapped CLI's installed version at session start, or `null` if not on PATH). Claude traces additionally include `session.claude_config_dir` — the per-run path under `claude_config/` where Claude Code's raw state ends up after the shutdown rename (`claude_config/<run_id>/`), or `null` if `--no-claude-config` was set. This differs from the `CLAUDE_CONFIG_DIR` literal in the launch command, which uses the stable runtime path `claude_config/active_<port>/`. Claude traces also carry `session.thinking_disabled` (`true`/`false`), recording whether the run was launched with `--disable-thinking`. Gemini traces do not have these fields. Codex traces include the parallel `session.codex_home` — the per-run path under `codex_home/` where Codex's raw state (including the rollout transcripts) ends up after the shutdown rename (`codex_home/<run_id>/`), or `null` if `--no-codex-home` was set.
