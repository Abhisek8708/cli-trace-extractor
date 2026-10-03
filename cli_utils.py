from __future__ import annotations

import json
import os
import platform
import re
import shlex
import socket
import subprocess
import zipfile
from functools import lru_cache
from pathlib import Path

# Extractor version (surfaced via --version and written into every JSON log)
VERSION = "1.42.0"

# Set to True to require --model argument, False to use defaults
MODEL_REQUIRED = True

# Set to True to require --effort-level/--reasoning-effort/--permission-mode, False to use defaults
CONFIG_REQUIRED = True

# Default models (used when MODEL_REQUIRED = False)
CLAUDE_DEFAULT_MODEL = "claude-opus-4-7"
CODEX_DEFAULT_MODEL = "gpt-5.5"
GEMINI_DEFAULT_MODEL = "gemini-3.1-pro-preview"

# Environment set on every Claude launch. Kept separate from the command so
# each shell renders it natively (NAME=value prefixes on Unix, $env: on
# Windows). A per-model entry may list one of these to fix its position in the
# launch command or to override its value for that model.
CLAUDE_CMD_ENV: dict[str, str] = {
    "API_FORCE_IDLE_TIMEOUT": "0",
    "API_TIMEOUT_MS": "1800000",
    "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "600000",
}

# Extra environment variables for the Claude launch command, gated per model:
# model prefix -> {NAME: value}, emitted in dict order. The --model value is
# prefix-matched, so an optional "[1m]" suffix (or a dated variant) still
# matches; the first matching entry wins, and a model that matches nothing gets
# no extra vars. Adding an entry here is the only change needed — both the Unix
# and Windows launch paths pick it up. Empty when no model needs any.
_CLAUDE_ENV_EAP_SET: dict[str, str] = {
    "DISABLE_AUTOUPDATER": "1",
    "_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL": "1",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "128000",
    "CLAUDE_CODE_POLISHED_DEWDROP": "drop",
    "CLAUDE_CODE_TOTAL_TOKENS_REMINDER": "padded-countdown",
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",
    "CLAUDE_CODE_EXTRA_BODY": '{"fallbacks":"default"}',
    "ANTHROPIC_BETAS": "server-side-fallback-2026-06-01,server-side-fallback-2026-07-01",
}
CLAUDE_ENV_BY_MODEL: dict[str, dict[str, str]] = {
    "claude-larimar-eap": _CLAUDE_ENV_EAP_SET,
    "claude-serpentine-eap": _CLAUDE_ENV_EAP_SET,
}

# Extra Claude CLI flags gated per model, same prefix-match rules as
# CLAUDE_ENV_BY_MODEL. Appended to the claude invocation; one injection site
# serves both the Unix and Windows paths since both build on the same command
# string. Empty when no model needs any.
_CLAUDE_EAP_DISALLOWED_TOOLS = '--disallowedTools "TaskCreate,TaskGet,TaskUpdate,TaskList,TodoWrite"'
CLAUDE_FLAGS_BY_MODEL: dict[str, str] = {
    "claude-larimar-eap": f'--client-data-url "https://downloads.claude.ai/claude-code-releases/eap/de0eff2ae4b024fb8a3f23d266c8a583ad6f8987aff020fe/config.json" {_CLAUDE_EAP_DISALLOWED_TOOLS}',
    "claude-serpentine-eap": f'--client-data-url "https://downloads.claude.ai/claude-code-releases/eap/3a35886531680e6e2470d3f8d8fbe42d11768cbb87a4a5a9/config.json" {_CLAUDE_EAP_DISALLOWED_TOOLS}',
}

# Text appended to the Claude system prompt via --append-system-prompt-file,
# gated per model with the same prefix-match rules. Written to a file per run
# by seed_claude_append_system_prompt; build_launch_command adds the flag.
# Empty when no model needs any.
CLAUDE_APPEND_SYSTEM_PROMPT_BY_MODEL: dict[str, str] = {}

# Default config values (used when CONFIG_REQUIRED = False)
CLAUDE_DEFAULT_EFFORT_LEVEL = "xhigh"
CODEX_DEFAULT_REASONING_EFFORT = "xhigh"
CLAUDE_DEFAULT_PERMISSION_MODE = "auto"

# Map our --permission-mode value to the actual Claude Code CLI flag
_CLAUDE_PERMISSION_FLAG = {
    "skip": "--dangerously-skip-permissions",
    "auto": "--permission-mode auto",
}
CLAUDE_PERMISSION_MODES = tuple(_CLAUDE_PERMISSION_FLAG)

# Settings always passed via Claude's --settings flag. By default forces thinking
# mode on regardless of the worker's local /config toggle (which would otherwise
# persist `alwaysThinkingEnabled: false` in ~/.claude/settings.json). When
# --disable-thinking is set, _build_claude_settings_json sets this to False AND
# adds `env: {MAX_THINKING_TOKENS: "0"}` (the authoritative API-level force-off,
# since the bool alone is only a session default). Flag-gated additions (e.g.
# showThinkingSummaries via --show-thinking-summaries) are merged in at
# launch-command-build time by _build_claude_settings_json.
CLAUDE_BASE_SETTINGS: dict[str, object] = {"alwaysThinkingEnabled": True}

# CLI command templates with placeholders. `settings_json` for Claude is built
# dynamically in build_launch_command from CLAUDE_BASE_SETTINGS plus any
# flag-gated additions.
CLAUDE_CMD_TEMPLATE = "claude --model {model} {permission_flag} --effort {effort_level} --settings {settings_json}"
CODEX_CMD_TEMPLATE = (
    'codex --sandbox workspace-write --ask-for-approval on-request --disable fast_mode '
    '--config sandbox_workspace_write.network_access=true '
    '--model {model} --config model_reasoning_effort="{reasoning_effort}" '
    '--config model_reasoning_summary="auto"'
)
GEMINI_CMD_TEMPLATE = "gemini --model {model}"

# Proxy base URL seeded into the redirected (empty) CODEX_HOME so Codex routes
# through the LLM proxy instead of api.openai.com.
CODEX_OPENAI_BASE_URL = "https://app-llmproxy.dataannotation.tech/api/llm_proxy/openai/v1"


@lru_cache(maxsize=1)
def get_platform_info() -> tuple[str, bool]:
    system = platform.system()
    is_wsl = False
    if system == "Linux":
        try:
            with open("/proc/version", "r") as f:
                is_wsl = "microsoft" in f.read().lower()
        except Exception:
            pass
    return system, is_wsl


@lru_cache(maxsize=1)
def get_cert_path() -> Path:
    return Path.home() / ".mitmproxy" / "mitmproxy-ca-cert.pem"


def get_shell_name() -> str:
    shell = os.environ.get("SHELL", "")
    if shell:
        return Path(shell).name
    system, _ = get_platform_info()
    return "cmd" if system == "Windows" else "bash"


def _ps_quote(value: str) -> str:
    """PowerShell single-quoted literal: only ' needs escaping, by doubling."""
    return "'" + value.replace("'", "''") + "'"


def _claude_append_system_prompt_for_model(model: str) -> str:
    for prefix, text in CLAUDE_APPEND_SYSTEM_PROMPT_BY_MODEL.items():
        if model.startswith(prefix):
            return text
    return ""


def seed_claude_append_system_prompt(target_dir: Path, model: str) -> Path | None:
    """Write the model's appended system prompt, if it has one, into target_dir
    and return the file path for --append-system-prompt-file; None otherwise."""
    text = _claude_append_system_prompt_for_model(model)
    if not text:
        return None
    path = target_dir / "append_system_prompt.txt"
    path.write_bytes(text.encode("utf-8"))
    return path


def seed_codex_config(codex_home: Path) -> None:
    """Write a minimal config.toml (just openai_base_url) into a redirected
    CODEX_HOME so Codex routes through the LLM proxy."""
    (codex_home / "config.toml").write_text(
        f'openai_base_url = "{CODEX_OPENAI_BASE_URL}"\n'
    )


def zip_subagent_transcripts(claude_config_dir: Path, run_dir: Path) -> int:
    """Bundle Claude Code's per-subagent transcripts from a redirected
    CLAUDE_CONFIG_DIR into run_dir/subagents.zip. Claude Code writes each
    subagent's transcript to projects/<cwd>/<session>/subagents/agent-<id>.jsonl
    with an agent-<id>.meta.json sidecar whose toolUseId ties the subagent to the
    Agent tool call in the main session JSONL. Entries are grouped one level deep
    by session (<session>/agent-<id>.jsonl) so a run with more than one session
    stays separate, alongside a manifest.json listing each transcript by session
    and meta. The zip is always written so the artifact set stays uniform: a run
    that spawned no subagents gets a manifest with an empty subagents list, an
    explicit none rather than a missing file (Claude Code leaves nothing on disk
    for such a run, so there is no native marker to pass through). Returns the
    number of transcripts bundled."""
    projects = claude_config_dir / "projects"
    transcripts: list[Path] = []
    metas: list[Path] = []
    if projects.is_dir():
        transcripts = sorted(p for p in projects.glob("**/subagents/agent-*.jsonl") if p.is_file())
        metas = sorted(p for p in projects.glob("**/subagents/agent-*.meta.json") if p.is_file())

    def _place(p: Path) -> tuple[str, str]:
        """(session, arcname) for a subagent file. The session owns the outermost
        subagents/ dir, so a nested subagent lands with its session too."""
        parts = p.relative_to(projects).parts
        i = parts.index("subagents")
        session = parts[i - 1] if i else ""
        return session, f"{session}/{p.name}" if session else p.name

    manifest = []
    for t in transcripts:
        session, arcname = _place(t)
        entry = {"session": session, "transcript": arcname}
        meta_path = t.parent / f"{t.stem}.meta.json"
        if meta_path.is_file():
            try:
                entry["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        manifest.append(entry)
    # Build under a temp name and swap in only on success, so a failure part-way
    # through cannot leave a readable-but-incomplete zip behind (manifest.json is
    # written last). The dot prefix keeps a stray temp out of the shutdown listing.
    tmp = run_dir / ".subagents.zip.tmp"
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in transcripts + metas:
                zf.write(p, arcname=_place(p)[1])
            zf.writestr("manifest.json", json.dumps({"subagents": manifest}, indent=2))
        tmp.replace(run_dir / "subagents.zip")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return len(transcripts)


def _claude_env_for_model(model: str) -> dict[str, str]:
    for prefix, env in CLAUDE_ENV_BY_MODEL.items():
        if model.startswith(prefix):
            return env
    return {}


def _claude_env_vars(model: str) -> dict[str, str]:
    """Per-model vars in their configured order, then any CLAUDE_CMD_ENV var the
    model did not set itself."""
    merged = dict(_claude_env_for_model(model))
    for name, value in CLAUDE_CMD_ENV.items():
        merged.setdefault(name, value)
    return merged


def _claude_flags_for_model(model: str) -> str:
    for prefix, flags in CLAUDE_FLAGS_BY_MODEL.items():
        if model.startswith(prefix):
            return flags
    return ""


def _build_claude_settings_json(show_thinking_summaries: bool, disable_thinking: bool = False) -> str:
    """Serialize Claude's --settings JSON, shell-quoted for direct insertion into the launch command."""
    settings = dict(CLAUDE_BASE_SETTINGS)
    if disable_thinking:
        # alwaysThinkingEnabled only sets the session DEFAULT — the in-session
        # Option+T toggle can still flip thinking back on. The authoritative
        # force-off is MAX_THINKING_TOKENS=0, which disables thinking on the
        # Anthropic API regardless of effort (per Claude Code docs; no effect on
        # Fable 5). Set both: the env var forces it off, the bool keeps the
        # session default and the display off too.
        settings["alwaysThinkingEnabled"] = False
        settings["env"] = {"MAX_THINKING_TOKENS": "0"}
    if show_thinking_summaries:
        settings["showThinkingSummaries"] = True
    return shlex.quote(json.dumps(settings, separators=(",", ":")))


def build_launch_command(
    cert_path: Path,
    port: int,
    cli: str,
    model: str,
    effort_level: str | None = None,
    reasoning_effort: str | None = None,
    permission_mode: str | None = None,
    tmux_session: str | None = None,
    verbose: bool = False,
    disable_autoupdate: bool = False,
    show_thinking_summaries: bool = False,
    disable_thinking: bool = False,
    claude_config_dir: Path | None = None,
    codex_home: Path | None = None,
    append_system_prompt_file: Path | None = None,
) -> str:
    system, _ = get_platform_info()
    proxy_url = f"http://127.0.0.1:{port}"
    env_vars = f'NODE_EXTRA_CA_CERTS="{cert_path}" HTTPS_PROXY="{proxy_url}"'
    gated_env = _claude_env_vars(model) if cli == "claude" else {}
    # A per-model entry may already set DISABLE_AUTOUPDATER; emit it once.
    autoupdate = cli == "claude" and disable_autoupdate and "DISABLE_AUTOUPDATER" not in gated_env

    if cli == "claude":
        effort = effort_level if effort_level is not None else CLAUDE_DEFAULT_EFFORT_LEVEL
        pmode = permission_mode if permission_mode is not None else CLAUDE_DEFAULT_PERMISSION_MODE
        permission_flag = _CLAUDE_PERMISSION_FLAG[pmode]
        settings_json = _build_claude_settings_json(show_thinking_summaries, disable_thinking)
        # Quoted: a "[1m]" suffix is a glob pattern to zsh once it is a plain argument.
        cmd = CLAUDE_CMD_TEMPLATE.format(
            model=shlex.quote(model),
            effort_level=effort,
            permission_flag=permission_flag,
            settings_json=settings_json,
        )
        extra_flags = _claude_flags_for_model(model)
        if extra_flags:
            cmd += f" {extra_flags}"
        if verbose:
            cmd += " --verbose"
        # CLAUDE_CONFIG_DIR redirects Claude Code's per-session state (session
        # JSONL, debug log, etc.) into the supplied dir for this invocation
        # only. --debug is paired so the debug log is actually written under it.
        if claude_config_dir is not None:
            cmd += " --debug"
            env_vars = f"CLAUDE_CONFIG_DIR={shlex.quote(str(claude_config_dir))} {env_vars}"
        if append_system_prompt_file is not None:
            cmd += f" --append-system-prompt-file {shlex.quote(str(append_system_prompt_file))}"
        if autoupdate:
            env_vars = f"DISABLE_AUTOUPDATER=1 {env_vars}"
        for name, value in gated_env.items():
            env_vars = f"{env_vars} {name}={shlex.quote(value)}"
    elif cli == "codex":
        effort = reasoning_effort if reasoning_effort is not None else CODEX_DEFAULT_REASONING_EFFORT
        cmd = CODEX_CMD_TEMPLATE.format(model=model, reasoning_effort=effort)
        # CODEX_HOME redirects Codex's per-run state (including the native session
        # rollout transcripts under sessions/) into the supplied dir for this
        # invocation only.
        if codex_home is not None:
            env_vars = f"CODEX_HOME={shlex.quote(str(codex_home))} {env_vars}"
    elif cli == "gemini":
        cmd = GEMINI_CMD_TEMPLATE.format(model=model)
    else:
        raise ValueError(f"Unknown CLI: {cli}")

    # Windows lacks native tmux; the wrap is not applied here.
    if system == "Windows":
        autoupdate_prefix = '$env:DISABLE_AUTOUPDATER="1"; ' if autoupdate else ""
        config_prefix = ""
        if cli == "claude" and claude_config_dir is not None:
            config_prefix = f'$env:CLAUDE_CONFIG_DIR="{claude_config_dir}"; '
        elif cli == "codex" and codex_home is not None:
            config_prefix = f'$env:CODEX_HOME="{codex_home}"; '
        gated_suffix = "".join(f"$env:{name}={_ps_quote(value)}; " for name, value in gated_env.items())
        return f'{autoupdate_prefix}{config_prefix}$env:NODE_EXTRA_CA_CERTS="{cert_path}"; $env:HTTPS_PROXY="{proxy_url}"; {gated_suffix}{cmd}'

    if get_shell_name() == "fish":
        base = f"env {env_vars} {cmd}"
    else:
        base = f"{env_vars} {cmd}"

    # shlex.quote is required because cmd may contain shell metacharacters (e.g. nested quotes).
    if tmux_session:
        return f"tmux new-session -s {tmux_session} {shlex.quote(base)}"
    return base


# Regex for extracting M.N.P from `<cli> --version` stdout. Observed outputs:
#   claude --version  -> "2.1.138 (Claude Code)"
#   codex --version   -> "codex-cli 0.125.0"
#   gemini --version  -> "0.39.1"
# Pre-release suffixes (e.g. "0.131.0-alpha.4") are not captured beyond M.N.P.
_CLI_VERSION_RE = re.compile(r"\d+\.\d+\.\d+")

# Subprocess timeout for `<cli> --version`. Bounds banner-display delay if the
# binary hangs.
_CLI_VERSION_TIMEOUT_S = 5


def get_cli_version(cli: str) -> str | None:
    """Return the installed semver of <cli>, or None if not detectable.

    Runs `<cli> --version` and extracts the first M.N.P semver from stdout.
    Returns None on: not-on-PATH, non-zero exit, timeout, no match, or any
    subprocess error. Never raises (called from proxy logger addon __init__).
    """
    try:
        result = subprocess.run(
            [cli, "--version"],
            capture_output=True,
            text=True,
            timeout=_CLI_VERSION_TIMEOUT_S,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    match = _CLI_VERSION_RE.search(result.stdout)
    return match.group(0) if match else None


def format_cli_install_hint(cli: str, version: str) -> str:
    """Return the shell command for installing <cli> at <version>.

    Claude uses Anthropic's native installer (platform-specific); codex and
    gemini use npm `@<version>` syntax. Returns "" for an unknown CLI.
    """
    if cli == "claude":
        # https://code.claude.com/docs/en/setup#install-a-specific-version
        system, _ = get_platform_info()
        if system == "Windows":
            template = "& ([scriptblock]::Create((irm https://claude.ai/install.ps1))) {version}"
        else:
            template = "curl -fsSL https://claude.ai/install.sh | bash -s {version}"
    elif cli == "codex":
        # https://www.npmjs.com/package/@openai/codex
        template = "npm install -g @openai/codex@{version}"
    elif cli == "gemini":
        # https://www.npmjs.com/package/@google/gemini-cli
        template = "npm install -g @google/gemini-cli@{version}"
    else:
        return ""
    return template.format(version=version)


def enforce_cli_version(cli: str, required: str) -> str | None:
    """Check that <cli> on PATH is exactly <required>. Returns None on match, else a multi-line error message ready to print."""
    installed = get_cli_version(cli)
    hint = format_cli_install_hint(cli, required)
    install_line = f"\n  To install the required version, run:\n    {hint}" if hint else ""
    if installed is None:
        return (
            f"ERROR: {cli} is not on PATH or its version could not be detected.\n"
            f"  --require-cli-version is set to {required}.{install_line}"
        )
    if installed != required:
        return (
            f"ERROR: {cli} version mismatch.\n"
            f"  Installed: {installed}\n"
            f"  Required:  {required} (via --require-cli-version){install_line}"
        )
    return None


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard. Returns True if successful."""
    system, _ = get_platform_info()
    try:
        if system == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode(), check=True)
            return True
        if system == "Linux":
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
                return True
            except FileNotFoundError:
                try:
                    subprocess.run(["xsel", "--clipboard", "--input"], input=text.encode(), check=True)
                    return True
                except FileNotFoundError:
                    return False
        if system == "Windows":
            subprocess.run(["clip"], input=text.encode(), check=True, shell=True)
            return True
    except Exception:
        pass
    return False
