"""Hermes Agent lifecycle manager for TalonSight Analyst mode.

Everything runs from within the Streamlit UI — no terminal required.

Setup flow (all driven from Streamlit):
  1. install()        — runs install.sh, kills before interactive wizard starts
  2. configure_llm()  — writes model: YAML block directly (non-interactive)
  3. register_mcp()   — `hermes mcp add` or direct YAML write
  4. verify()         — confirm hermes -z responds correctly

Per-question usage (Analyst mode):
  ask_hermes(question, output_cb)
      — runs `hermes -z "<question>" -t talonsight` as a subprocess,
        streams progress to output_cb, returns final answer text.

No persistent gateway process is needed.  Hermes -z handles each
question as a fresh sub-agent invocation against the talonsight MCP
toolset (database tools).

The user stays in Streamlit throughout.  Every step streams progress
strings to output_cb so the UI can display them in real time.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Config path discovery ─────────────────────────────────────────────────────
# Hermes stores its config under ~/.hermes/ (per the official docs).
# We check several candidates so the code works across Hermes versions.
_CONFIG_CANDIDATES = [
    Path.home() / ".hermes" / "config.yaml",
    Path.home() / ".config" / "hermes" / "config.yaml",
    Path.home() / ".hermes" / "config" / "config.yaml",
]

# Hermes Gateway default port
GATEWAY_PORT = 7860
GATEWAY_URL  = f"http://localhost:{GATEWAY_PORT}"

# ANSI / terminal control sequence stripper
_ANSI_RE = re.compile(
    r'\x1b(?:'
    r'\[[0-9;]*[mGKHFJABCDhlrsuPX]'   # CSI sequences
    r'|\[\?[0-9;]*[hl]'                 # DEC private mode
    r'|[()][AB012]'                     # charset selection
    r'|[78]'                            # save/restore cursor
    r'|M'                               # reverse index
    r')'
)


def _clean(text: str) -> str:
    """Strip ANSI escape codes and return printable text."""
    return _ANSI_RE.sub('', text).strip()


# ── Detection ─────────────────────────────────────────────────────────────────

def is_installed() -> bool:
    """Return True if the `hermes` CLI is on PATH."""
    _refresh_path()
    return shutil.which("hermes") is not None


def _hermes_version() -> str:
    try:
        r = subprocess.run(
            ["hermes", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return _clean(r.stdout or r.stderr).split('\n')[0]
    except Exception:
        return "unknown"


def _refresh_path() -> None:
    """Add common Hermes install dirs to PATH for the current process."""
    for p in [
        Path.home() / ".local" / "bin",
        Path.home() / ".hermes" / "bin",
        Path("/usr/local/bin"),
    ]:
        s = str(p)
        if p.exists() and s not in os.environ.get("PATH", ""):
            os.environ["PATH"] = s + os.pathsep + os.environ.get("PATH", "")


# ── Installation ──────────────────────────────────────────────────────────────
# Strategy
# --------
# 1. Download install.sh and run it via `bash -s`.
# 2. Stream output line-by-line, stripping ANSI, to output_cb.
# 3. The script installs packages and syncs skills, then launches an
#    interactive TUI setup wizard.  We KILL the process the moment we
#    see any "install done" marker — Hermes is fully installed at that
#    point.  The wizard is just post-install configuration which we
#    handle separately via `hermes config set` (non-interactive).
# 4. Reload PATH and verify `hermes` is resolvable.

_INSTALL_DONE_MARKERS = [
    "skills synced",
    "total bundled",
    "installation complete",
    "successfully installed",
    "hermes is ready",
    "starting setup wizard",   # kill right before it renders
    "let's configure",         # first wizard line
    "press ctrl+c",            # wizard safety message
    "setup wizard",
]


def install(output_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Download and run the Hermes Agent install script from within Python.

    Streams sanitised progress to output_cb.  Returns True when Hermes is
    installed and on PATH.
    """
    def emit(msg: str) -> None:
        clean = _clean(msg)
        if not clean:
            return
        logger.info("hermes-install: %s", clean)
        if output_cb:
            output_cb(clean)

    emit("Downloading Hermes Agent installer…")

    try:
        import urllib.request
        with urllib.request.urlopen(
            "https://hermes-agent.nousresearch.com/install.sh", timeout=30
        ) as r:
            script = r.read().decode("utf-8")
    except Exception as exc:
        emit(f"❌  Could not download installer: {exc}")
        return False

    emit("Running installer (2–3 minutes, once)…")

    env = {
        **os.environ,
        "NONINTERACTIVE":   "1",
        "CI":               "1",
        "HERMES_SKIP_SETUP":"1",   # newer versions honour this
        "TERM":             "dumb", # disables fancy TUI rendering
    }

    try:
        proc = subprocess.Popen(
            ["bash", "-s"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdin and proc.stdout

        # Write the script and close stdin so bash doesn't wait for more input
        proc.stdin.write(script)
        proc.stdin.close()

        install_done = False
        for raw_line in proc.stdout:
            line = _clean(raw_line)
            if line:
                emit(line)
            low = raw_line.lower()
            if any(m in low for m in _INSTALL_DONE_MARKERS):
                install_done = True
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                emit("✓  Packages installed — skipping interactive wizard")
                break

        if not install_done:
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                emit("❌  Installer timed out after 5 minutes")
                return False

    except Exception as exc:
        emit(f"❌  Installer error: {exc}")
        return False

    _refresh_path()

    if is_installed():
        emit(f"✓  Hermes Agent ready ({_hermes_version()})")
        return True

    emit("⚠  Installed but `hermes` not on PATH yet.")
    emit("   Trying common install locations…")

    # One more attempt — installer may have added to shell rc but not this process
    for extra in [
        Path.home() / ".local" / "bin" / "hermes",
        Path.home() / ".hermes"  / "bin" / "hermes",
    ]:
        if extra.exists():
            os.environ["PATH"] = str(extra.parent) + os.pathsep + os.environ.get("PATH","")
            emit(f"✓  Found hermes at {extra}")
            return True

    emit("❌  Could not locate `hermes` — please open a new terminal and retry.")
    return False


# ── LLM configuration ─────────────────────────────────────────────────────────
# We always write the config as a nested `llm:` block in YAML — this is the
# format Hermes expects.  `hermes config set` writes flat root-level keys
# which triggers "stale root-level provider/base_url" warnings from
# `hermes doctor`.  So we skip the CLI entirely and write YAML directly,
# then run `hermes doctor --fix` to clean up any pre-existing stale keys.

def configure_llm(
    provider: str, model: str, url: str, api_key: str = "",
    output_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    def emit(msg: str) -> None:
        if output_cb:
            output_cb(msg)

    if not is_installed():
        emit("⚠  Hermes not installed — skipping LLM config")
        return False

    # Always write the nested llm: format — avoids stale root-level key warnings
    emit("→  Writing LLM config…")
    ok = _write_config_yaml(provider, model, url, api_key, output_cb)

    # Auto-fix any stale root-level keys left from previous runs or older Hermes
    _doctor_fix(output_cb)

    if ok:
        emit(f"✓  LLM configured ({provider} / {model})")
    return ok


def _doctor_fix(output_cb: Optional[Callable[[str], None]] = None) -> None:
    """Run `hermes doctor --fix` to clean up stale config entries silently."""
    try:
        r = subprocess.run(
            ["hermes", "doctor", "--fix"],
            capture_output=True, text=True, timeout=20,
        )
        # Only surface output if something actually changed
        output = _clean(r.stdout + r.stderr)
        if output_cb and ("fix" in output.lower() or "fixed" in output.lower()):
            output_cb(f"✓  Config cleaned up (hermes doctor --fix)")
    except Exception:
        pass  # non-fatal


def _write_config_yaml(
    provider: str, model: str, url: str, api_key: str,
    output_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    """Write the ``model:`` YAML block that Hermes reads via ``model.default``,
    ``model.provider``, and ``model.base_url``.

    Hermes resolves providers from the ``model:`` section, NOT from a top-level
    ``llm:`` block.  Writing ``llm:`` triggers "stale root-level" doctor warnings
    and routes local models incorrectly (e.g. provider=ollama → OpenRouter).

    Mapping rules
    -------------
    * Ollama / local:  provider="custom", base_url needs /v1 suffix.
    * MLX:             provider="custom", base_url as-is.
    * OpenAI:          provider="openai", no base_url.
    * Anthropic:       provider="anthropic", no base_url.
    * custom:          provider="custom", base_url as-is.
    """
    def emit(msg: str) -> None:
        if output_cb:
            output_cb(msg)

    config_path = _find_config_path()

    # Normalise provider to what Hermes understands
    hermes_provider = provider
    if provider == "ollama":
        hermes_provider = "custom"
        # Ollama's OpenAI-compatible endpoint lives under /v1
        if url and not url.rstrip("/").endswith("/v1"):
            url = url.rstrip("/") + "/v1"
    elif provider == "mlx":
        hermes_provider = "custom"

    # Build model config dict — "default" is the key for the model name
    model_cfg: dict = {"provider": hermes_provider, "default": model}
    if url and hermes_provider not in ("openai", "anthropic"):
        model_cfg["base_url"] = url
    if api_key and hermes_provider in ("openai", "anthropic", "custom"):
        model_cfg["api_key"] = api_key

    try:
        import yaml  # type: ignore[import-untyped]
        existing: dict = {}
        if config_path.exists():
            try:
                existing = yaml.safe_load(config_path.read_text()) or {}
            except Exception:
                pass
        # Write model: block; also remove any stale root-level llm: block
        existing["model"] = model_cfg
        existing.pop("llm", None)          # remove stale llm: block
        existing.pop("provider", None)     # remove stale root-level keys
        existing.pop("base_url", None)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(existing, default_flow_style=False))
        emit(f"✓  Config written to {config_path}")
        return True
    except ImportError:
        pass  # no yaml library — write raw

    # Raw YAML write (no library)
    lines = ["model:"]
    for k, v in model_cfg.items():
        lines.append(f"  {k}: {v}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing_text = config_path.read_text() if config_path.exists() else ""
    # Replace existing model: block if present; also strip stale llm: block
    for block_key in ("model", "llm"):
        existing_text = re.sub(
            rf'^{block_key}:.*?(?=^\w|\Z)', '',
            existing_text, flags=re.MULTILINE | re.DOTALL,
        )
    config_path.write_text(existing_text.strip() + "\n" + "\n".join(lines) + "\n")
    emit(f"✓  Config written to {config_path}")
    return True


def _find_config_path() -> Path:
    """Return the existing Hermes config path, or the default location."""
    for p in _CONFIG_CANDIDATES:
        if p.exists():
            return p
    path = _CONFIG_CANDIDATES[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ── MCP registration ──────────────────────────────────────────────────────────

def register_mcp(output_cb: Optional[Callable[[str], None]] = None) -> bool:
    def emit(msg: str) -> None:
        if output_cb:
            output_cb(msg)

    mcp_cmd = shutil.which("talonsight-mcp") or "talonsight-mcp"

    if is_installed():
        try:
            r = subprocess.run(
                ["hermes", "mcp", "add", "talonsight",
                 "--command", mcp_cmd, "--transport", "stdio"],
                capture_output=True, text=True, timeout=15,
            )
            stderr_low = r.stderr.lower()
            if r.returncode == 0 or "already" in stderr_low or "exists" in stderr_low:
                emit("✓  Database tools registered with Hermes")
                return True
            emit(f"⚠  CLI registration: {_clean(r.stderr)[:100]} — trying YAML fallback")
        except Exception as exc:
            emit(f"⚠  CLI registration error ({exc}) — trying YAML fallback")

    return _register_mcp_yaml(mcp_cmd, output_cb)


def _register_mcp_yaml(mcp_cmd: str,
                        output_cb: Optional[Callable[[str], None]] = None) -> bool:
    def emit(msg: str) -> None:
        if output_cb:
            output_cb(msg)

    config_path = _find_config_path()
    entry = {"command": mcp_cmd, "transport": "stdio"}

    try:
        import yaml  # type: ignore[import-untyped]
        existing: dict = {}
        if config_path.exists():
            existing = yaml.safe_load(config_path.read_text()) or {}
        servers = existing.get("mcp_servers", {})
        servers["talonsight"] = entry
        existing["mcp_servers"] = servers
        config_path.write_text(yaml.dump(existing, default_flow_style=False))
        emit("✓  Database tools registered via config.yaml")
        return True
    except ImportError:
        snippet = (
            "\nmcp_servers:\n"
            "  talonsight:\n"
            f"    command: {mcp_cmd}\n"
            "    transport: stdio\n"
        )
        text = config_path.read_text() if config_path.exists() else ""
        if "mcp_servers:" not in text:
            config_path.write_text(text + snippet)
            emit("✓  Database tools registered via config.yaml")
            return True
    except Exception as exc:
        emit(f"⚠  Could not register MCP tools: {exc}")

    return False


# ── Hermes readiness check (no persistent gateway needed) ────────────────────
# Integration uses `hermes -z "<question>" -t talonsight` per-question.
# There is no persistent gateway process.  We just verify the CLI works.

def is_gateway_alive() -> bool:
    """Legacy name kept for call-site compatibility.  Returns True when
    hermes is installed and functional."""
    return is_functional()


def is_functional() -> bool:
    """Return True when `hermes -z` can answer a trivial question."""
    if not is_installed():
        return False
    try:
        r = subprocess.run(
            ["hermes", "-z", "Reply with the single word: ready", "-t", "talonsight"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def start_gateway(output_cb: Optional[Callable[[str], None]] = None) -> bool:
    """No persistent gateway is needed — `hermes -z` is invoked per-question.
    This function just verifies hermes is installed and functional."""
    def emit(msg: str) -> None:
        if output_cb:
            output_cb(msg)

    if not is_installed():
        emit("❌  Hermes not installed")
        return False

    emit("✓  Analyst engine ready (hermes -z per-question mode)")
    return True


def stop_gateway() -> None:
    """No-op — no persistent gateway process exists."""
    pass


# ── Per-question Hermes invocation ────────────────────────────────────────────

def _extract_sql(text: str) -> Optional[str]:
    """Pull a SQL SELECT statement out of a Hermes response.

    hermes3 (and similar small models) often emit tool calls as scratchpad
    JSON rather than invoking the MCP protocol.  We intercept here.

    Patterns handled (in priority order):
      1. {"arguments": {"sql": "SELECT ..."}, "name": "run_sql"}
      2. {"sql": "SELECT ..."}
      3. ```sql SELECT ... ```
      4. Bare SELECT ... statement
    """
    import json, re

    cleaned = re.sub(r'<SCRATCHPAD>|</SCRATCHPAD>', '', text, flags=re.IGNORECASE).strip()

    # Pattern 1 & 2: any JSON with a "sql" key anywhere
    for chunk in re.findall(r'\{[^{}]*"sql"[^{}]*\}', cleaned, re.DOTALL):
        try:
            obj = json.loads(chunk)
            if "arguments" in obj and isinstance(obj["arguments"], dict):
                sql = obj["arguments"].get("sql", "")
            else:
                sql = obj.get("sql", "")
            if sql and re.search(r'\bSELECT\b', sql, re.IGNORECASE):
                return _clean_sql(sql)
        except (json.JSONDecodeError, ValueError):
            pass

    # Pattern 3: fenced code block — extract content, then strip any trailing prose
    m = re.search(r'```(?:sql)?\s*(SELECT.+?)(?:```|$)', cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return _clean_sql(m.group(1))

    # Pattern 4: bare SELECT statement
    m = re.search(r'(SELECT\s+.+)', cleaned, re.DOTALL | re.IGNORECASE)
    if m:
        return _clean_sql(m.group(1))

    return None


def _clean_sql(sql: str) -> Optional[str]:
    """Strip trailing non-SQL prose and normalise a SQL string."""
    import re
    lines = sql.strip().splitlines()
    sql_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^(If|Please|Note|This|The|I |You |Here|Let|That|It )', stripped):
            break
        sql_lines.append(line)
    result = "\n".join(sql_lines).rstrip("; \n\t")
    if result and re.search(r'\bSELECT\b', result, re.IGNORECASE):
        return result
    return None


def _execute_sql_safe(sql: str) -> tuple[str, str]:
    """Run sql via the MCP server; return (markdown_table, error_or_empty)."""
    try:
        from talonsight.mcp_server import _run_sql
        return _run_sql(sql), ""
    except Exception as exc:
        return "", str(exc)


def _ask_hermes_raw(prompt: str, timeout: int = 90) -> str:
    """Low-level hermes -z call, returns raw stdout."""
    try:
        r = subprocess.run(
            ["hermes", "-z", prompt, "-t", "talonsight"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def _repair_and_execute(
    sql: str, error: str, schema_context: str, question: str
) -> tuple[str, str]:
    """Ask the model to fix a failed SQL given the error message.

    Returns (markdown_table, final_error).
    """
    repair_prompt = (
        f"{schema_context}\n\n"
        f"The following SQL failed:\n```sql\n{sql}\n```\n\n"
        f"Error: {error}\n\n"
        f"Write a corrected SELECT query that answers: {question!r}\n"
        f"Use ONLY the tables shown above. Do NOT JOIN tables not listed. "
        f"Return ONLY the SQL, no explanation."
    )
    raw = _ask_hermes_raw(repair_prompt, timeout=90)
    fixed_sql = _extract_sql(raw) if raw else None
    if not fixed_sql:
        return "", error
    result, err2 = _execute_sql_safe(fixed_sql)
    if err2:
        return "", err2
    return result, ""


def _build_enriched_prompt(
    question: str,
    history: Optional[list[dict]] = None,
) -> str:
    """Prefix the question with schema context for only the relevant tables,
    plus recent conversation history so follow-up questions have context.

    Scoring: pick top-3 tables by keyword overlap with the combined text of
    the question + recent history (so follow-ups like "show this in a table"
    inherit the right tables from prior turns).
    """
    try:
        from talonsight.mcp_server import _get_core, _get_schema
        ts = _get_core()

        # Combine current question + recent history text for table scoring
        history_text = ""
        if history:
            for msg in history[-6:]:
                content = msg.get("content") or ""
                if not content and "result" in msg:
                    content = getattr(msg["result"], "summary", "") or ""
                if content:
                    history_text += f" {content}"

        search_text = question + " " + history_text
        q_tokens = set(re.split(r'\W+', search_text.lower()))
        q_tokens.discard('')

        scores: list[tuple[int, str]] = []
        for tbl in ts.schema_context.tables:
            tbl_tokens = set(re.split(r'[_\W]+', tbl.name.lower()))
            col_tokens: set[str] = set()
            for col in tbl.columns:
                col_tokens.update(re.split(r'[_\W]+', col.name.lower()))
            score = len(q_tokens & (tbl_tokens | col_tokens))
            scores.append((score, tbl.name))

        scores.sort(key=lambda x: (-x[0], x[1]))

        if scores[0][0] == 0:
            by_cols = sorted(ts.schema_context.tables, key=lambda t: len(t.columns), reverse=True)
            chosen = [t.name for t in by_cols[:3]]
        else:
            chosen = [name for score, name in scores[:3] if score > 0]
            if not chosen:
                chosen = [scores[0][1]]

        schema = _get_schema(chosen)

        # Build conversation context block for follow-up awareness
        conv_block = ""
        if history:
            conv_lines: list[str] = []
            for msg in history[-6:]:
                role = msg.get("role", "")
                content = msg.get("content") or ""
                if not content and "result" in msg:
                    content = getattr(msg["result"], "summary", "") or ""
                if content and role in ("user", "assistant"):
                    label = "User" if role == "user" else "Assistant"
                    snippet = content[:400] + "…" if len(content) > 400 else content
                    conv_lines.append(f"{label}: {snippet}")
            if conv_lines:
                conv_block = "RECENT CONVERSATION:\n" + "\n".join(conv_lines) + "\n\n"

        context = (
            f"You are a SQL data analyst. Use ONLY these tables:\n\n"
            f"{schema}\n\n"
            f"RULES:\n"
            f"- Call run_sql with a valid SELECT query.\n"
            f"- Use ONLY the tables listed above. Do NOT JOIN tables not listed.\n"
            f"- Use EXACT column names shown (quoted names like \"Customer ID\" stay quoted).\n"
            f"- GROUP BY and ORDER BY for ranked results.\n\n"
            f"{conv_block}"
            f"QUESTION: {question}"
        )
        return context
    except Exception:
        return question


_ANALYST_TICKS = [
    # (elapsed_seconds, message)
    (3,   "🔍 Examining schema…"),
    (8,   "📋 Identifying relevant tables…"),
    (15,  "⚙️  Querying database…"),
    (25,  "📊 Analysing results…"),
    (40,  "🧩 Cross-referencing data…"),
    (60,  "✍️  Composing answer…"),
    (90,  "⏳ Still working — complex question takes a moment…"),
    (120, "⏳ Almost there…"),
    (180, "⏳ Finalising…"),
]


def ask_hermes(
    question: str,
    output_cb: Optional[Callable[[str], None]] = None,
    timeout: int = 300,
    history: Optional[list[dict]] = None,
) -> str:
    """Run `hermes -z <question> -t talonsight` and return the final answer.

    hermes -z suppresses all output until the final answer is ready, so we
    emit synthetic timed progress ticks via output_cb to keep the UI alive.
    The talonsight MCP toolset gives Hermes access to the connected database.

    Returns the agent's final answer as a string, or an error message.
    """
    def emit(msg: str) -> None:
        if output_cb and msg.strip():
            output_cb(msg.strip())

    if not is_installed():
        return "❌ Hermes Agent is not installed. Please complete onboarding first."

    emit("🧠 Analyst starting…")

    # Prepend schema + recent chat history so the model has full context
    enriched_question = _build_enriched_prompt(question, history=history)

    try:
        proc = subprocess.Popen(
            ["hermes", "-z", enriched_question, "-t", "talonsight"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        return f"❌ Could not launch Hermes: {exc}"

    # ── Timed progress ticker ─────────────────────────────────────────────────
    # hermes -z redirects all internal output to /dev/null and only writes the
    # final answer to stdout at the very end.  We fire synthetic status ticks
    # on a background thread so the UI doesn't look frozen.
    _done_event = threading.Event()

    def _tick() -> None:
        import time as _time
        start = _time.monotonic()
        tick_idx = 0
        while not _done_event.wait(timeout=1.0):
            elapsed = _time.monotonic() - start
            if tick_idx < len(_ANALYST_TICKS):
                threshold, msg = _ANALYST_TICKS[tick_idx]
                if elapsed >= threshold:
                    emit(msg)
                    tick_idx += 1

    ticker = threading.Thread(target=_tick, daemon=True)
    ticker.start()

    # Collect stdout (final answer written by hermes at the very end)
    output_lines: list[str] = []
    stderr_lines: list[str] = []
    assert proc.stdout and proc.stderr

    def _drain_stderr() -> None:
        for raw in proc.stderr:
            line = _clean(raw)
            if line:
                stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    stderr_thread.start()

    for raw in proc.stdout:
        line = _clean(raw)
        if line:
            output_lines.append(line)

    _done_event.set()  # stop the ticker

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return "❌ Analysis timed out after 5 minutes."
    finally:
        ticker.join(timeout=3)
        stderr_thread.join(timeout=3)

    if proc.returncode != 0:
        diag = "\n".join(stderr_lines[-5:]) if stderr_lines else "no details"
        return f"❌ Hermes exited with code {proc.returncode}.\n{diag}"

    answer = "\n".join(output_lines).strip()

    # ── Intercept tool-call-as-text responses ─────────────────────────────────
    # hermes3 and similar small models often output MCP tool calls as scratchpad
    # JSON rather than invoking the protocol.  Detect this, execute the SQL
    # ourselves, then ask Hermes to narrate the results.
    if not answer or _extract_sql(answer):
        sql = _extract_sql(answer) if answer else None
        if sql:
            emit("⚙️  Executing SQL…")
            table, err = _execute_sql_safe(sql)
            if err:
                # SQL had hallucinated columns / WHERE clauses — tell user
                return (
                    f"Generated SQL had an error: {err}\n\n"
                    f"**SQL attempted:**\n```sql\n{sql}\n```"
                )
            if not table or table.strip() == "Query returned no rows.":
                return f"The query returned no rows.\n\n```sql\n{sql}\n```"

            emit("✍️  Composing answer…")
            narration = _narrate_result(question, sql, table)
            return narration

    return answer if answer else "No answer was returned."


def _narrate_result(question: str, sql: str, table: str) -> str:
    """Ask Hermes to write a natural-language answer given the SQL result data."""
    prompt = (
        f"A SQL query was run to answer this question: {question!r}\n\n"
        f"SQL used:\n```sql\n{sql}\n```\n\n"
        f"Results:\n{table}\n\n"
        f"Write a clear, concise answer to the question using the data above. "
        f"Mention specific numbers, top entries, and any notable patterns. "
        f"Do not re-state the SQL. 3-5 sentences max."
    )
    try:
        result = subprocess.run(
            ["hermes", "-z", prompt, "-t", "talonsight"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        narration = result.stdout.strip()
        if narration and not _extract_sql(narration):
            # Return narrative + collapsible data
            return f"{narration}\n\n---\n**Data:**\n{table}"
    except Exception:
        pass
    # Fallback: return the raw table with the SQL
    return f"**Results:**\n{table}\n\n```sql\n{sql}\n```"


def run_doctor(output_cb: Optional[Callable[[str], None]] = None) -> None:
    """Run `hermes doctor` and stream its output — useful for diagnostics."""
    def emit(msg: str) -> None:
        if output_cb:
            output_cb(msg)
    if not is_installed():
        emit("Hermes not installed — nothing to diagnose")
        return
    try:
        r = subprocess.run(
            ["hermes", "doctor"], capture_output=True, text=True, timeout=30
        )
        for line in (r.stdout + r.stderr).splitlines():
            clean = _clean(line)
            if clean:
                emit(clean)
    except Exception as exc:
        emit(f"hermes doctor error: {exc}")


# ── All-in-one setup ──────────────────────────────────────────────────────────

def ensure_analyst_ready(
    provider: str,
    model: str,
    url: str,
    api_key: str = "",
    output_cb: Optional[Callable[[str], None]] = None,
) -> bool:
    """Install → configure LLM → register MCP → verify.

    No persistent gateway is started.  Each user question is handled by
    a fresh `hermes -z "<question>" -t talonsight` invocation.

    Everything streams to output_cb for live display in Streamlit.
    Returns True when Hermes is installed, configured, and verified.
    """
    def emit(msg: str) -> None:
        logger.info("hermes-setup: %s", msg)
        if output_cb:
            output_cb(msg)

    # 1 — Install
    if is_installed():
        emit(f"✓  Hermes Agent already installed ({_hermes_version()})")
    else:
        emit("Installing Hermes Agent…")
        if not install(output_cb=output_cb):
            emit("❌  Installation failed. Click Retry to try again.")
            return False

    # 2 — Configure LLM (writes model: YAML block)
    emit(f"Configuring LLM ({provider} / {model})…")
    configure_llm(provider, model, url, api_key, output_cb=output_cb)

    # 3 — Register MCP (talonsight database tools)
    emit("Registering database tools…")
    register_mcp(output_cb=output_cb)

    emit("✓  Setup complete — Analyst engine ready")
    return True


# ── Onboarding reset ──────────────────────────────────────────────────────────

def reset_onboarding() -> None:
    """Reset the onboarding state so the wizard shows again on next launch."""
    from talonsight.preferences import Preferences
    prefs = Preferences.load()
    prefs.onboarding_complete = False
    prefs.save()
