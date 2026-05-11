"""CLI entrypoint — runs first-time setup wizard if needed, then launches the Streamlit UI.

If the saved config selects MLX, the MLX server is started automatically in the
background, waited on until it is ready, and terminated when Streamlit exits.
"""

from __future__ import annotations

import atexit
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


# ── MLX server lifecycle ──────────────────────────────────────────────────────

def _extract_port(url: str) -> int:
    """Parse port from a URL string; default 8080."""
    try:
        return urlparse(url).port or 8080
    except Exception:
        return 8080


def _mlx_already_running(url: str) -> bool:
    """Return True if an MLX (or compatible) server is already answering."""
    try:
        import httpx
        httpx.get(f"{url}/models", timeout=2.0)
        return True
    except Exception:
        return False


def _start_mlx_server(model: str, port: int) -> "subprocess.Popen[bytes]":
    """Spawn mlx_lm server as a background process."""
    return subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server",
         "--model", model, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_mlx(url: str, proc: "subprocess.Popen[bytes]",
                  timeout: int = 120) -> bool:
    """Poll until the server responds or times out."""
    import httpx

    deadline = time.time() + timeout
    dots = 0
    while time.time() < deadline:
        # Check if the process crashed
        if proc.poll() is not None:
            print("\n  \033[31m✗\033[0m  MLX server exited unexpectedly.")
            return False

        try:
            httpx.get(f"{url}/models", timeout=1.5)
            return True
        except Exception:
            pass

        time.sleep(1)
        dots += 1
        if dots % 5 == 0:
            print("  … still loading model into memory", flush=True)

    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from talonsight.setup_wizard import CONFIG_PATH, load_config, run_setup

    _BOLD  = "\033[1m"
    _GREEN = "\033[32m"
    _YELL  = "\033[33m"
    _CYAN  = "\033[36m"
    _RST   = "\033[0m"

    # ── Strip talonsight-specific flags before forwarding args to Streamlit ──
    args = sys.argv[1:]
    force_setup = "--setup" in args
    args = [a for a in args if a != "--setup"]

    # ── First-time setup wizard ───────────────────────────────────────────
    if force_setup or not CONFIG_PATH.exists():
        run_setup()

    cfg = load_config()

    # ── Auto-start MLX server if configured ──────────────────────────────
    mlx_proc = None
    if cfg.get("llm_backend") == "MLX (Apple Silicon)":
        model = cfg.get("mlx_model", "mlx-community/Qwen3-8B-4bit")
        mlx_url = cfg.get("mlx_url", "http://localhost:8080/v1")
        port = _extract_port(mlx_url)

        # Non-fatal warning — always launch Streamlit so user can fix in sidebar
        def _mlx_warn(reason: str, hint: str = "") -> None:
            print(f"\n  \033[31m✗\033[0m  {reason}")
            if hint:
                print(f"  {_YELL}{hint}{_RST}")
            print(f"\n  {_YELL}Starting the app anyway — change LLM backend in the sidebar.{_RST}\n")

        # Pre-flight 1: ensure mlx_lm is installed
        import importlib.util, importlib
        _mlx_ok = True
        if importlib.util.find_spec("mlx_lm") is None:
            print(f"\n  {_YELL}!{_RST}  mlx_lm not found — installing {_CYAN}mlx-lm{_RST} now…\n")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "mlx-lm", "transformers>=4.47"],
                check=False,
            )
            importlib.invalidate_caches()
            if result.returncode != 0 or importlib.util.find_spec("mlx_lm") is None:
                _mlx_warn(
                    "mlx_lm could not be installed.",
                    f"Run manually:  {sys.executable} -m pip install mlx-lm",
                )
                _mlx_ok = False
            else:
                print(f"  {_GREEN}✓{_RST}  mlx_lm installed.\n")

        # Pre-flight 2: ensure model weights are on disk — download if missing
        if _mlx_ok:
            from talonsight.setup_wizard import _mlx_model_cached, _download_mlx
            if not _mlx_model_cached(model):
                print(f"\n  {_YELL}!{_RST}  Model {_CYAN}{model}{_RST} not found in cache — downloading now…")
                if not _download_mlx(model):
                    _mlx_warn(
                        f"Model '{model}' could not be downloaded.",
                        f"Download manually:  huggingface-cli download {model}",
                    )
                    _mlx_ok = False

        if _mlx_ok:
            if _mlx_already_running(mlx_url):
                print(f"  {_GREEN}✓{_RST}  MLX server already running at {mlx_url}")
            else:
                print(f"\n  {_BOLD}Starting MLX server{_RST} — {_CYAN}{model}{_RST}")
                print(f"  {_YELL}(loading model into Apple Silicon memory — takes ~30 s the first time){_RST}\n")

                mlx_proc = _start_mlx_server(model, port)

                # Register cleanup so the server dies when Streamlit exits
                def _kill_mlx() -> None:
                    if mlx_proc and mlx_proc.poll() is None:
                        mlx_proc.terminate()
                        try:
                            mlx_proc.wait(timeout=5)
                        except Exception:
                            mlx_proc.kill()

                atexit.register(_kill_mlx)

                ready = _wait_for_mlx(mlx_url, mlx_proc)
                if ready:
                    print(f"\n  {_GREEN}✓{_RST}  MLX server ready at {mlx_url}\n")
                else:
                    _kill_mlx()
                    _mlx_warn(
                        "MLX server did not become ready in time.",
                        f"Check port {port} is free, then run manually:\n"
                        f"    python -m mlx_lm server --model {model} --port {port}",
                    )

    # ── Launch Streamlit ──────────────────────────────────────────────────
    app_path = Path(__file__).with_name("app.py")
    sys.exit(
        subprocess.call(
            ["streamlit", "run", str(app_path), "--", *args],
        )
    )
