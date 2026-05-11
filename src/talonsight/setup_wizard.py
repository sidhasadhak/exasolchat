"""Interactive first-time setup wizard for talonsight.

Runs automatically on the first ``talonsight`` invocation (no config file found).
Re-run any time with ``talonsight --setup``.

Saved config: ~/.talonsight/config.json
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

# ── Config location ───────────────────────────────────────────────────────────
CONFIG_PATH = Path.home() / ".talonsight" / "config.json"

# ── Platform flag ─────────────────────────────────────────────────────────────
IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

# ── ANSI colour helpers ───────────────────────────────────────────────────────
_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[32m"
_CYAN  = "\033[36m"
_YELL  = "\033[33m"
_RED   = "\033[31m"
_RST   = "\033[0m"

def _b(s: str)  -> str: return f"{_BOLD}{s}{_RST}"
def _c(s: str)  -> str: return f"{_CYAN}{s}{_RST}"
def _g(s: str)  -> str: return f"{_GREEN}{s}{_RST}"
def _y(s: str)  -> str: return f"{_YELL}{s}{_RST}"
def _dim(s: str)-> str: return f"{_DIM}{s}{_RST}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user; return default on empty input or EOF."""
    try:
        val = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)
    return val if val else default


def _hr(char: str = "─", width: int = 54) -> str:
    return char * width


def _header() -> None:
    print()
    print(_b(_hr()))
    print(_b("  ⚡ talonsight — first-time setup"))
    print(_b(_hr()))
    print()


def _section(title: str) -> None:
    print(f"\n  {_b(title)}\n")


# ── Model download helpers ────────────────────────────────────────────────────

def _ensure_mlx_lm() -> bool:
    """Install mlx-lm into the current environment if not already present."""
    import importlib.util
    if importlib.util.find_spec("mlx_lm") is not None:
        return True
    print(f"\n  {_y('!')}  mlx_lm not installed — installing {_c('mlx-lm')} now…\n")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "mlx-lm", "transformers>=4.47"],
        check=False,
    )
    if result.returncode == 0:
        print(f"\n  {_g('✓')} mlx_lm installed.\n")
        return True
    print(f"\n  {_y('⚠')}  Install failed — run manually:  {sys.executable} -m pip install mlx-lm\n")
    return False


def _mlx_model_cached(model: str) -> bool:
    """Return True if the model weights already exist in the HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        repo_id_slug = model.replace("/", "--")
        for repo in info.repos:
            if repo_id_slug in repo.repo_id.replace("/", "--"):
                return True
        return False
    except Exception:
        # Fall back to checking the default cache path directly
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        slug = "models--" + model.replace("/", "--")
        return (cache_dir / slug).exists()


def _download_mlx(model: str) -> bool:
    """Cache MLX model weights from HuggingFace (requires huggingface_hub).

    Skips silently if already cached.  Returns True on success or already-cached.
    """
    if _mlx_model_cached(model):
        print(f"\n  {_g('✓')} Model already cached — skipping download.\n")
        return True

    print(f"\n  Downloading {_c(model)}")
    print(_dim("  (~5 GB — grab a coffee, this only happens once)\n"))
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=model,
            # skip non-MLX weight formats to save bandwidth
            ignore_patterns=["*.pt", "*.bin", "*.gguf", "original/*"],
        )
        print(f"\n  {_g('✓')} Model cached to ~/.cache/huggingface/hub/\n")
        return True
    except ImportError:
        # huggingface_hub ships with mlx_lm — if missing, install likely failed too.
        # mlx_lm server does NOT auto-download; it will crash without the model.
        print(
            f"\n  {_y('⚠')}  huggingface_hub not found.\n"
            f"  Download the model manually before starting talonsight:\n"
            f"      {_c('pip install huggingface_hub')}\n"
            f"      {_c(f'huggingface-cli download {model}')}\n"
        )
        return False
    except Exception as exc:
        print(
            f"\n  {_y('⚠')}  Download failed: {exc}\n"
            f"  Download manually before starting talonsight:\n"
            f"      {_c(f'huggingface-cli download {model}')}\n"
        )
        return False


def _pull_ollama(model: str) -> bool:
    """Pull an Ollama model (requires the Ollama daemon to be running)."""
    print(f"\n  Pulling {_c(model)} via Ollama…\n")
    try:
        subprocess.run(["ollama", "pull", model], check=True)
        print(f"\n  {_g('✓')} Model ready.\n")
        return True
    except FileNotFoundError:
        print(
            f"  {_y('⚠')}  Ollama not found on PATH.\n"
            "  Install it from https://ollama.com, then run:\n"
            f"      ollama pull {model}\n"
        )
        return False
    except subprocess.CalledProcessError:
        print(
            f"  {_y('⚠')}  Pull failed — is the Ollama daemon running?\n"
            f"  Retry manually: ollama pull {model}\n"
        )
        return False


# ── Config I/O ────────────────────────────────────────────────────────────────

def save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def load_config() -> dict:
    """Return saved config dict, or {} if none exists yet."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}


# ── Main wizard ───────────────────────────────────────────────────────────────

def run_setup() -> None:
    """Interactive first-time (or forced) setup wizard."""
    _header()

    # ── Platform banner ──────────────────────────────────────────────
    if IS_APPLE_SILICON:
        print(f"  Detected: {_c('Apple Silicon Mac')}  {_dim('(Metal / MLX ready)')}\n")
        options = [
            # (label, backend key, model, short note, tag)
            ("MLX (local)",  "MLX (Apple Silicon)", "mlx-community/Qwen3-8B-4bit",
             "fully local · Metal GPU · ~5 GB download",  "recommended"),
            ("Ollama",       "Ollama",               "qwen3:8b",
             "local via Ollama daemon",                    ""),
            ("Skip",         "",                     "",
             "I'll configure manually inside the app",     ""),
        ]
    elif sys.platform == "win32":
        print(f"  Detected: {_c('Windows')}\n")
        options = [
            ("Ollama",              "Ollama",               "qwen3:8b",
             "local via Ollama daemon",              "recommended"),
            ("OpenAI-compatible",   "OpenAI-compatible API","",
             "any OpenAI-compatible endpoint",       ""),
            ("Skip",                "",              "",
             "I'll configure manually inside the app",""),
        ]
    else:
        plat = "Linux" if sys.platform.startswith("linux") else "Intel Mac / other"
        print(f"  Detected: {_c(plat)}\n")
        options = [
            ("Ollama",              "Ollama",               "qwen3:8b",
             "local via Ollama daemon",              "recommended"),
            ("OpenAI-compatible",   "OpenAI-compatible API","",
             "any OpenAI-compatible endpoint",       ""),
            ("Skip",                "",              "",
             "I'll configure manually inside the app",""),
        ]

    # ── LLM choice ───────────────────────────────────────────────────
    _section("Choose your LLM backend")
    for i, (label, _, model, note, tag) in enumerate(options, 1):
        tag_str   = f"  {_g('← ' + tag)}" if tag else ""
        model_str = f"  ·  {_c(model)}" if model else ""
        print(f"    [{i}]  {_b(label)}{model_str}")
        print(f"         {_dim(note)}{tag_str}\n")

    choice = _ask(f"  Your choice [1]: ", "1")
    try:
        idx = max(0, min(int(choice) - 1, len(options) - 1))
    except ValueError:
        idx = 0

    label, backend, model, _, _ = options[idx]
    cfg: dict = {}

    # ── Per-backend flow ─────────────────────────────────────────────
    if backend == "MLX (Apple Silicon)":
        cfg = {
            "llm_backend": backend,
            "mlx_url":     "http://localhost:8080/v1",
            "mlx_model":   model,
        }
        _ensure_mlx_lm()
        _download_mlx(model)
        print(f"  {_g('✓')} The MLX server will start automatically each time you run {_b('talonsight')}.\n")

    elif backend == "Ollama":
        cfg = {
            "llm_backend":  backend,
            "ollama_url":   "http://localhost:11434",
            "ollama_model": model,
        }
        _pull_ollama(model)

    elif backend == "OpenAI-compatible API":
        api_url   = _ask("  API URL   [http://localhost:1234/v1]: ", "http://localhost:1234/v1")
        api_model = _ask("  Model     [local-model]: ",              "local-model")
        cfg = {
            "llm_backend": backend,
            "api_url":     api_url,
            "api_model":   api_model,
        }

    # ── Save & done ──────────────────────────────────────────────────
    if cfg:
        save_config(cfg)
        print(f"  {_g('✓')} Config saved → {_dim(str(CONFIG_PATH))}\n")

    print(f"  {_b('Launching talonsight…')}\n")
    print(_dim(_hr()) + "\n")
