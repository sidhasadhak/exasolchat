"""Preferences — single source of truth for talonsight configuration.

Stored at: ~/.talonsight/preferences.json
Wraps the older config.json format so everything is backwards-compatible.

Keys:
  mode               "assistant" | "analyst"
  onboarding_complete  bool
  llm_provider       "ollama" | "mlx" | "openai" | "anthropic" | "custom"
  llm_model          model name / id
  llm_url            base URL (Ollama, MLX, or custom endpoint)
  llm_api_key        API key (OpenAI / Anthropic / custom)
  hermes_installed   bool
  hermes_gateway_url e.g. "http://localhost:7860"
  last_connection    dict — connection config written on every DB connect
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DIR  = Path.home() / ".talonsight"
_PATH = _DIR / "preferences.json"

# Legacy config.json path (talonsight < 0.7) — merged on first load
_LEGACY_PATH = _DIR / "config.json"


@dataclass
class Preferences:
    # ── Core ──────────────────────────────────────────────────────────────────
    mode: str = "assistant"                       # "assistant" | "analyst"
    onboarding_complete: bool = False

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: str = "ollama"                  # "ollama"|"mlx"|"openai"|"anthropic"|"custom"
    llm_model: str = "hermes3:8b"
    llm_url: str = "http://localhost:11434"
    llm_api_key: str = ""

    # ── Hermes Agent (analyst mode only) ─────────────────────────────────────
    hermes_installed: bool = False
    hermes_gateway_url: str = "http://localhost:7860"

    # ── Last active DB connection (written by Streamlit; read by MCP server) ─
    last_connection: dict = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "Preferences":
        """Load from disk; merge legacy config.json if present."""
        _DIR.mkdir(parents=True, exist_ok=True)
        raw: dict = {}

        # Migrate from legacy config.json
        if _LEGACY_PATH.exists() and not _PATH.exists():
            try:
                raw = json.loads(_LEGACY_PATH.read_text(encoding="utf-8"))
                raw = _migrate_legacy(raw)
            except Exception as exc:
                logger.warning("Could not migrate legacy config: %s", exc)

        if _PATH.exists():
            try:
                stored = json.loads(_PATH.read_text(encoding="utf-8"))
                raw = {**raw, **stored}
            except Exception as exc:
                logger.warning("Could not load preferences: %s", exc)

        # Map raw dict to dataclass fields — ignore unknown keys
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in raw.items() if k in fields}
        return cls(**filtered)

    def save(self) -> None:
        """Persist to disk."""
        _DIR.mkdir(parents=True, exist_ok=True)
        try:
            _PATH.write_text(
                json.dumps(asdict(self), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error("Could not save preferences: %s", exc)

    def to_app_config(self) -> dict:
        """Return a dict in the format expected by the legacy setup_wizard / cli.py."""
        if self.llm_provider == "ollama":
            return {
                "llm_backend": "Ollama",
                "ollama_url":   self.llm_url,
                "ollama_model": self.llm_model,
                "mode":         self.mode,
            }
        if self.llm_provider == "mlx":
            return {
                "llm_backend": "MLX (Apple Silicon)",
                "mlx_url":     self.llm_url,
                "mlx_model":   self.llm_model,
                "mode":        self.mode,
            }
        if self.llm_provider in ("openai", "anthropic", "custom"):
            return {
                "llm_backend": "OpenAI-compatible API",
                "api_url":     self.llm_url,
                "api_model":   self.llm_model,
                "mode":        self.mode,
            }
        return {"mode": self.mode}

    @property
    def is_analyst(self) -> bool:
        return self.mode == "analyst"

    @property
    def is_assistant(self) -> bool:
        return self.mode == "assistant"


# ── Legacy migration ──────────────────────────────────────────────────────────

def _migrate_legacy(old: dict) -> dict:
    """Map old config.json keys to new preferences keys."""
    out: dict = {}
    backend = old.get("llm_backend", "")
    if "Ollama" in backend:
        out["llm_provider"] = "ollama"
        out["llm_url"]   = old.get("ollama_url", "http://localhost:11434")
        out["llm_model"] = old.get("ollama_model", "hermes3:8b")
    elif "MLX" in backend:
        out["llm_provider"] = "mlx"
        out["llm_url"]   = old.get("mlx_url", "http://localhost:8080/v1")
        out["llm_model"] = old.get("mlx_model", "mlx-community/Qwen3-8B-4bit")
    elif "OpenAI" in backend or "API" in backend:
        out["llm_provider"] = "custom"
        out["llm_url"]   = old.get("api_url", "http://localhost:1234/v1")
        out["llm_model"] = old.get("api_model", "local-model")
    # Legacy configs have no mode / onboarding — keep defaults
    return out
