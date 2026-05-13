"""Streamlit onboarding wizard for TalonSight.

Shown on first launch (when preferences.onboarding_complete is False).
Guides the user through:
  Screen 0 — Mode selection  (Assistant vs Analyst)
  Screen 1 — LLM selection   (only for Analyst mode)
  Screen 2 — Automated setup (Hermes install + configure)
  Screen 3 — Done            (transition to main app)

State lives in st.session_state under the "onboarding_" prefix.
Call render_onboarding_wizard() from app.py; it calls st.stop() so the
rest of the app never renders until setup is complete.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Optional

import streamlit as st


# ── Wizard entry point ────────────────────────────────────────────────────────

def render_onboarding_wizard() -> None:
    """Render the appropriate wizard screen. Calls st.stop() when done."""
    _inject_wizard_styles()

    # Initialise session state
    if "onboarding_step" not in st.session_state:
        st.session_state.onboarding_step = 0
    if "onboarding_mode" not in st.session_state:
        st.session_state.onboarding_mode = "analyst"
    if "onboarding_provider" not in st.session_state:
        st.session_state.onboarding_provider = "ollama"
    if "onboarding_model" not in st.session_state:
        st.session_state.onboarding_model = "hermes3:8b"
    if "onboarding_url" not in st.session_state:
        st.session_state.onboarding_url = "http://localhost:11434"
    if "onboarding_api_key" not in st.session_state:
        st.session_state.onboarding_api_key = ""

    step = st.session_state.onboarding_step

    # Centre column — wizard card
    _, col, _ = st.columns([1, 2, 1])
    with col:
        if step == 0:
            _screen_mode()
        elif step == 1:
            _screen_llm()
        elif step == 2:
            _screen_setup()
        elif step == 3:
            _screen_done()

    st.stop()


# ── Screen 0 — Mode ───────────────────────────────────────────────────────────

def _screen_mode() -> None:
    st.markdown("""
    <div class="wiz-header">
        <div class="wiz-logo">⚡</div>
        <h1 class="wiz-title">Welcome to TalonSight</h1>
        <p class="wiz-subtitle">Your data, explained — choose how you want to work.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### How should TalonSight work with your data?")
    st.markdown("<br>", unsafe_allow_html=True)

    mode = st.session_state.onboarding_mode

    col_a, col_b = st.columns(2)

    with col_a:
        analyst_cls = "mode-card mode-card-selected" if mode == "analyst" else "mode-card"
        st.markdown(f"""
        <div class="{analyst_cls}" onclick="">
            <div class="mode-icon">🧠</div>
            <div class="mode-name">Analyst <span class="mode-badge">Recommended</span></div>
            <div class="mode-desc">
                Autonomous investigation. Thinks like a senior analyst —
                decomposes questions, tests hypotheses, synthesises findings.
                Learns from every session. Runs entirely on your machine.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choose Analyst", key="choose_analyst", use_container_width=True,
                     type="primary" if mode == "analyst" else "secondary"):
            st.session_state.onboarding_mode = "analyst"
            st.rerun()

    with col_b:
        assistant_cls = "mode-card mode-card-selected" if mode == "assistant" else "mode-card"
        st.markdown(f"""
        <div class="{assistant_cls}" onclick="">
            <div class="mode-icon">💬</div>
            <div class="mode-name">Assistant</div>
            <div class="mode-desc">
                Direct SQL answers. Ask a question, get a result.
                Fast and lightweight — no additional setup required.
                Good for developers and power users.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Choose Assistant", key="choose_assistant", use_container_width=True,
                     type="primary" if mode == "assistant" else "secondary"):
            st.session_state.onboarding_mode = "assistant"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    btn_label = "Continue →" if mode == "analyst" else "Finish Setup →"
    if st.button(btn_label, key="mode_continue", use_container_width=True, type="primary"):
        if mode == "analyst":
            st.session_state.onboarding_step = 1  # → LLM screen
        else:
            # Assistant mode — save and done
            _save_preferences(
                mode="assistant",
                provider=st.session_state.onboarding_provider,
                model=st.session_state.onboarding_model,
                url=st.session_state.onboarding_url,
                api_key="",
                hermes_installed=False,
            )
            st.session_state.onboarding_step = 3  # → Done
        st.rerun()


# ── Screen 1 — LLM selection ──────────────────────────────────────────────────

_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"

_LLM_OPTIONS = [
    {
        "id":      "ollama",
        "label":   "Local — Ollama",
        "badge":   "Recommended",
        "icon":    "🦙",
        "desc":    "Runs on your machine. No data leaves your network. Best for sensitive data.",
        "model":   "hermes3:8b",
        "url":     "http://localhost:11434",
        "key_req": False,
    },
    {
        "id":      "mlx",
        "label":   "Local — MLX",
        "badge":   "Apple Silicon only" if _IS_APPLE_SILICON else "",
        "icon":    "🍎",
        "desc":    "Fastest local option on Apple Silicon. Requires macOS arm64.",
        "model":   "mlx-community/Qwen3-8B-4bit",
        "url":     "http://localhost:8080/v1",
        "key_req": False,
        "disabled": not _IS_APPLE_SILICON,
    },
    {
        "id":      "openai",
        "label":   "OpenAI",
        "badge":   "",
        "icon":    "✦",
        "desc":    "GPT-4o via OpenAI API. Fastest cloud option. Requires API key.",
        "model":   "gpt-4o",
        "url":     "https://api.openai.com/v1",
        "key_req": True,
    },
    {
        "id":      "anthropic",
        "label":   "Anthropic",
        "badge":   "",
        "icon":    "◆",
        "desc":    "Claude via Anthropic API. Strong reasoning. Requires API key.",
        "model":   "claude-3-5-sonnet-20241022",
        "url":     "https://api.anthropic.com",
        "key_req": True,
    },
    {
        "id":      "custom",
        "label":   "Custom endpoint",
        "badge":   "",
        "icon":    "⚙",
        "desc":    "Any OpenAI-compatible API — LM Studio, vLLM, Groq, Together, etc.",
        "model":   "local-model",
        "url":     "http://localhost:1234/v1",
        "key_req": False,
    },
]


def _screen_llm() -> None:
    st.markdown("""
    <div class="wiz-header">
        <h2 class="wiz-title" style="font-size:1.6rem">Where should your analyst think?</h2>
        <p class="wiz-subtitle">Choose a language model. You can change this later in Settings.</p>
    </div>
    """, unsafe_allow_html=True)

    provider = st.session_state.onboarding_provider

    for opt in _LLM_OPTIONS:
        disabled = opt.get("disabled", False)
        is_selected = provider == opt["id"]
        card_cls = "llm-card llm-card-selected" if is_selected else "llm-card"
        if disabled:
            card_cls += " llm-card-disabled"

        badge_html = f'<span class="llm-badge">{opt["badge"]}</span>' if opt["badge"] else ""
        st.markdown(f"""
        <div class="{card_cls}">
            <span class="llm-icon">{opt["icon"]}</span>
            <span class="llm-name">{opt["label"]}{badge_html}</span>
            <span class="llm-desc">{opt["desc"]}</span>
        </div>
        """, unsafe_allow_html=True)

        if not disabled:
            btn_type = "primary" if is_selected else "secondary"
            if st.button(
                f"{'✓  ' if is_selected else ''}Select {opt['label']}",
                key=f"llm_{opt['id']}",
                use_container_width=True,
                type=btn_type,
                disabled=disabled,
            ):
                st.session_state.onboarding_provider = opt["id"]
                st.session_state.onboarding_model    = opt["model"]
                st.session_state.onboarding_url      = opt["url"]
                st.rerun()
        st.markdown("")

    # Show additional fields for selected provider
    selected_opt = next((o for o in _LLM_OPTIONS if o["id"] == provider), _LLM_OPTIONS[0])

    with st.expander("⚙  Advanced — model / endpoint", expanded=selected_opt.get("key_req", False)):
        if provider in ("ollama", "mlx", "custom"):
            st.session_state.onboarding_url = st.text_input(
                "API endpoint URL", value=st.session_state.onboarding_url, key="adv_url"
            )
        st.session_state.onboarding_model = st.text_input(
            "Model name", value=st.session_state.onboarding_model, key="adv_model"
        )
        if selected_opt.get("key_req"):
            st.session_state.onboarding_api_key = st.text_input(
                "API key", value=st.session_state.onboarding_api_key,
                type="password", key="adv_key"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back", key="llm_back", use_container_width=True):
            st.session_state.onboarding_step = 0
            st.rerun()
    with c2:
        if st.button("Continue →", key="llm_continue", use_container_width=True, type="primary"):
            st.session_state.onboarding_step = 2
            st.rerun()


# ── Screen 2 — Automated setup ────────────────────────────────────────────────

def _screen_setup() -> None:
    st.markdown("""
    <div class="wiz-header">
        <h2 class="wiz-title" style="font-size:1.6rem">Setting up your analyst…</h2>
        <p class="wiz-subtitle">This takes about 2 minutes — once. Never again.</p>
    </div>
    """, unsafe_allow_html=True)

    provider = st.session_state.onboarding_provider
    model    = st.session_state.onboarding_model
    url      = st.session_state.onboarding_url
    api_key  = st.session_state.onboarding_api_key

    # For non-Hermes providers (OpenAI/Anthropic/custom), no Hermes setup needed
    needs_hermes = provider in ("ollama", "mlx", "custom")

    log_area = st.empty()
    progress = st.progress(0)
    log_lines: list[str] = []

    def _update_log(line: str) -> None:
        log_lines.append(line)
        log_area.markdown(
            "\n".join(f"```\n{l}\n```" if False else f"› {l}" for l in log_lines[-20:])
        )

    success = True

    if needs_hermes:
        from talonsight import hermes_bootstrap as hb

        steps = [
            ("Checking Ollama" if provider == "ollama" else "Checking environment", 10),
            ("Installing Hermes Agent", 40),
            ("Configuring LLM", 65),
            ("Registering database tools", 85),
            ("Verifying analyst engine", 95),
        ]

        # Step 0 — pre-flight
        _update_log(steps[0][0] + "…")
        progress.progress(steps[0][1])

        if provider == "ollama":
            import shutil
            if not shutil.which("ollama"):
                _update_log("⚠  Ollama not found — install from https://ollama.com, then re-run setup.")
                _update_log("   Continuing without Ollama check…")
            else:
                _update_log("✓  Ollama found")
                # Pull model if needed
                import subprocess
                _update_log(f"Pulling model {model}…")
                r = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True, text=True
                )
                if r.returncode == 0:
                    _update_log(f"✓  Model {model} ready")
                else:
                    _update_log(f"⚠  Could not pull {model}: {r.stderr[:120]}")

        time.sleep(0.3)

        # Steps 1-4 — Hermes setup
        _update_log(steps[1][0] + "…")
        progress.progress(steps[1][1])

        success = hb.ensure_analyst_ready(
            provider=provider,
            model=model,
            url=url,
            api_key=api_key,
            output_cb=_update_log,
        )

        progress.progress(100)

    else:
        # Cloud provider — no Hermes needed, just save config
        _update_log(f"Configuring {provider} / {model}…")
        progress.progress(50)
        time.sleep(0.5)
        _update_log("✓  LLM configured")
        progress.progress(100)
        _update_log("✓  Ready (cloud mode — no local engine required)")

    # Save preferences
    from talonsight import hermes_bootstrap as hb
    _save_preferences(
        mode="analyst",
        provider=provider,
        model=model,
        url=url,
        api_key=api_key,
        hermes_installed=hb.is_installed() if needs_hermes else False,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if success:
        st.success("Your analyst is ready.")
        if st.button("Continue →", key="setup_continue", use_container_width=True, type="primary"):
            st.session_state.onboarding_step = 3
            st.rerun()
    else:
        st.warning(
            "Setup completed with warnings. You can still use TalonSight — "
            "some Analyst features may fall back to Assistant mode."
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← Retry", key="setup_retry", use_container_width=True):
                st.session_state.onboarding_step = 1
                st.rerun()
        with c2:
            if st.button("Continue anyway →", key="setup_skip", use_container_width=True, type="primary"):
                st.session_state.onboarding_step = 3
                st.rerun()


# ── Screen 3 — Done ───────────────────────────────────────────────────────────

def _screen_done() -> None:
    mode = st.session_state.get("onboarding_mode", "assistant")
    is_analyst = mode == "analyst"

    st.markdown(f"""
    <div class="wiz-done">
        <div class="wiz-done-icon">✓</div>
        <h2 class="wiz-title">{"Your analyst is ready." if is_analyst else "TalonSight is ready."}</h2>
        <p class="wiz-subtitle">
            {"Connect a database to start your first investigation." if is_analyst
             else "Connect a database to start asking questions."}
        </p>
    </div>
    """, unsafe_allow_html=True)

    if is_analyst:
        st.markdown("""
        <div class="wiz-features">
            <div class="wiz-feature">🔍 Autonomous investigation — decomposes complex questions</div>
            <div class="wiz-feature">🧠 Learns from every session — gets smarter over time</div>
            <div class="wiz-feature">🔒 Fully local — your data never leaves your machine</div>
            <div class="wiz-feature">📊 Narrative answers — not just tables</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Connect to a database →", key="done_connect",
                 use_container_width=True, type="primary"):
        # Mark onboarding complete and reload
        from talonsight.preferences import Preferences
        prefs = Preferences.load()
        prefs.onboarding_complete = True
        prefs.save()
        # Clear wizard state
        for k in list(st.session_state.keys()):
            if k.startswith("onboarding_"):
                del st.session_state[k]
        st.rerun()


def render_reset_setup_button() -> None:
    """Small 'Re-run Setup' button for the Settings sidebar."""
    if st.button("⚙  Re-run Setup Wizard", key="rerun_setup"):
        from talonsight.hermes_bootstrap import reset_onboarding
        reset_onboarding()
        for k in list(st.session_state.keys()):
            if k.startswith("onboarding_"):
                del st.session_state[k]
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_preferences(
    mode: str,
    provider: str,
    model: str,
    url: str,
    api_key: str,
    hermes_installed: bool,
) -> None:
    from talonsight.preferences import Preferences
    prefs = Preferences.load()
    prefs.mode = mode
    prefs.llm_provider = provider
    prefs.llm_model = model
    prefs.llm_url = url
    prefs.llm_api_key = api_key
    prefs.hermes_installed = hermes_installed
    # Don't mark onboarding_complete yet — done on Screen 3 button
    prefs.save()

    # Also persist to legacy config.json so cli.py / setup_wizard keep working
    from talonsight.setup_wizard import save_config
    save_config(prefs.to_app_config())


# ── Styles ────────────────────────────────────────────────────────────────────

def _inject_wizard_styles() -> None:
    st.markdown("""
    <style>
    /* ── Wizard layout ─────────────────────────────────────────────────── */
    .wiz-header { text-align: center; padding: 2rem 0 1.5rem; }
    .wiz-logo   { font-size: 3rem; margin-bottom: 0.5rem; }
    .wiz-title  {
        font-size: 2rem; font-weight: 700;
        color: var(--cs-text-primary, #ebebf0);
        margin: 0 0 0.5rem;
    }
    .wiz-subtitle {
        color: var(--cs-text-secondary, #a4a4ad);
        font-size: 1rem; margin: 0;
    }

    /* ── Mode cards ────────────────────────────────────────────────────── */
    .mode-card {
        border: 2px solid var(--cs-border, #424650);
        border-radius: 12px;
        padding: 1.5rem 1.25rem;
        margin-bottom: 0.75rem;
        background: var(--cs-bg-container, #1b232d);
        transition: border-color 0.15s ease;
        min-height: 180px;
    }
    .mode-card-selected {
        border-color: var(--cs-blue, #42b4ff) !important;
        background: rgba(66,180,255,0.06) !important;
    }
    .mode-icon  { font-size: 2rem; display: block; margin-bottom: 0.6rem; }
    .mode-name  {
        font-size: 1.1rem; font-weight: 700;
        color: var(--cs-text-primary, #ebebf0);
        margin-bottom: 0.5rem; display: block;
    }
    .mode-badge {
        font-size: 0.65rem; font-weight: 700;
        background: rgba(41,173,127,0.15); color: #29ad7f;
        border: 1px solid rgba(41,173,127,0.3);
        border-radius: 20px; padding: 2px 8px;
        margin-left: 0.5rem; vertical-align: middle;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .mode-desc  { color: var(--cs-text-secondary, #a4a4ad); font-size: 0.875rem; line-height: 1.5; }

    /* ── LLM option cards ──────────────────────────────────────────────── */
    .llm-card {
        display: flex; align-items: flex-start; gap: 0.75rem;
        border: 1px solid var(--cs-border, #424650);
        border-radius: 8px; padding: 0.875rem 1rem;
        margin-bottom: 0.5rem;
        background: var(--cs-bg-container, #1b232d);
    }
    .llm-card-selected {
        border-color: var(--cs-blue, #42b4ff) !important;
        background: rgba(66,180,255,0.06) !important;
    }
    .llm-card-disabled { opacity: 0.4; }
    .llm-icon  { font-size: 1.25rem; flex-shrink: 0; margin-top: 1px; }
    .llm-name  {
        font-weight: 700; font-size: 0.95rem;
        color: var(--cs-text-primary, #ebebf0); min-width: 160px;
    }
    .llm-badge {
        font-size: 0.6rem; font-weight: 700;
        background: rgba(41,173,127,0.15); color: #29ad7f;
        border: 1px solid rgba(41,173,127,0.3);
        border-radius: 20px; padding: 1px 7px;
        margin-left: 0.5rem; vertical-align: middle;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .llm-desc  { color: var(--cs-text-secondary, #a4a4ad); font-size: 0.8rem; line-height: 1.4; }

    /* ── Done screen ───────────────────────────────────────────────────── */
    .wiz-done { text-align: center; padding: 3rem 0 2rem; }
    .wiz-done-icon {
        width: 72px; height: 72px; border-radius: 50%;
        background: rgba(41,173,127,0.15);
        border: 2px solid rgba(41,173,127,0.4);
        color: #29ad7f; font-size: 2rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
        margin: 0 auto 1.5rem;
    }
    .wiz-features {
        margin: 1.5rem auto; max-width: 380px; text-align: left;
    }
    .wiz-feature {
        color: var(--cs-text-secondary, #a4a4ad);
        font-size: 0.875rem; padding: 0.4rem 0;
        border-bottom: 1px solid var(--cs-border-subtle, #232b37);
    }
    .wiz-feature:last-child { border-bottom: none; }
    </style>
    """, unsafe_allow_html=True)
