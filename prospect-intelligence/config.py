"""
config.py — Application configuration loader.

Reads environment variables (via python-dotenv) and exposes a single
`Settings` instance that every module imports.  Fails loudly at startup
when required keys are missing so that problems surface before any API
call is attempted.

API keys are NEVER logged or displayed.

Supported LLM providers: "anthropic", "openai", "gemini"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate and load the .env file that sits next to this config.py
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_val(var: str, default: str = "") -> str:
    """Read a setting from os.environ or Streamlit secrets (for cloud deployment)."""
    val = os.getenv(var, "").strip()
    if not val:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and var in st.secrets:
                val = str(st.secrets[var]).strip()
        except Exception:
            pass
    return val or default


def _require(var: str) -> str:
    """Return the value of *var* or exit with a clear message."""
    value = _get_val(var)
    if not value:
        sys.exit(
            f"[config] Missing required environment variable or secret: {var}\n"
            f"         Set it in .env locally or in Streamlit Cloud Secrets (secrets.toml)."
        )
    return value


def _optional(var: str, default: str) -> str:
    return _get_val(var, default)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings:
    """Typed, validated application settings.  Import the singleton below."""

    # -- LLM provider --------------------------------------------------------
    LLM_PROVIDER: str   # "anthropic" | "openai" | "gemini"
    LLM_MODEL: str

    # -- API keys (never expose in logs or UI) --------------------------------
    TAVILY_API_KEY: str
    ANTHROPIC_API_KEY: str | None
    OPENAI_API_KEY: str | None
    GOOGLE_API_KEY: str | None

    # -- Behaviour -----------------------------------------------------------
    MAX_SEARCH_RESULTS: int
    REQUEST_TIMEOUT_SECONDS: int
    MAX_RETRIES: int

    # -- Paths ---------------------------------------------------------------
    DATABASE_PATH: Path
    EVAL_DATASET_PATH: Path

    _VALID_PROVIDERS = {"anthropic", "openai", "gemini"}

    # Default models per provider
    _DEFAULT_MODELS = {
        "anthropic": "claude-sonnet-4-5",
        "openai": "gpt-4o",
        "gemini": "gemini-2.5-flash",
    }

    def __init__(self) -> None:
        self.TAVILY_API_KEY = _require("TAVILY_API_KEY")

        self.LLM_PROVIDER = _optional("LLM_PROVIDER", "gemini").lower()
        if self.LLM_PROVIDER not in self._VALID_PROVIDERS:
            sys.exit(
                f"[config] LLM_PROVIDER must be one of {self._VALID_PROVIDERS}, "
                f"got: '{self.LLM_PROVIDER}'"
            )

        default_model = self._DEFAULT_MODELS[self.LLM_PROVIDER]
        self.LLM_MODEL = _optional("LLM_MODEL", default_model)

        # Require the key only for the active provider
        self.ANTHROPIC_API_KEY = _get_val("ANTHROPIC_API_KEY") or None
        self.OPENAI_API_KEY = _get_val("OPENAI_API_KEY") or None
        self.GOOGLE_API_KEY = _get_val("GOOGLE_API_KEY") or _get_val("GEMINI_API_KEY") or None

        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            sys.exit("[config] ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        elif self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            sys.exit("[config] OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        elif self.LLM_PROVIDER == "gemini" and not self.GOOGLE_API_KEY:
            sys.exit("[config] GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")

        # Numeric limits
        self.MAX_SEARCH_RESULTS = int(_optional("MAX_SEARCH_RESULTS", "10"))
        if not (1 <= self.MAX_SEARCH_RESULTS <= 20):
            sys.exit(
                f"[config] MAX_SEARCH_RESULTS must be between 1 and 20, "
                f"got: {self.MAX_SEARCH_RESULTS}"
            )

        self.REQUEST_TIMEOUT_SECONDS = int(
            _optional("REQUEST_TIMEOUT_SECONDS", "30")
        )
        self.MAX_RETRIES = int(_optional("MAX_RETRIES", "3"))

        # File paths (created lazily by the modules that use them)
        self.DATABASE_PATH = _PROJECT_ROOT / _optional(
            "DATABASE_PATH", "data/prospects.db"
        )
        self.EVAL_DATASET_PATH = _PROJECT_ROOT / _optional(
            "EVAL_DATASET_PATH", "data/eval_dataset.jsonl"
        )

    def __repr__(self) -> str:
        """Safe repr — API keys are never included."""
        return (
            f"Settings("
            f"provider={self.LLM_PROVIDER!r}, "
            f"model={self.LLM_MODEL!r}, "
            f"max_results={self.MAX_SEARCH_RESULTS}, "
            f"db={self.DATABASE_PATH}, "
            f"timeout={self.REQUEST_TIMEOUT_SECONDS}s"
            f")"
        )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
