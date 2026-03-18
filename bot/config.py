"""
Environment-based configuration management for the prediction market bot.

Loads all settings from a .env file (via python-dotenv) and exposes them as
typed attributes on a Pydantic BaseModel. This provides validation, defaults,
and a single source of truth for all configurable parameters across the system.

Configuration groups:
  - Kalshi API:     API key ID, private key (file or env var), demo/production toggle.
  - Anthropic:      API key for Claude AI market analysis.
  - Supabase:       URL and service key for persistent database storage.
  - Trading Params: Max bet size, min edge threshold, daily loss limit, max positions,
                    max events to analyze per scan.
  - Kelly Criterion: Fractional Kelly multiplier (default 0.5 = half-Kelly).
  - Notifications:  Slack and Discord webhook URLs for trade alerts.
  - Webhook:        Secret token for authenticating inbound webhook triggers.
  - Retrain:        Scheduled retraining days (e.g., mon,wed,fri) and hour (UTC).

The module instantiates a singleton `config` object at import time, which is
imported throughout the codebase. The Kalshi private key is resolved with a
priority chain: KALSHI_PRIVATE_KEY env var (for cloud deploys like Railway)
takes precedence over KALSHI_PRIVATE_KEY_PATH file (for local development).

Used by: Every other module in the bot package.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)


def _safe_int(key: str, default: int) -> int:
    """Parse an env var as int, falling back to default on invalid values."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _safe_float(key: str, default: float) -> float:
    """Parse an env var as float, falling back to default on invalid values."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


class Config(BaseModel):
    """Central configuration object loaded from environment variables (.env file).

    All fields have sensible defaults so the bot can start with minimal config.
    Only KALSHI_API_KEY_ID and a private key are strictly required for trading.
    """

    # ── Kalshi API Credentials ────────────────────────────────────────────────
    kalshi_api_key_id: str = os.getenv("KALSHI_API_KEY_ID", "")
    kalshi_private_key_path: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
    kalshi_private_key_raw: str = os.getenv("KALSHI_PRIVATE_KEY", "")  # PEM string for cloud deploy (Railway)
    kalshi_use_demo: bool = os.getenv("KALSHI_USE_DEMO", "true").lower() == "true"  # Demo vs production API

    # ── Anthropic API (Claude AI Analysis) ────────────────────────────────────
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ── Supabase Database ─────────────────────────────────────────────────────
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")  # Supabase service role key

    # ── Trading Parameters ────────────────────────────────────────────────────
    max_bet_amount_cents: int = _safe_int("MAX_BET_AMOUNT_CENTS", 2500)     # $25 max per trade
    min_edge_threshold: float = _safe_float("MIN_EDGE_THRESHOLD", 0.08)     # 8% minimum edge to trade
    max_daily_loss_cents: int = _safe_int("MAX_DAILY_LOSS_CENTS", 10000)    # $100 daily loss limit
    max_open_positions: int = _safe_int("MAX_OPEN_POSITIONS", 10)            # Max simultaneous positions
    max_events_to_analyze: int = _safe_int("MAX_EVENTS_TO_ANALYZE", 20)     # Events per on-demand scan

    # ── Kelly Criterion ───────────────────────────────────────────────────────
    kelly_fraction: float = min(1.0, max(0.01, _safe_float("KELLY_FRACTION", 0.5)))  # Half-Kelly, clamped [0.01, 1.0]

    # ── Notification Webhooks ─────────────────────────────────────────────────
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    # ── Inbound Webhook Authentication ────────────────────────────────────────
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")  # Shared secret for /api/webhook

    # ── Model Retrain Schedule ────────────────────────────────────────────────
    retrain_days: str = os.getenv("RETRAIN_DAYS", "mon,wed,fri")  # APScheduler cron day_of_week
    retrain_hour: int = min(23, max(0, _safe_int("RETRAIN_HOUR", 3)))  # Hour in UTC, clamped [0, 23]

    @property
    def kalshi_base_url(self) -> str:
        """Return the Kalshi API base URL based on the demo/production toggle."""
        if self.kalshi_use_demo:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://api.elections.kalshi.com/trade-api/v2"

    @property
    def kalshi_private_key(self) -> str:
        """Resolve the Kalshi private key PEM content.

        Priority: KALSHI_PRIVATE_KEY env var (for cloud deploys like Railway)
        > KALSHI_PRIVATE_KEY_PATH file (for local development).
        Returns empty string if neither is available.
        """
        if self.kalshi_private_key_raw:
            return self.kalshi_private_key_raw
        path = Path(self.kalshi_private_key_path)
        if path.exists():
            return path.read_text()
        return ""

    def validate_kalshi(self) -> bool:
        """Check if Kalshi API credentials are configured (key ID + private key)."""
        return bool(self.kalshi_api_key_id and self.kalshi_private_key)

    def validate_anthropic(self) -> bool:
        """Check if the Anthropic API key is configured for Claude AI analysis."""
        return bool(self.anthropic_api_key)

    def validate_supabase(self) -> bool:
        """Check if Supabase URL and key are configured for database persistence."""
        return bool(self.supabase_url and self.supabase_key)


# Singleton config instance — imported by all other modules
config = Config()
