"""Configuration management for the prediction market bot."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)


class Config(BaseModel):
    # Kalshi
    kalshi_api_key_id: str = os.getenv("KALSHI_API_KEY_ID", "")
    kalshi_private_key_path: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
    kalshi_private_key_raw: str = os.getenv("KALSHI_PRIVATE_KEY", "")  # For cloud deploy
    kalshi_use_demo: bool = os.getenv("KALSHI_USE_DEMO", "true").lower() == "true"

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")

    # Trading parameters
    max_bet_amount_cents: int = int(os.getenv("MAX_BET_AMOUNT_CENTS", "2500"))
    min_edge_threshold: float = float(os.getenv("MIN_EDGE_THRESHOLD", "0.08"))
    max_daily_loss_cents: int = int(os.getenv("MAX_DAILY_LOSS_CENTS", "10000"))
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "10"))
    max_events_to_analyze: int = int(os.getenv("MAX_EVENTS_TO_ANALYZE", "20"))

    # Kelly criterion
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.5"))  # Half-Kelly default

    # Notifications
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    # Webhook
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")

    # Retrain schedule
    retrain_days: str = os.getenv("RETRAIN_DAYS", "mon,wed,fri")
    retrain_hour: int = int(os.getenv("RETRAIN_HOUR", "3"))

    @property
    def kalshi_base_url(self) -> str:
        if self.kalshi_use_demo:
            return "https://demo-api.kalshi.co/trade-api/v2"
        return "https://api.elections.kalshi.com/trade-api/v2"

    @property
    def kalshi_private_key(self) -> str:
        # Prefer env var (for cloud), fall back to file (for local)
        if self.kalshi_private_key_raw:
            return self.kalshi_private_key_raw
        path = Path(self.kalshi_private_key_path)
        if path.exists():
            return path.read_text()
        return ""

    def validate_kalshi(self) -> bool:
        return bool(self.kalshi_api_key_id and self.kalshi_private_key)

    def validate_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    def validate_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)


config = Config()
