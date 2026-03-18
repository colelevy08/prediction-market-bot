"""
Notification system for trade alerts via Slack and Discord webhooks.

Sends formatted alert messages for key bot events:
  - Trade entries:  Ticker, side, price, size, and signal source (RF or AI).
  - Trade exits:    Entry/exit prices, P&L in dollars.
  - New signals:    Ticker, side, edge percentage, and confidence level.
  - Model retrains: Number of training samples and cross-validation accuracy.
  - Test messages:  Connectivity verification for configured channels.

Messages use Markdown formatting (Discord-native, converted to Slack bold syntax).
Both Slack and Discord channels are optional; the bot works without any notifications
configured. All webhook calls use a 10-second timeout and log errors without crashing.

Connects to: Slack Incoming Webhooks API, Discord Webhooks API (via httpx HTTP client).
Configuration: SLACK_WEBHOOK_URL and DISCORD_WEBHOOK_URL environment variables.
Used by: bot.server (auto-trade alerts, notification test endpoint).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from bot.config import config

logger = logging.getLogger("predictionbot")


class Notifier:
    """Sends formatted alert messages to Slack and/or Discord via webhooks.

    All notification methods are fire-and-forget: they log errors but never
    raise exceptions, so notification failures never block trading operations.
    """

    def __init__(self):
        """Initialize with a shared HTTP client (10s timeout for webhook calls)."""
        self._client = httpx.Client(timeout=10)

    @property
    def channels(self) -> list[str]:
        """List of configured notification channel names (e.g., ['slack', 'discord'])."""
        ch = []
        if config.slack_webhook_url:
            ch.append("slack")
        if config.discord_webhook_url:
            ch.append("discord")
        return ch

    @property
    def is_configured(self) -> bool:
        return len(self.channels) > 0

    def notify_trade_entry(self, ticker: str, side: str, price: float, size_cents: int, source: str = "RF"):
        """Send a trade entry alert with ticker, side, price, size, and signal source."""
        msg = (
            f"**ENTRY** {ticker}\n"
            f"Side: {side.upper()} | Price: {price*100:.0f}c | Size: ${size_cents/100:.2f}\n"
            f"Source: {source} | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        self._send(msg)

    def notify_trade_exit(self, ticker: str, side: str, entry_price: float, exit_price: float, pnl_cents: int):
        """Send a trade exit alert with entry/exit prices and P&L."""
        emoji = "+" if pnl_cents >= 0 else ""
        msg = (
            f"**EXIT** {ticker}\n"
            f"Side: {side.upper()} | Entry: {entry_price*100:.0f}c -> Exit: {exit_price*100:.0f}c\n"
            f"P&L: {emoji}${pnl_cents/100:.2f} | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        self._send(msg)

    def notify_signal(self, ticker: str, side: str, edge: float, confidence: float):
        """Send a new trading signal alert with edge and confidence percentages."""
        msg = (
            f"**SIGNAL** {ticker}\n"
            f"Side: {side.upper()} | Edge: {edge*100:.1f}% | Confidence: {confidence*100:.0f}%"
        )
        self._send(msg)

    def notify_retrain(self, samples: int, cv_accuracy: float):
        """Send a model retrain completion alert with sample count and accuracy."""
        msg = (
            f"**MODEL RETRAINED**\n"
            f"Samples: {samples} | CV Accuracy: {cv_accuracy*100:.1f}%"
        )
        self._send(msg)

    def send_test(self) -> dict:
        """Send a test notification to verify connectivity. Returns channel results."""
        msg = f"**PredictionBot Test** - Notifications working! {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        results = self._send(msg)
        return {"channels": self.channels, "results": results}

    def _send(self, message: str) -> dict:
        """Dispatch a message to all configured channels. Returns {channel: status} dict."""
        results = {}
        if config.slack_webhook_url:
            results["slack"] = self._send_slack(message)
        if config.discord_webhook_url:
            results["discord"] = self._send_discord(message)
        return results

    def _send_slack(self, message: str) -> str:
        """Post to Slack Incoming Webhook. Converts **bold** to *bold* (Slack syntax)."""
        try:
            # Convert Discord-style markdown bold (**) to Slack-style bold (*)
            text = message.replace("**", "*")
            r = self._client.post(config.slack_webhook_url, json={"text": text})
            return "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return f"error:{e}"

    def _send_discord(self, message: str) -> str:
        """Post to Discord Webhook. Discord natively supports **bold** markdown."""
        try:
            r = self._client.post(config.discord_webhook_url, json={"content": message})
            return "ok" if r.status_code in (200, 204) else f"error:{r.status_code}"
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return f"error:{e}"

    def close(self):
        self._client.close()
