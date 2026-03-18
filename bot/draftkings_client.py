"""DraftKings Predictions market data scraper.

DraftKings does not offer an official public trading API, so this client
scrapes publicly available market data for cross-platform analysis and
arbitrage detection. Actual order execution on DraftKings must be done
manually through their app.
"""

from __future__ import annotations

from typing import Any

import httpx

from bot.models import DraftKingsMarket


# Known DraftKings Predictions category endpoints (unofficial, may change)
DK_PREDICTIONS_BASE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1"
DK_API_BASE = "https://api.draftkings.com"


class DraftKingsClient:
    """Scrapes DraftKings Predictions market data for analysis."""

    def __init__(self):
        self._client = httpx.Client(
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "application/json",
            },
        )

    def get_available_sports(self) -> list[dict[str, Any]]:
        """List available sport categories."""
        try:
            resp = self._client.get(f"{DK_API_BASE}/sites/US-DK/sports/v1/sports", params={"format": "json"})
            resp.raise_for_status()
            return resp.json().get("sports", [])
        except Exception:
            return []

    def get_contests(self, sport: str | None = None) -> list[dict[str, Any]]:
        """Get available contests/markets, optionally filtered by sport."""
        try:
            params = {}
            if sport:
                params["sport"] = sport
            resp = self._client.get("https://www.draftkings.com/lobby/getcontests", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("Contests", [])
        except Exception:
            return []

    def get_prediction_markets(self, category: str = "all") -> list[DraftKingsMarket]:
        """
        Fetch DraftKings Predictions markets.

        Since DraftKings Predictions doesn't have a stable public API,
        this attempts to scrape available data. Results may be incomplete.
        """
        markets = []

        # Try fetching from the sportsbook events endpoint
        try:
            resp = self._client.get(
                f"{DK_API_BASE}/draftgroups/v1",
                params={"format": "json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for group in data.get("draftGroups", []):
                    markets.append(DraftKingsMarket(
                        market_id=str(group.get("draftGroupId", "")),
                        title=group.get("contestType", {}).get("name", "Unknown"),
                        category=group.get("sport", ""),
                        source="draftkings",
                    ))
        except Exception:
            pass

        return markets

    def get_event_odds(self, event_id: str) -> dict[str, Any]:
        """Fetch odds/prices for a specific event (best-effort)."""
        try:
            resp = self._client.get(
                f"{DK_API_BASE}/contests/v1/contests/{event_id}",
                params={"format": "json"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return {}

    def close(self):
        self._client.close()
