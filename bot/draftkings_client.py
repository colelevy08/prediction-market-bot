"""
DraftKings Predictions market data scraper for cross-platform arbitrage.

DraftKings does not offer an official public trading API, so this client
scrapes publicly available market data for cross-platform analysis and
arbitrage detection. Actual order execution on DraftKings must be done
manually through their app.

This client attempts to fetch data from several DraftKings endpoints:
  - /sites/US-DK/sports/v1/sports: Lists available sport categories.
  - /lobby/getcontests: Lists available contests/markets.
  - /draftgroups/v1: Fetches draft group data used to populate DraftKingsMarket objects.

All endpoints are unofficial and may change without notice. The client uses a
browser-like User-Agent header to avoid being blocked by DraftKings' servers.
All API calls are wrapped in try/except to gracefully handle failures.

Connects to: DraftKings unofficial REST endpoints (api.draftkings.com,
sportsbook-nash.draftkings.com).
Used by: bot.server (arbitrage scan), bot.main (--arbitrage CLI flag),
bot.arbitrage (detect_arbitrage function).
"""

from __future__ import annotations

from typing import Any

import httpx

from bot.models import DraftKingsMarket


# Known DraftKings Predictions category endpoints (unofficial, may change)
DK_PREDICTIONS_BASE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1"
DK_API_BASE = "https://api.draftkings.com"


class DraftKingsClient:
    """Scrapes DraftKings Predictions market data for cross-platform analysis.

    All methods are best-effort and return empty lists on failure. DraftKings does
    not have a stable public API, so endpoints may break without notice.
    """

    def __init__(self):
        """Initialize with a browser-like HTTP client to avoid being blocked."""
        self._client = httpx.Client(
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "application/json",
            },
        )

    def get_available_sports(self) -> list[dict[str, Any]]:
        """List available sport categories from DraftKings. Returns empty list on failure."""
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
        """Fetch DraftKings Predictions markets from the draft groups endpoint.

        Since DraftKings Predictions doesn't have a stable public API, this attempts
        to scrape available data from the /draftgroups/v1 endpoint. Results may be
        incomplete or empty if the endpoint structure changes.

        Args:
            category: Category filter (currently unused — fetches all).

        Returns:
            List of DraftKingsMarket objects (may be empty on API failures).
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
        """Fetch odds/prices for a specific DraftKings event (best-effort). Returns {} on failure."""
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
