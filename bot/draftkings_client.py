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

---------------------------------------------------------------------------
EDUCATIONAL OVERVIEW
---------------------------------------------------------------------------

WHAT IS WEB SCRAPING?
  Web scraping is the automated extraction of data from websites or web
  services that were not designed to be consumed programmatically. Most
  websites have a public-facing HTML interface for humans and may also expose
  internal API endpoints that their own JavaScript code calls. Developers
  sometimes use those same internal endpoints to collect data, even though
  they're not publicly documented. This is a legal grey area and carries risks:
  the endpoints can change without notice, the site may block your IP, or the
  terms of service may prohibit it. Always review a platform's terms before
  scraping.

WHAT IS httpx?
  httpx is a Python HTTP library — it lets your program make requests to web
  servers, just like a browser does. It's similar to the popular `requests`
  library but with modern features like async support. Here, we use the
  synchronous (blocking) client because the rest of the bot is synchronous.

WHAT IS A USER-AGENT HEADER?
  When your browser loads a web page, it sends a User-Agent header that
  identifies itself: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)..."
  Many servers use this to filter out bots. By mimicking a real browser's
  User-Agent, the client is less likely to be blocked. This is a common
  (though not foolproof) scraping technique.

WHAT IS "BEST-EFFORT"?
  Best-effort means the function tries to return useful data but makes no
  guarantees. If the endpoint is down, the response format changed, or DK
  blocked the request, the function returns an empty list instead of crashing
  the bot. This is appropriate here because DK market data is supplementary
  (used for optional arbitrage detection), not core to the trading strategy.

WHAT IS raise_for_status()?
  After an HTTP request, the server returns a status code:
    200 = OK (success)
    404 = Not Found
    429 = Too Many Requests (rate limited)
    500 = Internal Server Error
  raise_for_status() is an httpx method that automatically raises a Python
  exception if the status code indicates an error (anything >= 400).
  Wrapped in try/except, this means any HTTP error silently returns empty data.

WHY IS DraftKings DATA INCOMPLETE?
  DraftKings' internal endpoints change frequently, are not versioned for
  external use, and don't expose price data (YES/NO odds) in the same format
  as Kalshi. The client can retrieve market metadata (titles, categories) but
  typically cannot get live prices from these endpoints. A fully functional
  DK scraper would require reverse-engineering their mobile app traffic or
  using a browser automation tool like Playwright — significantly more complex.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

# httpx: modern HTTP client library for Python
import httpx

from bot.models import DraftKingsMarket


# Known DraftKings Predictions category endpoints (unofficial, may change)
# These URLs were reverse-engineered from DraftKings' own website JavaScript.
# They are not documented and may stop working at any time.
DK_PREDICTIONS_BASE = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusnj/v1"
DK_API_BASE = "https://api.draftkings.com"


class DraftKingsClient:
    """Scrapes DraftKings Predictions market data for cross-platform analysis.

    All methods are best-effort and return empty lists on failure. DraftKings does
    not have a stable public API, so endpoints may break without notice.

    DESIGN PRINCIPLE — "FAIL SILENTLY FOR OPTIONAL DATA":
    The bot's core trading logic (Kalshi RF model) works without DraftKings data.
    DK data is only used for the optional arbitrage feature. If DK is unavailable,
    the bot should continue trading normally rather than crashing. All methods here
    wrap their HTTP calls in try/except and return empty lists on any failure.
    This is the "graceful degradation" design pattern.

    LIFECYCLE:
    Create one instance, use it for all DK calls during a session, then call
    close() to release the underlying HTTP connection pool. Using a shared client
    (rather than creating a new one for every request) is more efficient because
    the TCP connection to DK's servers can be reused across requests.
    """

    def __init__(self):
        """Initialize with a browser-like HTTP client to avoid being blocked.

        timeout=30: If DraftKings doesn't respond within 30 seconds, give up.
        The User-Agent and Accept headers mimic a real browser so DK's servers
        are less likely to classify the request as automated and block it.
        """
        self._client = httpx.Client(
            timeout=30,
            headers={
                # Mimic a macOS Chrome browser — one of the most common user agents
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                # Tell the server we want JSON, not HTML
                "Accept": "application/json",
            },
        )

    def get_available_sports(self) -> list[dict[str, Any]]:
        """List available sport categories from DraftKings. Returns empty list on failure.

        Calls the /sports endpoint to get a list of sport types available on DK.
        Useful for discovering what categories of markets exist before fetching
        contests/markets within those categories.

        Returns:
            A list of sport dictionaries from the DK API (structure may vary).
            Returns [] if the request fails or DK changes the endpoint.
        """
        try:
            resp = self._client.get(f"{DK_API_BASE}/sites/US-DK/sports/v1/sports", params={"format": "json"})
            # Raise an exception if the status code is 4xx or 5xx
            resp.raise_for_status()
            # .json() parses the response body as JSON and returns a Python dict/list.
            # .get("sports", []) safely extracts the "sports" key, defaulting to []
            # if the key doesn't exist (prevents KeyError on unexpected response format).
            return resp.json().get("sports", [])
        except Exception:
            # Catch ALL exceptions (network errors, JSON parse errors, HTTP errors)
            # and return empty list. This client is optional; never crash the bot.
            return []

    def get_contests(self, sport: str | None = None) -> list[dict[str, Any]]:
        """Get available contests/markets, optionally filtered by sport.

        A "contest" in DraftKings terminology is similar to an event on Kalshi
        — it groups multiple markets around a single topic or game. This endpoint
        is primarily designed for DraftKings' daily fantasy product but also
        surfaces prediction markets.

        Args:
            sport: Optional sport filter string (e.g., "NFL", "NBA"). If None,
                   returns contests across all sports.

        Returns:
            List of contest dictionaries. Returns [] on any failure.
        """
        try:
            params = {}
            if sport:
                # Only add the sport parameter if a filter was requested
                params["sport"] = sport
            resp = self._client.get("https://www.draftkings.com/lobby/getcontests", params=params)
            resp.raise_for_status()
            data = resp.json()
            # DraftKings returns contests under the "Contests" key (capital C)
            return data.get("Contests", [])
        except Exception:
            return []

    def get_prediction_markets(self, category: str = "all") -> list[DraftKingsMarket]:
        """Fetch DraftKings Predictions markets from the draft groups endpoint.

        Since DraftKings Predictions doesn't have a stable public API, this attempts
        to scrape available data from the /draftgroups/v1 endpoint. Results may be
        incomplete or empty if the endpoint structure changes.

        WHAT IS A "DRAFT GROUP"?
        DraftKings organises contests into "draft groups" — a contest type + sport
        combination that defines a pool of players or markets. For prediction markets,
        draft groups roughly correspond to market categories. The /draftgroups/v1
        endpoint returns metadata about these groups, but NOT live prices — which is
        why DraftKingsMarket objects created here have yes_price = 0.0 by default.

        Args:
            category: Category filter (currently unused — fetches all).
                      Included as a parameter for future extensibility.

        Returns:
            List of DraftKingsMarket objects (may be empty on API failures).
            NOTE: yes_price and no_price will be 0.0 because this endpoint
            does not expose odds/prices — only metadata.
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
                # Iterate over each "draft group" in the response
                for group in data.get("draftGroups", []):
                    # Build a DraftKingsMarket from the metadata fields.
                    # str() around draftGroupId handles both int and string IDs.
                    # Nested .get() with a fallback dict handles missing keys safely:
                    #   group.get("contestType", {}) returns {} if "contestType" is absent,
                    #   then .get("name", "Unknown") returns "Unknown" if "name" is absent.
                    markets.append(DraftKingsMarket(
                        market_id=str(group.get("draftGroupId", "")),
                        title=group.get("contestType", {}).get("name", "Unknown"),
                        category=group.get("sport", ""),
                        source="draftkings",
                        # Note: yes_price and no_price are left at default 0.0
                        # because this endpoint doesn't provide odds data.
                    ))
        except Exception:
            # If the endpoint fails entirely, pass through to return whatever
            # markets were collected so far (possibly an empty list).
            pass

        return markets

    def get_event_odds(self, event_id: str) -> dict[str, Any]:
        """Fetch odds/prices for a specific DraftKings event (best-effort). Returns {} on failure.

        Attempts to retrieve odds for a specific contest using its event ID.
        This is useful if you already know the DraftKings event ID (e.g., from
        get_contests()) and want to look up its current prices.

        Args:
            event_id: The DraftKings contest/event identifier (string or numeric).

        Returns:
            A dictionary with event odds data from DK's API, or {} on failure.
            Structure varies by event type and DK API version.
        """
        try:
            resp = self._client.get(
                f"{DK_API_BASE}/contests/v1/contests/{event_id}",
                params={"format": "json"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            # Return an empty dict rather than raising — caller can check
            # `if not result:` to detect failure without try/except on their end.
            return {}

    def close(self):
        """Close the underlying HTTP client and release connection pool resources.

        Always call this when you're done using the client, ideally in a
        try/finally block. Failing to close the client may leave open TCP
        connections or file descriptors (on some systems). This is the same
        pattern used for file handles: open, use, close.
        """
        self._client.close()
