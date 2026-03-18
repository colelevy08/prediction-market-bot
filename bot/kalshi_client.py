"""
Kalshi Trade API v2 client with RSA-PSS cryptographic authentication.

This module handles all direct communication with the Kalshi prediction market
exchange. Kalshi requires every API request to be signed with an RSA or ECDSA
private key using PSS padding (RSA) or ECDSA(SHA256).

Key responsibilities:
  - Authentication: Signs each request with timestamp + method + path using the
    private key loaded from a PEM file or environment variable.
  - Market Data: Fetches events (with nested markets), individual markets, and
    orderbooks. Supports both paginated (get_events) and full-catalog
    (get_all_events, ~5000 events / ~41000 markets) retrieval.
  - Portfolio: Retrieves account balance, open positions, and portfolio summary.
  - Order Execution: Places limit/market orders, cancels orders, lists open orders.
  - Field Parsing: Handles Kalshi's dual field naming conventions (legacy cents-based
    fields vs. newer _dollars/_fp suffixed fields) via helper functions.

Helper functions (module-level):
  - _dollars_to_cents(): Converts Kalshi dollar strings/floats to integer cents (0-100).
  - _parse_fp(): Parses _fp (floating point) fields to integers.
  - _parse_market(): Builds a Market model from a raw Kalshi API dict, handling
    both old and new API field names gracefully.

Connects to:
  - Kalshi demo API (demo-api.kalshi.co) or production API (api.elections.kalshi.com)
    depending on KALSHI_USE_DEMO config flag.
  - bot.config for API credentials and base URL selection.
  - bot.models for Market, Event, Position, OrderRequest, OrderResponse, PortfolioSummary.

Used by: bot.server (all market/portfolio/order endpoints), bot.backtester
(HistoricalDataFetcher), bot.main (CLI scan and trade execution).
"""

from __future__ import annotations

import base64
import time
from typing import Any, Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, ec, utils as crypto_utils
from cryptography.exceptions import InvalidSignature

from bot.config import config
from bot.models import (
    Event, Market, OrderRequest, OrderResponse, Position, PortfolioSummary,
)


def _dollars_to_cents(val) -> int:
    """Convert a Kalshi dollars string/float/None to integer cents (0-100)."""
    if val is None:
        return 0
    try:
        return int(round(float(val) * 100))
    except (ValueError, TypeError):
        return 0


def _parse_fp(val) -> int:
    """Parse a _fp (floating point) field to int."""
    if val is None:
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _parse_market(m: dict) -> "Market":
    """Parse a market dict from the Kalshi API, handling both old and new field names."""
    # New API uses _dollars suffix; old API uses bare names
    yes_bid = _dollars_to_cents(m.get("yes_bid_dollars")) or m.get("yes_bid", 0) or 0
    yes_ask = _dollars_to_cents(m.get("yes_ask_dollars")) or m.get("yes_ask", 0) or 0
    no_bid = _dollars_to_cents(m.get("no_bid_dollars")) or m.get("no_bid", 0) or 0
    no_ask = _dollars_to_cents(m.get("no_ask_dollars")) or m.get("no_ask", 0) or 0
    volume = _parse_fp(m.get("volume_fp")) or m.get("volume", 0) or 0
    open_interest = _parse_fp(m.get("open_interest_fp")) or m.get("open_interest", 0) or 0
    last_price = _dollars_to_cents(m.get("last_price_dollars")) or 0
    prev_price = _dollars_to_cents(m.get("previous_price_dollars")) or 0

    # Ensure ints
    yes_bid = int(yes_bid) if isinstance(yes_bid, (int, float)) else 0
    yes_ask = int(yes_ask) if isinstance(yes_ask, (int, float)) else 0
    no_bid = int(no_bid) if isinstance(no_bid, (int, float)) else 0
    no_ask = int(no_ask) if isinstance(no_ask, (int, float)) else 0
    volume = int(volume) if isinstance(volume, (int, float)) else 0
    open_interest = int(open_interest) if isinstance(open_interest, (int, float)) else 0

    return Market(
        ticker=m.get("ticker", ""),
        event_ticker=m.get("event_ticker", ""),
        title=m.get("title", ""),
        subtitle=m.get("subtitle", ""),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=volume,
        open_interest=open_interest,
        status=m.get("status", "open"),
        close_time=m.get("close_time", ""),
        category=m.get("category", ""),
        result=m.get("result", ""),
        last_price=last_price,
        prev_price=prev_price,
    )


class KalshiClient:
    """HTTP client for the Kalshi Trade API v2 with cryptographic request signing.

    Handles authentication, market data retrieval, portfolio management, and
    order execution. Each API request is signed with the user's RSA or ECDSA
    private key using the Kalshi-specific signature scheme (timestamp + method + path).
    """

    def __init__(self):
        """Initialize the client with API credentials from config.

        Loads the private key (from env var or PEM file) for request signing.
        Uses a 30-second timeout for all HTTP requests to handle Kalshi's
        occasionally slow responses during high-traffic periods.
        """
        self.base_url = config.kalshi_base_url
        self.api_key_id = config.kalshi_api_key_id
        self._private_key = None
        self._client = httpx.Client(timeout=30)

        if config.kalshi_private_key:
            self._load_private_key(config.kalshi_private_key)

    def _load_private_key(self, pem_data: str):
        """Load and parse the PEM-encoded private key for request signing."""
        self._private_key = serialization.load_pem_private_key(
            pem_data.encode(), password=None
        )

    def _sign_request(self, method: str, full_path: str, timestamp_ms: int) -> str:
        """Generate a cryptographic signature for Kalshi API authentication.

        Kalshi's auth scheme requires signing: "{timestamp_ms}{METHOD}{path}" (no query params).
        Supports both RSA keys (PSS padding with SHA256) and ECDSA keys (SHA256).

        Args:
            method: HTTP method in uppercase (GET, POST, DELETE).
            full_path: Full API path including /trade-api/v2 prefix (query params stripped).
            timestamp_ms: Current time in milliseconds since epoch.

        Returns:
            Base64-encoded signature string for the KALSHI-ACCESS-SIGNATURE header.
        """
        # Strip query params before signing — Kalshi only signs the path portion
        path_no_query = full_path.split("?")[0]
        # Construct the message: timestamp + method + path (no body)
        message = f"{timestamp_ms}{method}{path_no_query}".encode("utf-8")

        if isinstance(self._private_key, ec.EllipticCurvePrivateKey):
            # ECDSA signing for EC private keys
            signature = self._private_key.sign(
                message, ec.ECDSA(hashes.SHA256())
            )
        else:
            # RSA signing — Kalshi requires PSS padding (not PKCS1v15)
            # Salt length = digest length (32 bytes for SHA256)
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, full_path: str) -> dict[str, str]:
        """Build the authentication headers required by every Kalshi API request.

        Returns a dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE,
        KALSHI-ACCESS-TIMESTAMP, and Content-Type headers.
        """
        ts = int(time.time() * 1000)  # Current time in milliseconds
        sig = self._sign_request(method.upper(), full_path, ts)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "Content-Type": "application/json",
        }

    def _request(
        self, method: str, path: str, params: dict | None = None, json: dict | None = None
    ) -> dict[str, Any]:
        """Execute an authenticated HTTP request to the Kalshi API.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: API path relative to /trade-api/v2 (e.g., "/events", "/portfolio/orders").
            params: Optional query parameters.
            json: Optional JSON request body.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses (propagated to callers).
        """
        url = f"{self.base_url}{path}"
        # Sign with the full path including /trade-api/v2 prefix
        full_path = f"/trade-api/v2{path}"
        headers = self._auth_headers(method.upper(), full_path)
        resp = self._client.request(method, url, headers=headers, params=params, json=json)
        resp.raise_for_status()
        return resp.json()

    # ── Market Data ──────────────────────────────────────────────

    def get_events(
        self, limit: int = 50, status: str = "open", with_nested_markets: bool = True
    ) -> list[Event]:
        """Fetch top events sorted by volume."""
        data = self._request("GET", "/events", params={
            "limit": min(limit, 200),
            "status": status,
            "with_nested_markets": str(with_nested_markets).lower(),
        })
        events = []
        for ev in data.get("events", []):
            markets = []
            for m in ev.get("markets", []):
                markets.append(_parse_market(m))
            events.append(Event(
                event_ticker=ev.get("event_ticker", ""),
                title=ev.get("title", ""),
                category=ev.get("category", ""),
                markets=markets,
                volume=sum(m.volume for m in markets),
            ))
        # Sort by volume descending
        events.sort(key=lambda e: e.volume, reverse=True)
        return events[:limit]

    def get_all_events(self, status: str = "open", with_nested_markets: bool = True) -> list[Event]:
        """Fetch ALL events from Kalshi using cursor-based pagination.

        This paginates through the entire event catalog (~5000 events, ~41000 markets)
        to ensure we scan every available market for edges. Each page returns up to
        200 events. Pagination continues until no cursor is returned or the page is empty.
        Results are sorted by total volume descending.

        This is the method used by the auto-scan background job. For lighter on-demand
        scans, use get_events() with a limit instead.
        """
        all_events = []
        cursor = None
        while True:
            params = {
                "limit": 200,
                "status": status,
                "with_nested_markets": str(with_nested_markets).lower(),
            }
            if cursor:
                params["cursor"] = cursor
            data = self._request("GET", "/events", params=params)
            for ev in data.get("events", []):
                markets = []
                for m in ev.get("markets", []):
                    markets.append(_parse_market(m))
                all_events.append(Event(
                    event_ticker=ev.get("event_ticker", ""),
                    title=ev.get("title", ""),
                    category=ev.get("category", ""),
                    markets=markets,
                    volume=sum(m.volume for m in markets),
                ))
            cursor = data.get("cursor", None)
            if not cursor or not data.get("events"):
                break
        # Sort by volume descending
        all_events.sort(key=lambda e: e.volume, reverse=True)
        return all_events

    def get_market(self, ticker: str) -> Market | None:
        """Fetch a single market by ticker."""
        try:
            data = self._request("GET", f"/markets/{ticker}")
            m = data.get("market", {})
            return _parse_market(m)
        except httpx.HTTPStatusError:
            return None

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Fetch orderbook for a market."""
        return self._request("GET", f"/markets/{ticker}/orderbook")

    # ── Portfolio ────────────────────────────────────────────────

    def get_balance(self) -> int:
        """Get account balance in cents."""
        data = self._request("GET", "/portfolio/balance")
        return data.get("balance", 0)

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        data = self._request("GET", "/portfolio/positions", params={"limit": 200})
        positions = []
        for p in data.get("market_positions", []):
            positions.append(Position(
                ticker=p.get("ticker", ""),
                event_ticker=p.get("event_ticker", ""),
                side="yes" if p.get("yes_amount", 0) > 0 else "no",
                quantity=max(p.get("yes_amount", 0), p.get("no_amount", 0)),
                avg_price_cents=p.get("average_price", 0),
            ))
        return [p for p in positions if p.quantity > 0]

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get full portfolio state."""
        balance = self.get_balance()
        positions = self.get_positions()
        return PortfolioSummary(
            balance_cents=balance,
            positions=positions,
        )

    # ── Orders ───────────────────────────────────────────────────

    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place an order on Kalshi.

        For limit orders, the API requires yes_price in cents. If the order side
        is NO, we convert: yes_price = 100 - no_price (since YES + NO = 100c).
        Market orders do not require a price field.

        Args:
            order: OrderRequest with ticker, side, type, count, and price.

        Returns:
            OrderResponse with order_id, status, and fill information.
        """
        body = {
            "ticker": order.ticker,
            "action": order.action.value,
            "side": order.side.value,
            "type": order.order_type.value,
            "count": order.count,
        }
        if order.order_type == "limit":
            # Kalshi API always takes yes_price; convert NO side to equivalent YES price
            body["yes_price"] = order.price_cents if order.side == "yes" else (100 - order.price_cents)

        data = self._request("POST", "/portfolio/orders", json=body)
        o = data.get("order", {})
        return OrderResponse(
            order_id=o.get("order_id", ""),
            ticker=o.get("ticker", ""),
            status=o.get("status", ""),
            side=o.get("side", ""),
            action=o.get("action", ""),
            price_cents=o.get("yes_price", 0),
            count=o.get("count", 0),
            remaining_count=o.get("remaining_count", 0),
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self._request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except httpx.HTTPStatusError:
            return False

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        data = self._request("GET", "/portfolio/orders", params={"status": "resting"})
        return data.get("orders", [])

    def close(self):
        """Close the underlying HTTP client connection pool."""
        self._client.close()
