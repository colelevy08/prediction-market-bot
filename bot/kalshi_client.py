"""Kalshi API client with RSA signature authentication."""

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


class KalshiClient:
    """Client for the Kalshi Trade API v2."""

    def __init__(self):
        self.base_url = config.kalshi_base_url
        self.api_key_id = config.kalshi_api_key_id
        self._private_key = None
        self._client = httpx.Client(timeout=30)

        if config.kalshi_private_key:
            self._load_private_key(config.kalshi_private_key)

    def _load_private_key(self, pem_data: str):
        self._private_key = serialization.load_pem_private_key(
            pem_data.encode(), password=None
        )

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Generate RSA signature for Kalshi API authentication."""
        message = f"{timestamp_ms}{method}{path}".encode()

        if isinstance(self._private_key, ec.EllipticCurvePrivateKey):
            signature = self._private_key.sign(
                message, ec.ECDSA(hashes.SHA256())
            )
        else:
            # RSA key
            signature = self._private_key.sign(
                message,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        ts = int(time.time() * 1000)
        sig = self._sign_request(method.upper(), path, ts)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "Content-Type": "application/json",
        }

    def _request(
        self, method: str, path: str, params: dict | None = None, json: dict | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._auth_headers(method.upper(), path)
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
                markets.append(Market(
                    ticker=m.get("ticker", ""),
                    event_ticker=m.get("event_ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 0),
                    no_bid=m.get("no_bid", 0),
                    no_ask=m.get("no_ask", 0),
                    volume=m.get("volume", 0),
                    open_interest=m.get("open_interest", 0),
                    status=m.get("status", "open"),
                    close_time=m.get("close_time", ""),
                    category=m.get("category", ""),
                ))
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

    def get_market(self, ticker: str) -> Market | None:
        """Fetch a single market by ticker."""
        try:
            data = self._request("GET", f"/markets/{ticker}")
            m = data.get("market", {})
            return Market(
                ticker=m.get("ticker", ""),
                event_ticker=m.get("event_ticker", ""),
                title=m.get("title", ""),
                subtitle=m.get("subtitle", ""),
                yes_bid=m.get("yes_bid", 0),
                yes_ask=m.get("yes_ask", 0),
                no_bid=m.get("no_bid", 0),
                no_ask=m.get("no_ask", 0),
                volume=m.get("volume", 0),
                open_interest=m.get("open_interest", 0),
                status=m.get("status", "open"),
                close_time=m.get("close_time", ""),
                category=m.get("category", ""),
            )
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
        """Place an order on Kalshi."""
        body = {
            "ticker": order.ticker,
            "action": order.action.value,
            "side": order.side.value,
            "type": order.order_type.value,
            "count": order.count,
        }
        if order.order_type == "limit":
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
        self._client.close()
