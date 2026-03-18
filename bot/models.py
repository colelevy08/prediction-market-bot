"""Data models for markets, orders, and trading decisions."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Side(str, Enum):
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class Market(BaseModel):
    """A single prediction market contract."""
    ticker: str
    event_ticker: str
    title: str
    subtitle: str = ""
    yes_bid: int = 0          # in cents (0-100)
    yes_ask: int = 0
    no_bid: int = 0
    no_ask: int = 0
    volume: int = 0
    open_interest: int = 0
    status: str = "open"
    close_time: str = ""
    result: str = ""
    category: str = ""
    last_price: int = 0       # last trade price in cents
    prev_price: int = 0       # previous day price in cents

    @property
    def mid_price_yes(self) -> float:
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2
        return self.yes_ask or self.yes_bid or 50

    @property
    def spread(self) -> int:
        return self.yes_ask - self.yes_bid


class Event(BaseModel):
    """A prediction market event containing one or more markets."""
    event_ticker: str
    title: str
    category: str = ""
    markets: list[Market] = Field(default_factory=list)
    volume: int = 0


class TradingSignal(BaseModel):
    """AI-generated trading signal for a market."""
    ticker: str
    market_title: str
    side: Side
    confidence: float = Field(ge=0.0, le=1.0)
    fair_probability: float = Field(ge=0.0, le=1.0)
    market_probability: float = Field(ge=0.0, le=1.0)
    edge: float = 0.0          # fair_prob - market_prob (for YES side)
    reasoning: str = ""
    recommended_size_cents: int = 0


class OrderRequest(BaseModel):
    """Order to place on Kalshi."""
    ticker: str
    action: OrderAction = OrderAction.BUY
    side: Side = Side.YES
    order_type: OrderType = OrderType.LIMIT
    count: int = 1             # number of contracts
    price_cents: int = 50      # limit price in cents (1-99)


class OrderResponse(BaseModel):
    """Response from Kalshi after placing an order."""
    order_id: str = ""
    ticker: str = ""
    status: str = ""
    side: str = ""
    action: str = ""
    price_cents: int = 0
    count: int = 0
    remaining_count: int = 0


class Position(BaseModel):
    """An open position on Kalshi."""
    ticker: str
    event_ticker: str = ""
    market_title: str = ""
    side: str = ""
    quantity: int = 0
    avg_price_cents: int = 0
    current_price_cents: int = 0
    unrealized_pnl_cents: int = 0


class PortfolioSummary(BaseModel):
    """Account portfolio state."""
    balance_cents: int = 0
    portfolio_value_cents: int = 0
    positions: list[Position] = Field(default_factory=list)
    daily_pnl_cents: int = 0


class DraftKingsMarket(BaseModel):
    """A DraftKings Predictions market."""
    market_id: str = ""
    title: str = ""
    category: str = ""
    yes_price: float = 0.0
    no_price: float = 0.0
    volume: int = 0
    close_time: str = ""
    source: str = "draftkings"
