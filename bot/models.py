"""
Pydantic data models shared across the entire prediction market bot.

Defines the core data structures used for inter-module communication:

  - Market:           A single prediction market contract on Kalshi with bid/ask/volume/OI.
                      Prices are in cents (0-100). Computed properties: mid_price_yes, spread.
  - Event:            A Kalshi event containing one or more Market contracts (e.g.,
                      "Will BTC exceed $100K?" with YES/NO markets).
  - TradingSignal:    Output of the signal generation pipeline (RF model or Claude AI).
                      Contains fair probability, market probability, edge, confidence, and
                      recommended position size.
  - OrderRequest:     Input to KalshiClient.place_order() specifying ticker, side, type, etc.
  - OrderResponse:    Kalshi's response after an order is placed (order ID, status, fills).
  - Position:         An open position on Kalshi (ticker, side, quantity, avg price).
  - PortfolioSummary: Full account state (balance + all positions + daily P&L).
  - DraftKingsMarket: A DraftKings Predictions market for cross-platform arbitrage detection.

Enums:
  - Side:        YES or NO (the two sides of a binary prediction market contract).
  - OrderAction: BUY or SELL.
  - OrderType:   MARKET or LIMIT.

Used by: Every module in the bot package. These models form the shared vocabulary
for market data, signals, orders, and portfolio state.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class Side(str, Enum):
    """The two sides of a binary prediction market contract."""
    YES = "yes"  # Betting the event will happen
    NO = "no"    # Betting the event will NOT happen


class OrderAction(str, Enum):
    """Whether to buy (open) or sell (close) a position."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order execution type — market (immediate fill) or limit (price-specified)."""
    MARKET = "market"
    LIMIT = "limit"


# ── Market Data Models ────────────────────────────────────────────────────────

class Market(BaseModel):
    """A single prediction market contract on Kalshi.

    All prices are in cents (0-100), where 50c = 50% implied probability.
    YES + NO prices should sum to approximately 100c (minus the exchange's vig).
    """
    ticker: str                    # Unique market identifier (e.g., "KXBTC-24MAR14-T100000")
    event_ticker: str              # Parent event identifier
    title: str                     # Human-readable market question
    subtitle: str = ""             # Additional context
    yes_bid: int = 0               # Highest bid price for YES contracts (cents, 0-100)
    yes_ask: int = 0               # Lowest ask price for YES contracts (cents, 0-100)
    no_bid: int = 0                # Highest bid price for NO contracts (cents, 0-100)
    no_ask: int = 0                # Lowest ask price for NO contracts (cents, 0-100)
    volume: int = 0                # Total contracts traded
    open_interest: int = 0         # Contracts currently outstanding
    status: str = "open"           # "open", "closed", or "settled"
    close_time: str = ""           # ISO 8601 market expiration time
    result: str = ""               # "yes" or "no" (only set after settlement)
    category: str = ""             # Market category (politics, crypto, sports, etc.)
    last_price: int = 0            # Most recent trade price in cents
    prev_price: int = 0            # Previous day's close price in cents

    @property
    def mid_price_yes(self) -> float:
        """Calculate the midpoint of the YES bid-ask spread.

        Falls back to ask, bid, or 50 if the spread is not available.
        Returns a value in cents (0-100).
        """
        if self.yes_bid and self.yes_ask:
            return (self.yes_bid + self.yes_ask) / 2
        return self.yes_ask or self.yes_bid or 50

    @property
    def spread(self) -> int:
        """Bid-ask spread for YES contracts in cents. Lower = more liquid."""
        return self.yes_ask - self.yes_bid


class Event(BaseModel):
    """A prediction market event containing one or more Market contracts.

    Example: "Bitcoin Price" event may contain markets like "BTC > $100K by March 14"
    and "BTC > $90K by March 14", each as separate Market objects.
    """
    event_ticker: str                                    # Unique event identifier
    title: str                                           # Event title/description
    category: str = ""                                   # Category (politics, crypto, etc.)
    markets: list[Market] = Field(default_factory=list)  # Nested market contracts
    volume: int = 0                                      # Aggregate volume across all markets


# ── Signal & Order Models ─────────────────────────────────────────────────────

class TradingSignal(BaseModel):
    """Output of the signal generation pipeline (RF ensemble or Claude AI).

    Represents a recommendation to trade a specific market. The edge field
    is the key metric: model's fair_probability minus the market's current price.
    """
    ticker: str                                            # Market ticker to trade
    market_title: str                                      # Human-readable market name
    side: Side                                             # Recommended side (YES or NO)
    confidence: float = Field(ge=0.0, le=1.0)              # Model confidence (0-1)
    fair_probability: float = Field(ge=0.0, le=1.0)        # Model's estimated true probability
    market_probability: float = Field(ge=0.0, le=1.0)      # Current market-implied probability
    edge: float = 0.0                                      # fair_prob - market_prob (positive = undervalued)
    reasoning: str = ""                                    # Human-readable explanation
    recommended_size_cents: int = 0                        # Suggested position size in cents


class OrderRequest(BaseModel):
    """Order request to be placed on the Kalshi exchange via KalshiClient.place_order()."""
    ticker: str                            # Market ticker to trade
    action: OrderAction = OrderAction.BUY  # Buy (open position) or sell (close position)
    side: Side = Side.YES                  # YES or NO side
    order_type: OrderType = OrderType.LIMIT  # Market or limit order
    count: int = 1                         # Number of contracts to buy/sell
    price_cents: int = 50                  # Limit price in cents (1-99); ignored for market orders


class OrderResponse(BaseModel):
    """Response from Kalshi after placing an order, with order status and fill info."""
    order_id: str = ""              # Unique order identifier from Kalshi
    ticker: str = ""                # Market ticker
    status: str = ""                # "resting", "executed", "pending", etc.
    side: str = ""                  # "yes" or "no"
    action: str = ""                # "buy" or "sell"
    price_cents: int = 0            # Limit price in cents
    count: int = 0                  # Total contracts requested
    remaining_count: int = 0        # Contracts not yet filled


# ── Portfolio Models ──────────────────────────────────────────────────────────

class Position(BaseModel):
    """An open position on Kalshi (one side of a market contract)."""
    ticker: str                     # Market ticker
    event_ticker: str = ""          # Parent event ticker
    market_title: str = ""          # Human-readable market name (enriched by server)
    side: str = ""                  # "yes" or "no"
    quantity: int = 0               # Number of contracts held
    avg_price_cents: int = 0        # Average entry price in cents
    current_price_cents: int = 0    # Current market price (enriched by server)
    unrealized_pnl_cents: int = 0   # Unrealized profit/loss (computed by server)


class PortfolioSummary(BaseModel):
    """Complete account portfolio state from Kalshi."""
    balance_cents: int = 0                                   # Available cash balance in cents
    portfolio_value_cents: int = 0                           # Total value of open positions
    positions: list[Position] = Field(default_factory=list)  # All open positions
    daily_pnl_cents: int = 0                                 # Today's realized P&L


# ── Cross-Platform Models ────────────────────────────────────────────────────

class DraftKingsMarket(BaseModel):
    """A DraftKings Predictions market used for cross-platform arbitrage detection."""
    market_id: str = ""          # DraftKings market/draft group identifier
    title: str = ""              # Market title (matched against Kalshi by keyword overlap)
    category: str = ""           # Sport/category
    yes_price: float = 0.0       # YES price as decimal (0-1)
    no_price: float = 0.0        # NO price as decimal (0-1)
    volume: int = 0              # Trading volume
    close_time: str = ""         # Expiration time
    source: str = "draftkings"   # Platform identifier for multi-platform tracking
