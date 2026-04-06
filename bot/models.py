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

---------------------------------------------------------------------------
EDUCATIONAL OVERVIEW
---------------------------------------------------------------------------

WHAT IS A PREDICTION MARKET?
  A prediction market is a financial exchange where you bet on the outcome of
  a real-world event. Each contract pays out $1 (= 100 cents on Kalshi) if the
  event happens ("YES resolves") and $0 if it doesn't. The current price of a
  YES contract reflects the market's collective estimate of the probability.
  For example, if "Will BTC exceed $100K by March 14?" is trading at 40 cents,
  the market implies a ~40% chance of that happening.

  Kalshi is a regulated US prediction market exchange. It operates like a stock
  exchange — there are buyers and sellers, bids and asks, and an order book.

WHAT IS AN ORDER BOOK?
  An order book is a list of all pending buy and sell orders at various prices.
  For YES contracts:
    - "YES bid" = the highest price any buyer is currently willing to pay
    - "YES ask" = the lowest price any seller is currently willing to accept
  The gap between bid and ask is the "spread." You buy at the ask and sell at
  the bid. The midpoint between them is an approximation of fair value.

WHAT IS PYDANTIC IN THIS CONTEXT?
  Pydantic models (classes inheriting from BaseModel) are Python objects that
  automatically validate their fields. If kalshi's API returns a string where
  we expect an integer, Pydantic either converts it or raises a clear error.
  The @field_validator decorators add custom validation logic on top of that.

WHAT IS AN ENUM?
  An Enum (Enumeration) is a fixed set of named constants. Instead of using
  raw strings "yes" and "no" throughout the code (easy to typo), we use
  Side.YES and Side.NO. Python will raise an error if you try to create a
  Side with any other value. This makes the code both self-documenting and
  type-safe.

WHAT ARE CENTS (THE UNIT USED HERE)?
  Kalshi prices range from 1 to 99 cents, where 1 cent ≈ 1% implied probability.
  A YES contract at 60 cents means the market implies a ~60% chance of the event
  happening. At settlement: YES holders receive 100 cents (= $1) if the event
  occurs, or 0 cents if it doesn't. Using integers (whole cents) throughout
  avoids floating-point rounding errors in financial calculations.

WHAT IS OPEN INTEREST?
  Open interest is the total number of contracts currently held by all traders
  combined. It's different from volume: volume counts every trade executed,
  while open interest only counts contracts still open (not yet settled or sold).
  High open interest means many participants are committed to a position, which
  usually indicates more liquidity and market depth.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

# BaseModel: the base class for all Pydantic models (auto-validates fields)
# Field: lets you add constraints like ge=0 (greater-than-or-equal-to zero)
# field_validator: a decorator for custom per-field validation logic
from pydantic import BaseModel, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class Side(str, Enum):
    """The two sides of a binary prediction market contract.

    WHY str, Enum? By inheriting from both str and Enum, a Side value behaves
    like a plain string when needed (e.g., Side.YES == "yes" is True) while
    still being type-safe. This makes JSON serialisation and API comparisons
    work naturally without extra conversion steps.
    """
    YES = "yes"  # Betting the event will happen
    NO = "no"    # Betting the event will NOT happen


class OrderAction(str, Enum):
    """Whether to buy (open) or sell (close) a position.

    BUY: You are opening a new position by purchasing contracts.
    SELL: You are closing an existing position by selling contracts back.
    Note: on Kalshi you can also sell contracts you own to other buyers before
    settlement, essentially "cashing out" of your position early.
    """
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order execution type — market (immediate fill) or limit (price-specified).

    MARKET order: Fill immediately at whatever the current best price is.
      - Pros: Guaranteed execution (as long as there's a counterparty).
      - Cons: You have no control over the exact price; you could pay more
              than you expected if the market is illiquid.

    LIMIT order: Only fill at a specific price or better.
      - Pros: Price certainty — you never pay more than your limit price.
      - Cons: The order may not fill if no counterparty is willing to
              trade at your specified price.

    This bot defaults to LIMIT orders to control entry price and avoid
    overpaying in thin markets.
    """
    MARKET = "market"
    LIMIT = "limit"


# ── Market Data Models ────────────────────────────────────────────────────────

class Market(BaseModel):
    """A single prediction market contract on Kalshi.

    All prices are in cents (0-100), where 50c = 50% implied probability.
    YES + NO prices should sum to approximately 100c (minus the exchange's vig).

    WHAT IS "VIG" (VIGORISH)?
    The vig (also called "juice" or "the house edge") is the exchange's profit
    margin. On Kalshi, YES + NO prices typically sum to slightly less than 100c
    — for example, YES ask = 52c, NO ask = 51c. If you simultaneously bought
    both YES and NO, you'd pay 103c but receive exactly 100c at settlement —
    a guaranteed 3-cent loss. That 3 cents is the vig that flows to Kalshi.
    The bot must find edges larger than the vig to profit.
    """
    ticker: str                    # Unique market identifier (e.g., "KXBTC-24MAR14-T100000")
    event_ticker: str              # Parent event identifier
    title: str                     # Human-readable market question
    subtitle: str = ""             # Additional context

    # BID AND ASK PRICES:
    # yes_bid: highest price a buyer is willing to pay for YES contracts right now
    # yes_ask: lowest price a seller is willing to accept for YES contracts right now
    # To buy YES, you pay yes_ask. To sell YES, you receive yes_bid.
    # The same logic applies for NO contracts (no_bid / no_ask).
    yes_bid: int = 0               # Highest bid price for YES contracts (cents, 0-100)
    yes_ask: int = 0               # Lowest ask price for YES contracts (cents, 0-100)
    no_bid: int = 0                # Highest bid price for NO contracts (cents, 0-100)
    no_ask: int = 0                # Lowest ask price for NO contracts (cents, 0-100)

    volume: int = 0                # Total contracts traded
    open_interest: int = 0         # Contracts currently outstanding
    status: str = "open"           # "open", "closed", or "settled"
    close_time: str = ""           # ISO 8601 market expiration time
    # ISO 8601 is the international standard format for dates/times:
    # e.g., "2024-03-14T15:00:00Z" where Z = UTC timezone.
    result: str = ""               # "yes" or "no" (only set after settlement)
    category: str = ""             # Market category (politics, crypto, sports, etc.)
    last_price: int = 0            # Most recent trade price in cents
    prev_price: int = 0            # Previous day's close price in cents

    # STRIKE PRICES (for binary/range markets):
    # Binary markets pay YES if a value crosses a threshold at settlement.
    # floor_strike: the lower threshold. e.g., "BTC > $100K" has floor_strike=100000.
    #   YES resolves if settlement price >= floor_strike.
    # cap_strike: for bracket markets (e.g., "BTC between $90K and $100K"),
    #   this is the upper bound. YES resolves if floor_strike <= price <= cap_strike.
    floor_strike: float = 0.0      # Binary markets: lower bound / threshold price.
                                   # YES resolves if settlement price >= floor_strike.
    cap_strike: float = 0.0        # Range/bracket markets: upper bound of price bracket.
                                   # YES resolves if floor_strike <= settlement <= cap_strike.

    # ORDER BOOK DEPTH (how much can you actually buy/sell at the best price?):
    # yes_ask_size = how many contracts are available at yes_ask right now.
    # If you want to buy 100 contracts but yes_ask_size is only 10, you'll
    # only get 10 at the best price; the rest fill at worse prices.
    yes_ask_size: int = 0          # Contracts available at the best YES ask (live liquidity depth)
    yes_bid_size: int = 0          # Contracts available at the best YES bid

    # Fix 51: Validate yes_bid >= 0 and yes_ask >= 0
    # WHY VALIDATE? Kalshi's API occasionally returns None or negative values
    # for price fields (due to edge cases or market state). Clamping to 0
    # prevents downstream math (like spread calculation) from producing
    # nonsensical results. mode="before" means this runs before Pydantic's
    # own type conversion, so it can handle None safely.
    @field_validator("yes_bid", "yes_ask", "no_bid", "no_ask", mode="before")
    @classmethod
    def _clamp_prices_non_negative(cls, v: int) -> int:
        return max(0, int(v)) if v is not None else 0

    # Fix 52: Validate volume >= 0
    @field_validator("volume", mode="before")
    @classmethod
    def _clamp_volume_non_negative(cls, v: int) -> int:
        return max(0, int(v)) if v is not None else 0

    # Fix 53: Check required fields are present
    @property
    def is_valid(self) -> bool:
        """Check that required fields are present and minimally valid.

        A Market with no ticker, event_ticker, or title is useless — we can't
        reference it, display it, or trade it. This property lets the rest of
        the codebase quickly filter out malformed market data from the API.
        """
        return bool(self.ticker and self.event_ticker and self.title)

    @property
    def mid_price_yes(self) -> float:
        """Calculate the midpoint of the YES bid-ask spread.

        Falls back to ask, bid, or 50 if the spread is not available.
        Returns a value in cents, clamped to [1, 99].

        WHY USE THE MIDPOINT?
        The midpoint (average of bid and ask) is a common approximation of
        "fair value" as seen by the market. If YES bid = 38 and YES ask = 42,
        the midpoint is 40 — suggesting the market thinks the event has about
        a 40% chance of occurring. Using the midpoint rather than bid or ask
        alone avoids directional bias in our probability estimates.

        WHY CLAMP TO [1, 99]?
        A price of exactly 0 or 100 would mean the market has fully settled
        (impossible or certain). Active markets always trade between 1 and 99.
        Clamping prevents division-by-zero and log-of-zero errors in formulas
        that use the price as a probability input.
        """
        if self.yes_bid and self.yes_ask:
            raw = (self.yes_bid + self.yes_ask) / 2
        elif self.yes_ask or self.yes_bid:
            # Only one side of the market is quoted; use whatever is available
            raw = self.yes_ask or self.yes_bid
        else:
            # No price data at all — return 0.0 to signal "no data"
            return 0.0
        # Fix 54: Clamp to [1, 99] range
        return max(1.0, min(99.0, raw))

    @property
    def spread(self) -> int:
        """Bid-ask spread for YES contracts in cents. Lower = more liquid.

        WHY DOES SPREAD MATTER?
        The spread is an implicit transaction cost. Every time you enter a
        position you pay the ask; every time you exit you receive the bid.
        The round-trip cost is approximately the spread. A market with a
        1-cent spread is much cheaper to trade than one with a 10-cent spread.
        The bot uses spread as a liquidity filter — wide spreads erode edge
        and should be avoided.
        """
        # Fix 55: Return max(0, spread) to prevent negative
        # Negative spread would mean bid > ask, which is impossible in a real
        # order book but can happen with stale/bad data from the API.
        return max(0, self.yes_ask - self.yes_bid)


class Event(BaseModel):
    """A prediction market event containing one or more Market contracts.

    Example: "Bitcoin Price" event may contain markets like "BTC > $100K by March 14"
    and "BTC > $90K by March 14", each as separate Market objects.

    WHY SEPARATE EVENTS AND MARKETS?
    Kalshi organises its prediction markets hierarchically: an Event is the
    overarching topic (e.g., "Ethereum ETF Approval") while each Market inside
    it is a specific binary question (e.g., "Before January 2025?",
    "Before April 2025?"). This two-level structure lets you reason about a
    topic as a whole (the Event) or drill into specific contracts (the Markets).
    The bot fetches events from the API, then processes each market inside them.
    """
    event_ticker: str                                    # Unique event identifier
    title: str                                           # Event title/description
    category: str = ""                                   # Category (politics, crypto, etc.)
    # Field(default_factory=list) is Pydantic's way of saying "default value
    # is an empty list." Using `markets: list = []` directly would be a Python
    # anti-pattern (mutable default argument shared between instances).
    markets: list[Market] = Field(default_factory=list)  # Nested market contracts
    volume: int = 0                                      # Aggregate volume across all markets

    # Fix 59: total_markets property
    @property
    def total_markets(self) -> int:
        """Return the number of markets in this event.

        A convenience property so callers can write `event.total_markets`
        instead of `len(event.markets)`. Small readability improvement when
        constructing log messages or filtering events by market count.
        """
        return len(self.markets)


# ── Signal & Order Models ─────────────────────────────────────────────────────

class TradingSignal(BaseModel):
    """Output of the signal generation pipeline (RF ensemble or Claude AI).

    Represents a recommendation to trade a specific market. The edge field
    is the key metric: model's fair_probability minus the market's current price.

    HOW IS A SIGNAL GENERATED?
    The bot uses two independent methods to generate signals:
      1. Random Forest ensemble (rf_model.py): A machine learning model trained
         on historical market data (prices, volume, timing, crypto feed data).
         It outputs a probability estimate and a confidence score.
      2. Claude AI analysis (analyzer.py): The Claude language model reads the
         market title and context and provides a qualitative probability estimate
         with reasoning.
    Both methods produce a TradingSignal with the same fields, so the rest of
    the system can treat them identically.

    WHAT IS "FAIR PROBABILITY"?
    The model's estimate of the true probability of the event occurring — what
    the price *should* be if the market were perfectly efficient. If the model
    says fair_probability = 0.65 but the market prices YES at 0.50, the model
    believes the market is underpricing the event by 15 percentage points.

    WHAT IS "EDGE"?
    Edge = fair_probability - market_probability (when betting YES)
    A positive edge means the bet has a positive expected value (EV). Over many
    trades, positive-EV bets lead to profit. This is the core idea behind all
    systematic trading: find situations where your estimate of probability is
    more accurate than the market's, and bet accordingly.
    """
    ticker: str                                            # Market ticker to trade
    market_title: str                                      # Human-readable market name
    side: Side                                             # Recommended side (YES or NO)
    # Field(ge=0.0, le=1.0): Pydantic constraint — confidence must be
    # between 0.0 and 1.0 (ge = greater-than-or-equal, le = less-than-or-equal)
    confidence: float = Field(ge=0.0, le=1.0)              # Model confidence (0-1)
    fair_probability: float = Field(ge=0.0, le=1.0)        # Model's estimated true probability
    market_probability: float = Field(ge=0.0, le=1.0)      # Current market-implied probability
    # EDGE: positive edge = the market is underpricing the bet (good)
    #       negative edge = the market is overpricing the bet (would suggest the
    #                        opposite side has edge, not this one)
    edge: float = 0.0                                      # fair_prob - market_prob (positive = undervalued)
    reasoning: str = ""                                    # Human-readable explanation
    recommended_size_cents: int = 0                        # Suggested position size in cents
    category: str = ""                                     # Market category (for correlation-aware limits)
    # SIGNAL QUALITY: A composite score that blends edge, confidence, and
    # liquidity into a single ranking metric. Higher quality = higher priority.
    # Computed by the model or risk_manager, used to sort signals for execution.
    signal_quality: float = 0.0                            # Composite quality score (edge * confidence * liquidity)
    close_time: str = ""                                     # ISO timestamp of market close

    # Fix 56: Validate edge is in [-1, 1] range
    # Edge theoretically ranges from -1 (completely wrong) to +1 (perfect),
    # but values outside that range indicate a bug. Clamping prevents downstream
    # Kelly calculations from producing nonsensical bet sizes.
    @field_validator("edge", mode="before")
    @classmethod
    def _clamp_edge(cls, v: float) -> float:
        return max(-1.0, min(1.0, float(v)))

    # Fix 57: Validate confidence is in [0, 1] range (already enforced by Field(ge/le),
    # but add a before-validator for extra safety with out-of-range inputs)
    # "before" validators run before Pydantic's own type coercion, catching
    # extreme values like confidence=1.5 that would otherwise fail with a less
    # helpful error message.
    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    # Fix 58: Validate fair_probability is in [0, 1] range
    # Probabilities are always between 0 (impossible) and 1 (certain). A model
    # that returns 1.2 has a bug; clamping here provides a safety net.
    @field_validator("fair_probability", mode="before")
    @classmethod
    def _clamp_fair_probability(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class OrderRequest(BaseModel):
    """Order request to be placed on the Kalshi exchange via KalshiClient.place_order().

    This model represents a fully-formed instruction to the exchange. All the
    decision-making (sizing, side, price) is done before creating this object.
    KalshiClient takes an OrderRequest and translates it into the JSON payload
    that Kalshi's REST API expects.

    WHY SEPARATE THE SIGNAL FROM THE ORDER?
    A TradingSignal says "I think this is a good trade."
    An OrderRequest says "Execute exactly this trade at exactly this price."
    Separating them lets the RiskManager sit in between: it receives a signal,
    applies sizing and safety rules, and produces the final OrderRequest.
    """
    ticker: str                            # Market ticker to trade
    action: OrderAction = OrderAction.BUY  # Buy (open position) or sell (close position)
    side: Side = Side.YES                  # YES or NO side
    order_type: OrderType = OrderType.LIMIT  # Market or limit order
    # count: how many contracts to buy. Each contract costs price_cents and
    # pays out 100 cents if YES resolves. e.g., 10 contracts at 40c each
    # costs $4.00 and pays $10.00 if the event happens.
    count: int = Field(default=1, ge=1)     # Number of contracts to buy/sell
    # price_cents: the limit price in cents. ge=1 and le=99 enforce that the
    # price is always a valid market price (1-99 cents).
    price_cents: int = Field(default=50, ge=1, le=99)  # Limit price in cents (1-99)
    # time_in_force: controls order lifetime on the exchange.
    #   "good_till_canceled" — rests in order book until filled or manually cancelled (default)
    #   "immediate_or_cancel" — fill what's available RIGHT NOW, cancel the rest immediately
    #   "fill_or_kill"        — fill everything or cancel entirely (no partial fills)
    # IOC is the closest equivalent to a market order on Kalshi: instant fill-or-cancel.
    time_in_force: str = "good_till_canceled"


class OrderResponse(BaseModel):
    """Response from Kalshi after placing an order, with order status and fill info.

    WHAT DO THE STATUSES MEAN?
    - "resting": The order is in the order book, waiting for a counterparty.
    - "executed": The order has been fully filled. You own the contracts.
    - "pending": The order is being processed (transitional state).
    - "cancelled": The order was cancelled (either by you or the exchange).

    WHAT IS "remaining_count"?
    If you place an order for 10 contracts but only 3 are immediately available
    at your limit price, Kalshi fills 3 and leaves the remaining 7 "resting"
    in the order book. remaining_count = 7 in that scenario.
    """
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
    """An open position on Kalshi (one side of a market contract).

    A "position" is a collection of contracts you currently own. If you bought
    50 YES contracts on "BTC > $100K" at an average price of 42 cents, your
    position is: ticker="KXBTC-...", side="yes", quantity=50, avg_price_cents=42.

    UNREALISED P&L:
    Unrealised profit/loss is what you *would* make if you sold right now.
    If the YES price moved to 55 cents: unrealised P&L = (55 - 42) * 50 = 650
    cents = +$6.50. It becomes "realised" only when you actually sell or the
    market settles.
    """
    ticker: str                     # Market ticker
    event_ticker: str = ""          # Parent event ticker
    market_title: str = ""          # Human-readable market name (enriched by server)
    # Literal["yes", "no"] is a stricter type hint than str — Python's type
    # checker will flag any code that tries to assign a different string.
    side: Literal["yes", "no"] = "yes"  # "yes" or "no"
    quantity: int = 0               # Number of contracts held
    avg_price_cents: int = 0        # Average entry price in cents
    current_price_cents: int = 0    # Current market price (enriched by server)
    unrealized_pnl_cents: int = 0   # Unrealized profit/loss (computed by server)
    category: str = ""              # Market category (for correlation-aware limits)

    # Fix 60: Validate quantity >= 0
    # Negative quantity would be nonsensical for a long-only strategy and would
    # cause incorrect P&L calculations. Kalshi doesn't support short selling
    # in the traditional sense (you'd buy NO instead).
    @field_validator("quantity", mode="before")
    @classmethod
    def _clamp_quantity_non_negative(cls, v: int) -> int:
        return max(0, int(v)) if v is not None else 0


class PortfolioSummary(BaseModel):
    """Complete account portfolio state from Kalshi.

    Fetched from the Kalshi API at the start of each scan loop and after
    each trade. The bot's risk checks (daily loss limit, max open positions,
    drawdown cap) all read from this object.

    BALANCE vs PORTFOLIO VALUE:
    balance_cents: Cash in your account that hasn't been deployed yet.
    portfolio_value_cents: The current market value of all your open positions
      (what you'd receive if you sold everything right now at mid-price).
    Total account equity = balance + portfolio_value.

    DAILY P&L:
    daily_pnl_cents is the sum of all realised gains/losses from trades that
    settled or were closed today. It does NOT include unrealised gains on open
    positions. The daily loss limit check uses this field.
    """
    balance_cents: int = 0                                   # Available cash balance in cents
    portfolio_value_cents: int = 0                           # Total value of open positions
    positions: list[Position] = Field(default_factory=list)  # All open positions
    daily_pnl_cents: int = 0                                 # Today's realized P&L


# ── Cross-Platform Models ────────────────────────────────────────────────────

class DraftKingsMarket(BaseModel):
    """A DraftKings Predictions market used for cross-platform arbitrage detection.

    WHAT IS DRAFTKINGS PREDICTIONS?
    DraftKings is primarily a sports betting and daily fantasy platform, but it
    also offers a "Predictions" product similar to Kalshi. Users can bet on
    real-world events (politics, sports, entertainment) using binary YES/NO
    contracts. Because DraftKings and Kalshi are separate markets with
    independent user bases, the same event can be priced differently on each
    platform — which is the arbitrage opportunity this bot looks for.

    WHAT IS ARBITRAGE?
    Arbitrage is profiting from a price discrepancy for the same asset across
    different markets. If YES for "BTC > $100K" trades at 40% on Kalshi but
    50% on DraftKings, you could theoretically buy YES on Kalshi and sell YES
    on DraftKings, locking in a ~10% profit regardless of the outcome.
    In practice, this is complicated by the lack of a DraftKings trading API
    (manual execution required) and the risk that prices move before you execute
    both legs of the trade.

    NOTE: prices here are stored as decimals (0.0 to 1.0 = 0% to 100%),
    unlike Kalshi's integer cent representation (0 to 100). The arbitrage
    detector converts between these formats.
    """
    market_id: str = ""          # DraftKings market/draft group identifier
    title: str = ""              # Market title (matched against Kalshi by keyword overlap)
    category: str = ""           # Sport/category
    yes_price: float = 0.0       # YES price as decimal (0-1)
    no_price: float = 0.0        # NO price as decimal (0-1)
    volume: int = 0              # Trading volume
    close_time: str = ""         # Expiration time
    source: str = "draftkings"   # Platform identifier for multi-platform tracking
