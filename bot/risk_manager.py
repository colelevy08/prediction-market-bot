"""
Risk management module enforcing trading limits and position sizing.

This module sits between signal generation and order execution as a safety gate.
Every trade signal must pass through RiskManager.check_signal() before an order
is placed. The risk checks include:

  1. Daily loss limit:    Stops trading if cumulative daily P&L hits the configured
                          max_daily_loss_cents threshold (default $100).
  2. Max open positions:  Prevents over-concentration by capping simultaneous positions
                          (default 10).
  3. Duplicate positions: Rejects signals for markets where a position already exists.
  4. Minimum edge:        Filters out low-edge signals below min_edge_threshold (default 8%).
  5. Balance check:       Ensures sufficient account balance to cover the trade cost.
  6. Bet size cap:        Enforces max_bet_amount_cents per trade (default $25).

Position sizing uses the Kelly Criterion:
  - Full Kelly: f* = (b*p - q) / b, where b = payout odds, p = model prob, q = 1-p.
  - Applied as fractional Kelly (default half-Kelly via kelly_fraction=0.5) to reduce
    variance at the cost of slightly lower expected growth.
  - The result is capped at max_bet_amount_cents.

Also provides build_order() to convert a validated TradingSignal into an OrderRequest
ready for KalshiClient.place_order().

Connects to: bot.config (all threshold/limit values), bot.models (signal/order types).
Used by: bot.server (signal validation, order building), bot.main (CLI trade execution).
"""

from __future__ import annotations

from bot.config import config
from bot.models import OrderRequest, PortfolioSummary, TradingSignal


class RiskManager:
    """Pre-trade risk gate that validates signals and sizes positions.

    Every trading signal must pass through check_signal() before an order is placed.
    The risk manager maintains daily counters that should be reset via reset_daily()
    at the start of each trading day.
    """

    def __init__(self):
        """Initialize with zeroed daily counters."""
        self.daily_pnl_cents: int = 0   # Running daily P&L (can go negative)
        self.trades_today: int = 0       # Number of trades executed today

    def check_signal(
        self, signal: TradingSignal, portfolio: PortfolioSummary
    ) -> tuple[bool, str]:
        """Validate a trading signal against all risk rules.

        Checks (in order): daily loss limit, max positions, duplicate positions,
        minimum edge, sufficient balance, and bet size cap.

        Args:
            signal: The TradingSignal to validate.
            portfolio: Current portfolio state for balance and position checks.

        Returns:
            Tuple of (allowed: bool, reason: str). If allowed is False, reason
            explains why the signal was rejected.
        """

        # Check daily loss limit
        if self.daily_pnl_cents <= -config.max_daily_loss_cents:
            return False, f"Daily loss limit reached (${config.max_daily_loss_cents / 100:.2f})"

        # Check max open positions
        if len(portfolio.positions) >= config.max_open_positions:
            return False, f"Max open positions ({config.max_open_positions}) reached"

        # Check if already have a position in this market
        for pos in portfolio.positions:
            if pos.ticker == signal.ticker:
                return False, f"Already have position in {signal.ticker}"

        # Check minimum edge
        if signal.edge < config.min_edge_threshold:
            return False, f"Edge {signal.edge:.1%} below threshold {config.min_edge_threshold:.1%}"

        # Check sufficient balance
        cost = signal.recommended_size_cents
        if cost > portfolio.balance_cents:
            return False, f"Insufficient balance: need {cost}c, have {portfolio.balance_cents}c"

        # Check bet size limits
        if signal.recommended_size_cents > config.max_bet_amount_cents:
            return False, f"Size {signal.recommended_size_cents}c exceeds max {config.max_bet_amount_cents}c"

        if signal.recommended_size_cents <= 0:
            return False, "Recommended size is zero"

        return True, "OK"

    def kelly_size(self, signal: TradingSignal, bankroll_cents: int) -> int:
        """Calculate position size using the Kelly Criterion.

        Full Kelly formula: f* = (b*p - q) / b
        where:
          b = payout odds = (1 - market_price) / market_price
          p = model's predicted probability of winning
          q = 1 - p (probability of losing)

        The result is multiplied by kelly_fraction (default 0.5 = half-Kelly) to
        reduce variance at the cost of slightly lower expected growth rate.
        Capped at max_bet_amount_cents to prevent oversized positions.

        Args:
            signal: TradingSignal with fair_probability and market_probability.
            bankroll_cents: Current available balance in cents.

        Returns:
            Position size in cents (0 if Kelly fraction is negative or inputs are invalid).
        """
        p = signal.fair_probability
        q = 1 - p
        market_price = signal.market_probability

        if market_price <= 0 or market_price >= 1 or p <= 0:
            return signal.recommended_size_cents

        # Payout odds: if you buy YES at market_price cents and win, you get 100c
        # So payout ratio b = (100 - market_price) / market_price = profit / cost
        b = (1.0 - market_price) / market_price

        if b <= 0:
            return 0  # No positive payout possible

        # Full Kelly fraction: f* = (b*p - q) / b
        # This maximizes long-term geometric growth rate of the bankroll
        kelly_f = (b * p - q) / b
        kelly_f = max(0, kelly_f)  # Never bet negative (would mean the edge is negative)

        # Apply fractional Kelly (default 0.5 = half-Kelly) to reduce variance
        # Half-Kelly achieves ~75% of the growth rate with much less volatility
        kelly_f *= config.kelly_fraction

        # Cap at max bet to prevent any single position from being too large
        size = min(int(kelly_f * bankroll_cents), config.max_bet_amount_cents)
        return max(0, size)

    def build_order(self, signal: TradingSignal, bankroll_cents: int = 0) -> OrderRequest:
        """Convert a validated TradingSignal into a Kalshi OrderRequest ready for execution.

        The limit price is set to the current market probability (converted to cents).
        For NO side, the price is inverted (100 - yes_price) since Kalshi always uses yes_price.
        Contract count is derived from: total_size / price_per_contract.

        Args:
            signal: Validated TradingSignal (should have passed check_signal first).
            bankroll_cents: If > 0, uses Kelly sizing; otherwise uses signal's recommended_size.

        Returns:
            OrderRequest ready to pass to KalshiClient.place_order().
        """
        # For limit orders, set price to current market ask in cents
        price = int(signal.market_probability * 100) if signal.side.value == "yes" else int((1 - signal.market_probability) * 100)

        # Use Kelly sizing if bankroll provided, otherwise fallback to recommended size
        if bankroll_cents > 0:
            size = self.kelly_size(signal, bankroll_cents)
        else:
            size = signal.recommended_size_cents

        count = max(1, size // max(price, 1))

        return OrderRequest(
            ticker=signal.ticker,
            side=signal.side,
            price_cents=price,
            count=count,
        )

    def record_trade(self, pnl_cents: int = 0):
        """Update daily counters after a trade is executed (increment count, add P&L)."""
        self.trades_today += 1
        self.daily_pnl_cents += pnl_cents

    def reset_daily(self):
        """Reset daily P&L and trade count to zero. Call at the start of each trading day."""
        self.daily_pnl_cents = 0
        self.trades_today = 0
