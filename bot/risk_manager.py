"""Risk management: position limits, daily loss limits, and order validation."""

from __future__ import annotations

from bot.config import config
from bot.models import OrderRequest, PortfolioSummary, TradingSignal


class RiskManager:
    """Enforces trading limits and risk rules before order execution."""

    def __init__(self):
        self.daily_pnl_cents: int = 0
        self.trades_today: int = 0

    def check_signal(
        self, signal: TradingSignal, portfolio: PortfolioSummary
    ) -> tuple[bool, str]:
        """Validate a trading signal against risk rules. Returns (allowed, reason)."""

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

    def build_order(self, signal: TradingSignal) -> OrderRequest:
        """Convert a validated signal into an order request."""
        # For limit orders, use the current ask price
        price = int(signal.market_probability * 100) if signal.side.value == "yes" else int((1 - signal.market_probability) * 100)
        # Contracts = total_risk / price_per_contract
        count = max(1, signal.recommended_size_cents // max(price, 1))

        return OrderRequest(
            ticker=signal.ticker,
            side=signal.side,
            price_cents=price,
            count=count,
        )

    def record_trade(self, pnl_cents: int = 0):
        """Update daily tracking after a trade."""
        self.trades_today += 1
        self.daily_pnl_cents += pnl_cents

    def reset_daily(self):
        """Reset daily counters (call at start of each trading day)."""
        self.daily_pnl_cents = 0
        self.trades_today = 0
