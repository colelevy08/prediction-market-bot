"""
Performance tracking and trade analytics for the prediction market bot.

This module records every completed trade (paper or live) and computes the
quantitative metrics used to evaluate strategy quality:

Key metrics computed:
  - Sharpe Ratio: SR = (mean(log_returns) - Rf) / std(log_returns)
    where Rf = 5% annual / 365 days. Labels: <1 Bad, 1-2 Good, >2 Excellent.
  - Log Returns: ln(P_exit / P_entry) for YES trades, ln(P_entry / P_exit) for NO.
    Log returns are additive over time and handle large price moves correctly.
  - MAE (Max Adverse Excursion): Deepest drawdown during a trade's lifetime.
    Used to optimize stop-loss placement.
  - MFE (Max Favorable Excursion): Best unrealized profit during a trade.
    Used to optimize take-profit targets.
  - Win Rate, Profit Factor, Max Drawdown, Best/Worst trade, Average Edge.

Data structures:
  - TradeRecord: Immutable record of a single completed trade with all fields.
  - PerformanceMetrics: Aggregated statistics computed from all TradeRecords.
  - PerformanceTracker: Stateful tracker that accumulates trades and computes metrics.

The tracker also maintains an equity curve (list of {time, equity_cents, trade_num})
for charting in the frontend, and supports per-category breakdowns and trade journal
notes that persist to Supabase.

Connects to: bot.database (optional Supabase persistence for trades and notes).
Used by: bot.backtester (Backtester and PaperTrader), bot.server (performance endpoints).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TradeRecord:
    """Immutable record of a single completed trade (entry + exit).

    All prices are in the 0-1 range (not cents). PnL and log return are computed
    at recording time by PerformanceTracker.record_trade().
    """
    ticker: str                            # Kalshi market ticker
    side: str                              # "yes" or "no"
    entry_price: float                     # Entry price as decimal (0-1), e.g., 0.45 = 45 cents
    exit_price: float                      # Exit price as decimal (0-1)
    entry_time: str = ""                   # ISO 8601 timestamp
    exit_time: str = ""                    # ISO 8601 timestamp
    contracts: int = 1                     # Number of contracts traded
    pnl_cents: int = 0                     # Profit/loss in cents (computed)
    log_return: float = 0.0                # ln(P_exit / P_entry) or inverse for NO side (computed)
    mae: float = 0.0                       # Max Adverse Excursion: worst drawdown during trade
    mfe: float = 0.0                       # Max Favorable Excursion: best unrealized gain
    model_probability: float = 0.0         # Model's predicted probability at entry time
    market_probability_at_entry: float = 0.0  # Market price at entry time (for edge calculation)
    won: bool = False                      # True if pnl_cents > 0 (computed)
    category: str = ""                     # Market category (politics, crypto, sports, etc.)
    notes: str = ""                        # User-added trade journal notes (editable via API)


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics computed from all completed trades.

    All float values are rounded for display. This dataclass is returned by
    PerformanceTracker.get_metrics() and serialized to JSON for the frontend.
    """
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_cents: int = 0
    sharpe_ratio: float = 0.0
    avg_log_return: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    avg_edge: float = 0.0
    best_trade_pnl: int = 0
    worst_trade_pnl: int = 0
    profit_factor: float = 0.0
    max_drawdown_cents: int = 0
    avg_holding_period_hours: float = 0.0
    sharpe_label: str = "N/A"


class PerformanceTracker:
    """
    Tracks all trades and computes performance metrics from the guide.

    Key formulas:
    - log_return = ln(P1 / P0)
    - Sharpe Ratio = (mean(log_returns) - risk_free) / std(log_returns)
    - MAE/MFE tracking for exit optimization
    """

    # Risk-free rate for Sharpe Ratio calculation: ~5% annual / 365 days
    RISK_FREE_DAILY = 0.05 / 365

    def __init__(self, db=None, mode: str = "paper"):
        """Initialize the performance tracker.

        Args:
            db: Optional Database instance for persisting trades to Supabase.
            mode: "paper" or "live" — used as the mode tag when inserting trades to DB.
        """
        self.db = db                         # Optional Database instance for Supabase persistence
        self.mode = mode                     # "paper" or "live" — tags trades in DB
        self.trades: list[TradeRecord] = []  # Chronological list of all completed trades
        self.equity_curve: list[dict] = []   # [{time, equity_cents, trade_num}] for charting
        self._peak_equity = 0                # High-water mark for drawdown calculation
        self._current_equity = 0             # Running cumulative P&L in cents

    def record_trade(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        exit_price: float,
        contracts: int = 1,
        mae: float = 0.0,
        mfe: float = 0.0,
        model_probability: float = 0.0,
        market_probability_at_entry: float = 0.0,
        entry_time: str = "",
        exit_time: str = "",
        category: str = "",
        notes: str = "",
    ) -> TradeRecord:
        """Record a completed trade, compute its P&L and log return, and update the equity curve.

        Args:
            ticker: Kalshi market ticker.
            side: "yes" or "no".
            entry_price: Entry price as decimal 0-1.
            exit_price: Exit price as decimal 0-1 (1.0 for winning YES, 0.0 for losing YES).
            contracts: Number of contracts.
            mae: Max Adverse Excursion observed during the trade.
            mfe: Max Favorable Excursion observed during the trade.
            model_probability: Model's predicted probability at entry.
            market_probability_at_entry: Market price at entry.
            entry_time: ISO timestamp of entry (defaults to now).
            exit_time: ISO timestamp of exit (defaults to now).
            category: Market category for per-category analytics.
            notes: Optional trade journal notes.

        Returns:
            The completed TradeRecord with computed pnl_cents, log_return, and won fields.
        """

        # PnL in cents
        if side == "yes":
            pnl_cents = int((exit_price - entry_price) * 100 * contracts)
        else:
            pnl_cents = int((entry_price - exit_price) * 100 * contracts)

        # Log return: ln(P1 / P0) — adjusted for side direction
        # For YES trades: profit when price rises, so ln(exit/entry)
        # For NO trades: profit when price drops, so ln(entry/exit)
        if entry_price > 0 and exit_price > 0:
            if side == "yes":
                log_return = math.log(exit_price / entry_price)
            else:
                log_return = math.log(entry_price / exit_price)
        else:
            log_return = 0.0

        won = pnl_cents > 0

        trade = TradeRecord(
            ticker=ticker,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time or datetime.now(timezone.utc).isoformat(),
            exit_time=exit_time or datetime.now(timezone.utc).isoformat(),
            contracts=contracts,
            pnl_cents=pnl_cents,
            log_return=log_return,
            mae=mae,
            mfe=mfe,
            model_probability=model_probability,
            market_probability_at_entry=market_probability_at_entry,
            won=won,
            category=category,
            notes=notes,
        )
        self.trades.append(trade)

        # Persist to DB if connected
        if self.db:
            try:
                self.db.insert_trade(self.mode, trade)
            except Exception:
                pass  # DB writes are best-effort

        # Update equity curve
        self._current_equity += pnl_cents
        self._peak_equity = max(self._peak_equity, self._current_equity)
        self.equity_curve.append({
            "time": trade.exit_time,
            "equity_cents": self._current_equity,
            "trade_num": len(self.trades),
        })

        return trade

    def get_metrics(self) -> PerformanceMetrics:
        """Compute all performance metrics from the accumulated trade history.

        Calculations:
          - Sharpe Ratio: (mean_log_return - risk_free_daily) / std_log_returns
          - Profit Factor: gross_profit / gross_loss
          - Max Drawdown: largest peak-to-trough decline in the equity curve
          - Win Rate: wins / total_trades
          - Average Edge: mean(model_prob - market_prob) across all trades

        Returns:
            PerformanceMetrics dataclass with all computed fields.
        """
        if not self.trades:
            return PerformanceMetrics()

        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        total = len(self.trades)

        # Log returns for Sharpe Ratio
        log_returns = [t.log_return for t in self.trades]
        avg_log_return = sum(log_returns) / total if total > 0 else 0

        # Sharpe Ratio: SR = (Rp - Rf) / σ
        std_returns = self._std(log_returns)
        sharpe = (avg_log_return - self.RISK_FREE_DAILY) / std_returns if std_returns > 0 else 0

        # Sharpe labels from the guide
        if sharpe < 1:
            sharpe_label = "Bad"
        elif sharpe < 2:
            sharpe_label = "Good"
        else:
            sharpe_label = "Excellent"

        # Profit factor
        gross_profit = sum(t.pnl_cents for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_cents for t in losses)) if losses else 1
        profit_factor = gross_profit / max(gross_loss, 1)

        # Max drawdown
        max_dd = 0
        peak = 0
        for point in self.equity_curve:
            eq = point["equity_cents"]
            peak = max(peak, eq)
            dd = peak - eq
            max_dd = max(max_dd, dd)

        pnls = [t.pnl_cents for t in self.trades]

        return PerformanceMetrics(
            total_trades=total,
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / total if total > 0 else 0,
            total_pnl_cents=sum(pnls),
            sharpe_ratio=round(sharpe, 2),
            avg_log_return=round(avg_log_return, 4),
            avg_mae=round(sum(t.mae for t in self.trades) / total, 4) if total > 0 else 0,
            avg_mfe=round(sum(t.mfe for t in self.trades) / total, 4) if total > 0 else 0,
            avg_edge=round(
                sum(t.model_probability - t.market_probability_at_entry for t in self.trades) / total, 4
            ) if total > 0 else 0,
            best_trade_pnl=max(pnls) if pnls else 0,
            worst_trade_pnl=min(pnls) if pnls else 0,
            profit_factor=round(profit_factor, 2),
            max_drawdown_cents=max_dd,
            sharpe_label=sharpe_label,
        )

    def get_equity_curve(self) -> list[dict]:
        """Return equity curve data for charting."""
        return self.equity_curve

    def get_trade_history(self) -> list[dict]:
        """Return trade history as serializable dicts."""
        return [
            {
                "ticker": t.ticker,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "contracts": t.contracts,
                "pnl_cents": t.pnl_cents,
                "log_return": round(t.log_return, 4),
                "mae": round(t.mae, 4),
                "mfe": round(t.mfe, 4),
                "model_probability": round(t.model_probability, 4),
                "won": t.won,
                "category": t.category,
                "notes": t.notes,
            }
            for t in self.trades
        ]

    def get_metrics_by_category(self) -> dict:
        """Compute per-category performance metrics for the frontend heatmap/breakdown.

        Groups trades by their category field (politics, crypto, sports, etc.) and
        computes win rate, total P&L, and best/worst trade for each category.

        Returns:
            Dict mapping category name to a metrics dict.
        """
        from collections import defaultdict
        cats = defaultdict(list)
        for t in self.trades:
            cats[t.category or "uncategorized"].append(t)
        result = {}
        for cat, trades in cats.items():
            wins = [t for t in trades if t.won]
            pnls = [t.pnl_cents for t in trades]
            result[cat] = {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(trades) - len(wins),
                "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
                "total_pnl_cents": sum(pnls),
                "avg_pnl_cents": round(sum(pnls) / len(pnls), 2) if pnls else 0,
                "best_trade": max(pnls) if pnls else 0,
                "worst_trade": min(pnls) if pnls else 0,
            }
        return result

    def update_trade_notes(self, trade_index: int, notes: str) -> bool:
        """Update the journal notes on a specific trade (identified by list index).

        Persists the update to Supabase if connected. Returns False if the index
        is out of range.
        """
        if 0 <= trade_index < len(self.trades):
            self.trades[trade_index].notes = notes
            # Persist to DB if connected
            if self.db:
                try:
                    trade = self.trades[trade_index]
                    self.db.update_trade_notes(trade.ticker, trade.entry_time, notes)
                except Exception:
                    pass  # Best-effort DB persistence
            return True
        return False

    @staticmethod
    def _std(values: list[float]) -> float:
        """Compute sample standard deviation (Bessel's correction: N-1 denominator)."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)
