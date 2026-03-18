"""
Performance tracking: Sharpe Ratio, log returns, MAE/MFE, win rate.

From the guide:
- Sharpe Ratio: SR = (Rp - Rf) / σ  — profit per unit of risk
  SR < 1 = bad, SR 1-2 = good, SR > 2 = excellent
- Log returns: ln(P1 / P0) — additive over time, correct for big moves
- MAE: how deep position went into the red before closing
- MFE: how high it went before you sold
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    ticker: str
    side: str
    entry_price: float       # 0-1
    exit_price: float        # 0-1
    entry_time: str = ""
    exit_time: str = ""
    contracts: int = 1
    pnl_cents: int = 0
    log_return: float = 0.0
    mae: float = 0.0         # max adverse excursion (worst drawdown during trade)
    mfe: float = 0.0         # max favorable excursion (best unrealized gain)
    model_probability: float = 0.0
    market_probability_at_entry: float = 0.0
    won: bool = False
    category: str = ""       # market category (politics, crypto, sports, etc.)
    notes: str = ""          # user-added trade journal notes


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics."""
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

    RISK_FREE_DAILY = 0.05 / 365  # ~5% annual risk-free rate

    def __init__(self, db=None, mode: str = "paper"):
        self.db = db  # Optional Database instance
        self.mode = mode
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[dict] = []  # [{time, equity_cents}]
        self._peak_equity = 0
        self._current_equity = 0

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
        """Record a completed trade and compute its metrics."""

        # Log return: ln(P1 / P0) — from the guide
        if entry_price > 0 and exit_price > 0:
            log_return = math.log(exit_price / entry_price)
        else:
            log_return = 0.0

        # PnL in cents
        if side == "yes":
            pnl_cents = int((exit_price - entry_price) * 100 * contracts)
        else:
            pnl_cents = int((entry_price - exit_price) * 100 * contracts)

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
        """Compute all performance metrics from the guide."""
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
        """Compute per-category performance metrics."""
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
        """Update notes on a trade by index."""
        if 0 <= trade_index < len(self.trades):
            self.trades[trade_index].notes = notes
            return True
        return False

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)
