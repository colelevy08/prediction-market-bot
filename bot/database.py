"""Optional Supabase persistence layer. All methods are no-ops when not configured."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

from bot.config import config
from bot.performance import TradeRecord, PerformanceMetrics


class Database:
    """Supabase wrapper — every method silently returns when DB is not configured."""

    def __init__(self):
        self.client: Client | None = None
        if HAS_SUPABASE and config.validate_supabase():
            self.client = create_client(config.supabase_url, config.supabase_key)

    @property
    def is_connected(self) -> bool:
        return self.client is not None

    # ── Trades ───────────────────────────────────────────────────

    def insert_trade(self, mode: str, trade: TradeRecord) -> None:
        if not self.client:
            return
        self.client.table("trades").insert({
            "mode": mode,
            "ticker": trade.ticker,
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "contracts": trade.contracts,
            "pnl_cents": trade.pnl_cents,
            "log_return": trade.log_return,
            "mae": trade.mae,
            "mfe": trade.mfe,
            "model_probability": trade.model_probability,
            "market_probability_at_entry": trade.market_probability_at_entry,
            "won": trade.won,
            "entry_time": trade.entry_time or datetime.now(timezone.utc).isoformat(),
            "exit_time": trade.exit_time or datetime.now(timezone.utc).isoformat(),
            "category": getattr(trade, "category", ""),
            "notes": getattr(trade, "notes", ""),
        }).execute()

    def get_trades(self, mode: str | None = None, limit: int = 100) -> list[dict]:
        if not self.client:
            return []
        query = self.client.table("trades").select("*").order("created_at", desc=True).limit(limit)
        if mode:
            query = query.eq("mode", mode)
        return query.execute().data

    # ── Scan Logs ────────────────────────────────────────────────

    def insert_scan_log(
        self, scan_type: str, signals: int, entries: int, exits: int, open_positions: int
    ) -> None:
        if not self.client:
            return
        self.client.table("scan_logs").insert({
            "scan_type": scan_type,
            "signals_found": signals,
            "entries": entries,
            "exits": exits,
            "open_positions": open_positions,
        }).execute()

    def get_scan_logs(self, limit: int = 50) -> list[dict]:
        if not self.client:
            return []
        return (
            self.client.table("scan_logs")
            .select("*")
            .order("scanned_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )

    # ── Paper State ──────────────────────────────────────────────

    def save_paper_state(
        self, balance: int, scans: int, signals: int, positions: dict
    ) -> None:
        if not self.client:
            return
        self.client.table("paper_state").upsert({
            "id": 1,
            "balance_cents": balance,
            "total_scans": scans,
            "signals_seen": signals,
            "positions": positions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

    def load_paper_state(self) -> dict | None:
        if not self.client:
            return None
        rows = self.client.table("paper_state").select("*").eq("id", 1).execute().data
        return rows[0] if rows else None

    # ── Performance Snapshots ────────────────────────────────────

    def insert_performance_snapshot(
        self, mode: str, metrics: PerformanceMetrics, equity_curve: list
    ) -> None:
        if not self.client:
            return
        self.client.table("performance_snapshots").insert({
            "mode": mode,
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate": metrics.win_rate,
            "total_pnl_cents": metrics.total_pnl_cents,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sharpe_label": metrics.sharpe_label,
            "profit_factor": metrics.profit_factor,
            "max_drawdown_cents": metrics.max_drawdown_cents,
            "avg_mae": metrics.avg_mae,
            "avg_mfe": metrics.avg_mfe,
            "avg_edge": metrics.avg_edge,
            "avg_log_return": metrics.avg_log_return,
            "equity_curve": equity_curve,
        }).execute()

    def get_performance_history(self, mode: str = "paper", limit: int = 50) -> list[dict]:
        if not self.client:
            return []
        return (
            self.client.table("performance_snapshots")
            .select("*")
            .eq("mode", mode)
            .order("snapshot_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )

    # ── Model Training Runs ──────────────────────────────────────

    def insert_training_run(
        self,
        samples: int,
        cv_accuracy: float,
        oob_score: float,
        n_features: int,
        n_estimators: int,
        feature_importance: dict,
    ) -> None:
        if not self.client:
            return
        self.client.table("model_training_runs").insert({
            "samples_used": samples,
            "cv_accuracy": cv_accuracy,
            "oob_score": oob_score,
            "n_features": n_features,
            "n_estimators": n_estimators,
            "feature_importance": feature_importance,
        }).execute()

    def get_training_history(self, limit: int = 10) -> list[dict]:
        if not self.client:
            return []
        return (
            self.client.table("model_training_runs")
            .select("*")
            .order("trained_at", desc=True)
            .limit(limit)
            .execute()
            .data
        )
