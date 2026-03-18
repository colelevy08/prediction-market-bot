"""
Supabase persistence layer for the prediction market bot.

This module wraps the Supabase Python client to provide persistent storage for
all bot data. Every method is designed as a safe no-op when Supabase is not
configured (missing SUPABASE_URL or SUPABASE_KEY), so the bot runs fully
functional in local-only mode without a database.

Supabase tables used:
  - trades:                 Completed trade records (paper and live modes).
  - scan_logs:              Auto-scan execution logs (events/markets scanned, entries/exits).
  - paper_state:            Paper trader state snapshot (balance, positions, scan count).
                            Uses upsert with id=1 (singleton row pattern).
  - performance_snapshots:  Periodic performance metric snapshots for trend analysis.
  - model_training_runs:    Training run metadata (samples, CV accuracy, OOB score,
                            feature importance dict).
  - training_samples:       Cumulative training data (features + outcomes), deduplicated
                            by ticker. Batch-upserted in 500-row chunks to stay under
                            Supabase payload limits.

Design principles:
  - All writes are best-effort: exceptions are caught silently so DB issues never
    crash the bot or block trading.
  - All reads return empty lists/None when DB is unavailable.
  - The is_connected property provides a simple check for callers.

Connects to: Supabase cloud instance configured via SUPABASE_URL and SUPABASE_KEY
environment variables. Uses the supabase-py client library.

Used by: bot.server (history endpoints), bot.backtester (PaperTrader persistence,
TrainingDataStore sync), bot.performance (trade insertion).
"""

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
    """Supabase wrapper — every method silently returns empty data when DB is not configured.

    This class is designed to be safely instantiated regardless of whether Supabase
    credentials are present. All read methods return empty lists/None, and all write
    methods are no-ops when self.client is None.
    """

    def __init__(self):
        """Initialize the Supabase client if credentials are configured.

        Only creates a connection if both SUPABASE_URL and SUPABASE_KEY are set
        AND the supabase-py library is installed (optional dependency).
        """
        self.client: Client | None = None
        if HAS_SUPABASE and config.validate_supabase():
            self.client = create_client(config.supabase_url, config.supabase_key)

    @property
    def is_connected(self) -> bool:
        """Check if the Supabase client is initialized and ready for queries."""
        return self.client is not None

    # ── Trades ───────────────────────────────────────────────────

    def insert_trade(self, mode: str, trade: TradeRecord) -> None:
        """Insert a completed trade record into the 'trades' table.

        Args:
            mode: "paper" or "live" — distinguishes simulated vs real trades.
            trade: TradeRecord with all trade details.
        """
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
        """Fetch recent trades from the 'trades' table, ordered by most recent first.

        Args:
            mode: Optional filter — "paper" or "live". If None, returns all modes.
            limit: Maximum number of trades to return.
        """
        if not self.client:
            return []
        query = self.client.table("trades").select("*").order("created_at", desc=True).limit(limit)
        if mode:
            query = query.eq("mode", mode)
        return query.execute().data

    def update_trade_notes(self, ticker: str, entry_time: str, notes: str) -> None:
        """Update notes on a trade identified by ticker + entry_time."""
        if not self.client:
            return
        self.client.table("trades").update(
            {"notes": notes}
        ).eq("ticker", ticker).eq("entry_time", entry_time).execute()

    # ── Scan Logs ────────────────────────────────────────────────

    def insert_scan_log(
        self, scan_type: str, signals: int, entries: int, exits: int, open_positions: int
    ) -> None:
        """Log a scan execution to the 'scan_logs' table for historical tracking.

        Args:
            scan_type: "paper" or "live".
            signals: Number of signals generated during the scan.
            entries: Number of new positions opened.
            exits: Number of positions closed.
            open_positions: Total open positions after the scan.
        """
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
        """Persist the paper trader's current state to the 'paper_state' table.

        Uses upsert with id=1 (singleton row pattern) so there's always exactly
        one row representing the current paper trading state.
        """
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
        """Load the paper trader's persisted state. Returns None if no state exists."""
        if not self.client:
            return None
        rows = self.client.table("paper_state").select("*").eq("id", 1).execute().data
        return rows[0] if rows else None

    # ── Performance Snapshots ────────────────────────────────────

    def insert_performance_snapshot(
        self, mode: str, metrics: PerformanceMetrics, equity_curve: list
    ) -> None:
        """Save a periodic snapshot of performance metrics for trend analysis over time."""
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
        """Record a model training run with its parameters and accuracy metrics.

        The feature_importance dict is stored as a JSON column and maps feature
        names to their importance scores (combined RF + GB).
        """
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

    # ── Training Samples (Cumulative) ─────────────────────────────

    def insert_training_samples(self, samples: list[dict]) -> None:
        """Batch-upsert training samples, deduplicated by ticker.

        Uses Supabase upsert with on_conflict="ticker" so re-inserting an already
        seen ticker just updates the row rather than creating a duplicate.
        Batches in chunks of 500 rows to stay under Supabase's payload size limits.
        """
        if not self.client or not samples:
            return
        rows = []
        for s in samples:
            ticker = s.get("ticker", "")
            if not ticker:
                continue
            rows.append({
                "ticker": ticker,
                "features": s.get("features", {}),  # 106-feature dict stored as JSON column
                "outcome": s.get("outcome", 0),      # Binary outcome: 1 = YES resolved, 0 = NO resolved
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
        if rows:
            # Batch in chunks of 500 to avoid Supabase payload limits (~2MB)
            for i in range(0, len(rows), 500):
                chunk = rows[i:i+500]
                try:
                    self.client.table("training_samples").upsert(
                        chunk, on_conflict="ticker"
                    ).execute()
                except Exception:
                    pass  # Best-effort — table may not exist yet in new deployments

    def get_training_samples(self, limit: int = 50000) -> list[dict]:
        """Load all persisted training samples."""
        if not self.client:
            return []
        try:
            return (
                self.client.table("training_samples")
                .select("ticker,features,outcome,fetched_at")
                .limit(limit)
                .execute()
                .data
            )
        except Exception:
            return []  # Table may not exist yet
