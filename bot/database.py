"""
Local SQLite persistence layer for the prediction market bot.

Replaces the previous Supabase cloud database with a fully local SQLite file
at data/bot.db. No external dependencies, no network calls, no API keys needed.

SQLite is built into Python's standard library (import sqlite3). Data survives
server restarts and is stored in a single file you own entirely.

Design principles (same as before):
  - All writes are best-effort: exceptions are caught silently so DB issues never
    crash the bot or block trading.
  - All reads return empty lists/None when DB is unavailable.
  - The is_connected property provides a simple check for callers.
  - Thread-safe via check_same_thread=False + WAL mode.

Tables:
  - trades:                Completed trade records (paper and live modes).
  - scan_logs:             Auto-scan execution logs.
  - paper_state:          Paper trader state (singleton row, id=1).
  - performance_snapshots: Periodic performance metric snapshots.
  - model_training_runs:  Training run metadata.
  - training_samples:     Cumulative training data, deduplicated by ticker.

Used by: bot.server (history endpoints), bot.backtester (PaperTrader persistence,
TrainingDataStore sync), bot.performance (trade insertion).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from bot.performance import TradeRecord, PerformanceMetrics

# DB file location — lives alongside other local data files
_DB_PATH = Path(__file__).parent.parent / "data" / "bot.db"

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    mode            TEXT,
    ticker          TEXT,
    side            TEXT,
    entry_price     REAL,
    exit_price      REAL,
    contracts       INTEGER,
    pnl_cents       INTEGER,
    log_return      REAL,
    mae             REAL,
    mfe             REAL,
    model_probability           REAL,
    market_probability_at_entry REAL,
    won             INTEGER,
    entry_time      TEXT,
    exit_time       TEXT,
    category        TEXT,
    notes           TEXT,
    archived        INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scan_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scanned_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    scan_type       TEXT,
    signals_found   INTEGER,
    entries         INTEGER,
    exits           INTEGER,
    open_positions  INTEGER
);

CREATE TABLE IF NOT EXISTS paper_state (
    id              INTEGER PRIMARY KEY,
    balance_cents   INTEGER,
    total_scans     INTEGER,
    signals_seen    INTEGER,
    positions       TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS performance_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    mode                TEXT,
    total_trades        INTEGER,
    wins                INTEGER,
    losses              INTEGER,
    win_rate            REAL,
    total_pnl_cents     INTEGER,
    sharpe_ratio        REAL,
    sharpe_label        TEXT,
    profit_factor       REAL,
    max_drawdown_cents  INTEGER,
    avg_mae             REAL,
    avg_mfe             REAL,
    avg_edge            REAL,
    avg_log_return      REAL,
    equity_curve        TEXT
);

CREATE TABLE IF NOT EXISTS model_training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    samples_used    INTEGER,
    cv_accuracy     REAL,
    oob_score       REAL,
    n_features      INTEGER,
    n_estimators    INTEGER,
    feature_importance TEXT,
    model_version   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS training_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fetched_at  TEXT,
    ticker      TEXT    NOT NULL UNIQUE,
    features    TEXT,
    outcome     INTEGER
);
"""


class Database:
    """SQLite wrapper — every method silently returns empty data when DB is not available.

    This class is designed to be safely instantiated regardless of whether the
    DB file can be created. All read methods return empty lists/None, and all write
    methods are no-ops when self._conn is None.

    The single most important property of this class: it NEVER crashes the bot.
    """

    def __init__(self):
        self._conn: sqlite3.Connection | None = None
        try:
            _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(_DB_PATH),
                check_same_thread=False,
                timeout=10,
            )
            self._conn.row_factory = sqlite3.Row
            # WAL mode: readers don't block writers, safer for concurrent access
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_CREATE_TABLES)
            self._conn.commit()
            self._apply_migrations()
            logger.info(f"Local SQLite DB connected: {_DB_PATH}")
        except Exception as e:
            logger.warning(f"Failed to open local DB at {_DB_PATH}: {e}")
            self._conn = None

    @property
    def is_connected(self) -> bool:
        if self._conn is None:
            return False
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _row_to_dict(self, row) -> dict:
        """Convert a sqlite3.Row to a plain dict, deserializing JSON fields."""
        if row is None:
            return {}
        d = dict(row)
        # Deserialize JSON-encoded columns back to Python objects
        for key in ("positions", "equity_curve", "feature_importance", "features"):
            if key in d and isinstance(d[key], str):
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        return d

    def _apply_migrations(self) -> None:
        """Run one-time schema migrations (idempotent)."""
        if not self._conn:
            return
        try:
            # M1: deduplicate existing trades rows, then add UNIQUE index.
            # All columns are nullable so we can't use a partial index — group by
            # the four fields that identify a real-world position uniquely.
            idx = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='uq_trades_position'"
            ).fetchone()
            if idx is None:
                # Delete duplicates keeping lowest id per group
                self._conn.execute("""
                    DELETE FROM trades
                    WHERE id NOT IN (
                        SELECT MIN(id) FROM trades
                        GROUP BY mode, ticker, entry_time, side, entry_price
                    )
                """)
                self._conn.execute("""
                    CREATE UNIQUE INDEX uq_trades_position
                    ON trades (mode, ticker, entry_time, side, entry_price)
                """)
                self._conn.commit()
                logger.info("Migration M1: deduped trades table + added UNIQUE index")
        except Exception as e:
            logger.warning(f"Migration failed (non-fatal): {e}")

    # ── Trades ───────────────────────────────────────────────────────────────

    def insert_trade(self, mode: str, trade: TradeRecord) -> None:
        if not self._conn:
            return
        row = (
            mode,
            trade.ticker,
            trade.side,
            trade.entry_price,
            trade.exit_price,
            trade.contracts,
            trade.pnl_cents,
            trade.log_return,
            trade.mae,
            trade.mfe,
            trade.model_probability,
            trade.market_probability_at_entry,
            int(bool(trade.won)),
            trade.entry_time or datetime.now(timezone.utc).isoformat(),
            trade.exit_time or datetime.now(timezone.utc).isoformat(),
            getattr(trade, "category", ""),
            getattr(trade, "notes", ""),
        )
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO trades
                   (mode,ticker,side,entry_price,exit_price,contracts,pnl_cents,
                    log_return,mae,mfe,model_probability,market_probability_at_entry,
                    won,entry_time,exit_time,category,notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            self._conn.commit()
            logger.info(f"Trade saved: {trade.ticker} {trade.side} pnl={trade.pnl_cents}c")
        except Exception as e:
            logger.error(f"Failed to insert trade {trade.ticker}: {e}")

    def get_trades(self, mode: str | None = None, limit: int = 100, ticker: str | None = None) -> list[dict]:
        if not self._conn:
            return []
        try:
            filters, params = [], []
            if mode:
                filters.append("mode = ?")
                params.append(mode)
            if ticker:
                filters.append("ticker = ?")
                params.append(ticker)
            where = ("WHERE " + " AND ".join(filters)) if filters else ""
            params.append(limit)
            cur = self._conn.execute(
                f"SELECT * FROM trades {where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to fetch trades: {e}")
            return []

    def update_trade_notes(self, ticker: str, entry_time: str, notes: str) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                "UPDATE trades SET notes=? WHERE ticker=? AND entry_time=?",
                (notes, ticker, entry_time),
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to update trade notes for {ticker}: {e}")

    def archive_old_trades(self, cutoff_iso: str) -> dict:
        if not self._conn:
            return {"archived_count": 0}
        try:
            cur = self._conn.execute(
                "UPDATE trades SET archived=1 WHERE exit_time < ? AND archived=0",
                (cutoff_iso,),
            )
            self._conn.commit()
            return {"archived_count": cur.rowcount}
        except Exception as e:
            logger.warning(f"Archive trades failed: {e}")
            return {"archived_count": 0, "error": str(e)}

    # ── Scan Logs ────────────────────────────────────────────────────────────

    def insert_scan_log(
        self, scan_type: str, signals: int, entries: int, exits: int, open_positions: int
    ) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                "INSERT INTO scan_logs (scan_type,signals_found,entries,exits,open_positions) VALUES (?,?,?,?,?)",
                (scan_type, signals, entries, exits, open_positions),
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert scan log: {e}")

    def get_scan_logs(self, limit: int = 50) -> list[dict]:
        if not self._conn:
            return []
        try:
            cur = self._conn.execute(
                "SELECT * FROM scan_logs ORDER BY scanned_at DESC LIMIT ?", (limit,)
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to fetch scan logs: {e}")
            return []

    # ── Paper State ──────────────────────────────────────────────────────────

    def save_paper_state(
        self, balance: int, scans: int, signals: int, positions: dict
    ) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                """INSERT INTO paper_state (id,balance_cents,total_scans,signals_seen,positions,updated_at)
                   VALUES (1,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     balance_cents=excluded.balance_cents,
                     total_scans=excluded.total_scans,
                     signals_seen=excluded.signals_seen,
                     positions=excluded.positions,
                     updated_at=excluded.updated_at""",
                (balance, scans, signals, json.dumps(positions), datetime.now(timezone.utc).isoformat()),
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to save paper state: {e}")

    def load_paper_state(self) -> dict | None:
        if not self._conn:
            return None
        try:
            cur = self._conn.execute("SELECT * FROM paper_state WHERE id=1")
            row = cur.fetchone()
            return self._row_to_dict(row) if row else None
        except Exception as e:
            logger.warning(f"Failed to load paper state: {e}")
            return None

    # ── Performance Snapshots ────────────────────────────────────────────────

    def insert_performance_snapshot(
        self, mode: str, metrics: PerformanceMetrics, equity_curve: list
    ) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                """INSERT INTO performance_snapshots
                   (mode,total_trades,wins,losses,win_rate,total_pnl_cents,
                    sharpe_ratio,sharpe_label,profit_factor,max_drawdown_cents,
                    avg_mae,avg_mfe,avg_edge,avg_log_return,equity_curve)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mode,
                    metrics.total_trades,
                    metrics.wins,
                    metrics.losses,
                    metrics.win_rate,
                    metrics.total_pnl_cents,
                    metrics.sharpe_ratio,
                    metrics.sharpe_label,
                    metrics.profit_factor,
                    metrics.max_drawdown_cents,
                    metrics.avg_mae,
                    metrics.avg_mfe,
                    metrics.avg_edge,
                    metrics.avg_log_return,
                    json.dumps(equity_curve),
                ),
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to insert performance snapshot: {e}")

    def get_performance_history(self, mode: str = "paper", limit: int = 50) -> list[dict]:
        if not self._conn:
            return []
        try:
            cur = self._conn.execute(
                "SELECT * FROM performance_snapshots WHERE mode=? ORDER BY snapshot_at DESC LIMIT ?",
                (mode, limit),
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to fetch performance history: {e}")
            return []

    # ── Model Training Runs ──────────────────────────────────────────────────

    def insert_training_run(
        self,
        samples: int,
        cv_accuracy: float,
        oob_score: float,
        n_features: int,
        n_estimators: int,
        feature_importance: dict,
        model_version: int = 0,
    ) -> None:
        if not self._conn:
            return
        try:
            self._conn.execute(
                """INSERT INTO model_training_runs
                   (samples_used,cv_accuracy,oob_score,n_features,n_estimators,feature_importance,model_version)
                   VALUES (?,?,?,?,?,?,?)""",
                (samples, cv_accuracy, oob_score, n_features, n_estimators,
                 json.dumps(feature_importance), model_version),
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert training run: {e}")

    def get_training_history(self, limit: int = 10) -> list[dict]:
        if not self._conn:
            return []
        try:
            cur = self._conn.execute(
                "SELECT * FROM model_training_runs ORDER BY trained_at DESC LIMIT ?", (limit,)
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.warning(f"Failed to fetch training history: {e}")
            return []

    # ── Training Samples ─────────────────────────────────────────────────────

    def insert_training_samples(self, samples: list[dict]) -> None:
        if not self._conn or not samples:
            return
        rows = []
        for s in samples:
            ticker = s.get("ticker", "")
            if not ticker:
                continue
            rows.append((
                ticker,
                json.dumps(s.get("features", {})),
                s.get("outcome", 0),
                datetime.now(timezone.utc).isoformat(),
            ))
        if not rows:
            return
        try:
            self._conn.executemany(
                """INSERT INTO training_samples (ticker,features,outcome,fetched_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(ticker) DO UPDATE SET
                     features=excluded.features,
                     outcome=excluded.outcome,
                     fetched_at=excluded.fetched_at""",
                rows,
            )
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Failed to upsert training samples: {e}")

    def get_training_samples(self, limit: int = 50000) -> list[dict]:
        if not self._conn:
            return []
        try:
            cur = self._conn.execute(
                "SELECT ticker,features,outcome,fetched_at FROM training_samples LIMIT ?", (limit,)
            )
            return [self._row_to_dict(r) for r in cur.fetchall()]
        except Exception:
            return []
