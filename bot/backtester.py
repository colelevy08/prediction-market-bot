"""
Historical backtester, cumulative training data store, and paper trading simulator.

This module provides three major components for testing and running the strategy:

1. TrainingDataStore:
   - Persists training samples (106-feature vectors + binary outcomes) to JSON + Supabase.
   - Samples are deduplicated by market ticker so each settled market is stored once.
   - Every training run builds cumulatively on all previously seen data rather than
     discarding old samples, so the model improves monotonically with more data.
   - Primary storage: local JSON file (data/training_samples.json).
   - Fallback/sync: Supabase table 'training_samples' (batch upsert, 500-row chunks).

2. HistoricalDataFetcher:
   - Fetches settled (resolved) markets from Kalshi via paginated API calls.
   - Reconstructs realistic pre-settlement prices from trade history (last 50 trades)
     so the model trains on actual market conditions, not post-settlement 0/100 prices.
   - Falls back to last_price or previous_price when trade history is unavailable.

3. Backtester:
   - Runs the full RF+GB ensemble strategy on historical data.
   - Splits data into train/test sets (default 60/40), trains the model, then replays
     the strategy on the test set using the guide's entry/exit/confidence rules.
   - Reports: Sharpe Ratio, win rate, profit factor, max drawdown, MAE/MFE, equity curve.
   - parameter_sweep() tests grid of (entry_threshold x confidence_level) combinations,
     sorted by Sharpe Ratio to find optimal settings without overfitting.

4. PaperTrader:
   - Live paper trading simulator that processes real market data without placing orders.
   - Runs the full signal generation pipeline: snapshot recording, entry signals, exit checks.
   - Tracks positions (PaperPosition dataclass) with MAE/MFE during the trade's lifetime.
   - Persists state to both JSON file and Supabase for crash recovery.
   - Integrates TrainingDataStore for cumulative model retraining.

Connects to: KalshiClient (market data), RFSignalGenerator (signals), PerformanceTracker
(metrics), Database (Supabase persistence), config (trading parameters).

Used by: bot.server (REST endpoints for /api/backtest, /api/paper/*).
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bot.config import config
from bot.kalshi_client import KalshiClient, _dollars_to_cents, _parse_fp
from bot.models import Event, Market, Side
from bot.rf_model import extract_features, FEATURE_NAMES, PredictionModel, RFSignalGenerator
from bot.performance import PerformanceTracker, TradeRecord

# Default persistence directory
DATA_DIR = Path(__file__).parent.parent / "data"

import logging
logger = logging.getLogger("predictionbot")


# ── Cumulative Training Data Store ───────────────────────────────────────────

class TrainingDataStore:
    """
    Persists training samples (features + outcomes) so each training run
    builds on all previously seen data rather than discarding it.

    Samples are deduplicated by ticker (each settled market has a unique ticker).
    Stored as a JSON file locally and optionally synced to Supabase.
    """

    def __init__(self, path: Path | None = None, db=None):
        self.path = path or DATA_DIR / "training_samples.json"
        self.db = db
        self.samples: dict[str, dict] = {}  # keyed by ticker for dedup
        self._load()

    def _load(self):
        """Load existing samples from disk (primary) or DB (fallback)."""
        # Try local JSON first
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for s in data.get("samples", []):
                    ticker = s.get("ticker", "")
                    if ticker:
                        self.samples[ticker] = s
                logger.info(f"TrainingDataStore: loaded {len(self.samples)} samples from disk")
                return
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"TrainingDataStore: failed to load from disk: {e}")

        # Fallback to DB
        if self.db and self.db.is_connected:
            try:
                rows = self.db.get_training_samples()
                for s in rows:
                    ticker = s.get("ticker", "")
                    if ticker:
                        self.samples[ticker] = {
                            "ticker": ticker,
                            "features": s.get("features", {}),
                            "outcome": s.get("outcome", 0),
                            "fetched_at": s.get("fetched_at", ""),
                        }
                if self.samples:
                    logger.info(f"TrainingDataStore: loaded {len(self.samples)} samples from DB")
                    self._save_to_disk()  # Cache locally
            except Exception as e:
                logger.warning(f"TrainingDataStore: failed to load from DB: {e}")

    def add_samples(self, settled_data: list[dict]) -> int:
        """
        Merge new training samples into the store, deduplicating by ticker.
        Returns the number of NEW samples added.
        """
        new_count = 0
        now = datetime.now(timezone.utc).isoformat()

        for item in settled_data:
            market = item.get("market")
            ticker = market.ticker if hasattr(market, "ticker") else item.get("ticker", "")
            if not ticker or ticker in self.samples:
                continue

            self.samples[ticker] = {
                "ticker": ticker,
                "features": item["features"],
                "outcome": item["outcome"],
                "fetched_at": now,
            }
            new_count += 1

        if new_count > 0:
            self._save_to_disk()
            self._save_to_db(settled_data[-new_count:] if new_count <= len(settled_data) else settled_data)
            logger.info(f"TrainingDataStore: added {new_count} new samples (total: {len(self.samples)})")

        return new_count

    def get_all_samples(self) -> list[dict]:
        """Return all stored samples as a list of {features, outcome} dicts."""
        return list(self.samples.values())

    def get_features_and_outcomes(self) -> tuple[list[dict], list[int]]:
        """Return features list and outcomes list ready for model training."""
        features = []
        outcomes = []
        for s in self.samples.values():
            features.append(s["features"])
            outcomes.append(s["outcome"])
        return features, outcomes

    @property
    def count(self) -> int:
        return len(self.samples)

    def _save_to_disk(self):
        """Persist all samples to JSON file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_count": len(self.samples),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "samples": list(self.samples.values()),
        }
        self.path.write_text(json.dumps(data, default=str))

    def _save_to_db(self, new_samples: list[dict]):
        """Batch-insert new samples to Supabase (best-effort)."""
        if not self.db or not self.db.is_connected:
            return
        try:
            self.db.insert_training_samples([
                {
                    "ticker": s.get("ticker", s.get("market", {}).ticker if hasattr(s.get("market", {}), "ticker") else ""),
                    "features": s.get("features", {}),
                    "outcome": s.get("outcome", 0),
                }
                for s in new_samples
            ])
        except Exception as e:
            logger.warning(f"TrainingDataStore: DB save failed: {e}")


# ── Historical Data Fetcher ──────────────────────────────────────────────────

class HistoricalDataFetcher:
    """Fetches settled markets from Kalshi for backtesting and training."""

    def __init__(self, kalshi: KalshiClient):
        self.kalshi = kalshi

    def fetch_settled_markets(self, limit: int = 200) -> list[dict]:
        """
        Fetch settled (resolved) markets with their outcomes.
        Reconstructs pre-settlement features from trade history so the model
        can learn from realistic price/volume data instead of post-settlement zeros.
        Paginates through multiple pages to get up to `limit` samples.
        """
        settled = []
        cursor = None
        page_size = min(limit, 200)

        while len(settled) < limit:
            params = {
                "limit": page_size,
                "status": "settled",
                "with_nested_markets": "true",
            }
            if cursor:
                params["cursor"] = cursor

            data = self.kalshi._request("GET", "/events", params=params)

            events = data.get("events", [])
            if not events:
                break

            for ev in events:
                event = Event(
                    event_ticker=ev.get("event_ticker", ""),
                    title=ev.get("title", ""),
                    category=ev.get("category", ""),
                )
                for m in ev.get("markets", []):
                    result = m.get("result", "")
                    if result not in ("yes", "no"):
                        continue

                    ticker = m.get("ticker", "")
                    volume = _parse_fp(m.get("volume_fp")) or m.get("volume", 0) or 0
                    last_price = _dollars_to_cents(m.get("last_price_dollars")) or 0
                    prev_price = _dollars_to_cents(m.get("previous_price_dollars")) or 0

                    # Reconstruct pre-settlement prices from trade history
                    trade_prices = []
                    try:
                        trades_data = self.kalshi._request("GET", "/markets/trades", params={
                            "ticker": ticker, "limit": 50,
                        })
                        for t in trades_data.get("trades", []):
                            tp = _dollars_to_cents(t.get("yes_price")) or t.get("yes_price", 0)
                            if isinstance(tp, (int, float)) and 1 <= tp <= 99:
                                trade_prices.append(int(tp))
                    except Exception:
                        pass

                    # Use trade history to reconstruct realistic bid/ask/mid
                    if trade_prices:
                        avg_price = int(np.mean(trade_prices))
                        spread = max(2, int(np.std(trade_prices) * 0.5)) if len(trade_prices) > 2 else 4
                    elif last_price and 1 <= last_price <= 99:
                        avg_price = last_price
                        spread = 4
                    elif prev_price and 1 <= prev_price <= 99:
                        avg_price = prev_price
                        spread = 6
                    else:
                        # Skip markets with no price data
                        continue

                    yes_bid = max(1, avg_price - spread // 2)
                    yes_ask = min(99, avg_price + spread // 2)
                    no_bid = max(1, 100 - yes_ask)
                    no_ask = min(99, 100 - yes_bid)

                    market = Market(
                        ticker=ticker,
                        event_ticker=m.get("event_ticker", ""),
                        title=m.get("title", ""),
                        subtitle=m.get("subtitle", ""),
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        no_bid=no_bid,
                        no_ask=no_ask,
                        volume=int(volume) if isinstance(volume, (int, float)) else 0,
                        open_interest=_parse_fp(m.get("open_interest_fp")) or m.get("open_interest", 0) or 0,
                        status="settled",
                        close_time=m.get("close_time", ""),
                        result=result,
                        category=m.get("category", ""),
                        last_price=last_price,
                        prev_price=prev_price,
                    )

                    # Build features with the reconstructed prices
                    history = [{"yes_mid": p, "volume": volume} for p in trade_prices] if trade_prices else None
                    features = extract_features(market, event, history)
                    settled.append({
                        "market": market,
                        "event": event,
                        "features": features,
                        "outcome": 1 if result == "yes" else 0,
                        "result": result,
                        "n_trades": len(trade_prices),
                    })

            # Check for next page cursor
            cursor = data.get("cursor")
            if not cursor:
                break

        return settled[:limit]

    def fetch_historical_trades(self, ticker: str, limit: int = 100) -> list[dict]:
        """Fetch trade history for a specific market."""
        try:
            data = self.kalshi._request("GET", "/markets/trades", params={
                "ticker": ticker,
                "limit": limit,
            })
            return data.get("trades", [])
        except Exception:
            return []


# ── Backtester ───────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """Configuration parameters for a single backtest run.

    These mirror the guide's recommended settings and can be swept via
    Backtester.parameter_sweep() to find optimal values.
    """
    initial_balance_cents: int = 100_00  # Starting paper balance ($100)
    max_bet_cents: int = 25_00           # Maximum bet per trade ($25)
    entry_threshold: float = 0.5         # Entry rule: market_price <= model_prob * threshold
    exit_threshold: float = 0.9          # Exit rule: market_price >= model_prob * threshold
    min_confidence: float = 0.70         # Minimum model confidence to enter (70%)
    min_volume: int = 50                 # Skip markets with < 50 contracts traded
    max_positions: int = 10              # Max simultaneous open positions
    train_ratio: float = 0.6            # Train/test split (60% train, 40% test)


@dataclass
class BacktestResult:
    """Complete backtest results including training metrics, trading metrics, and details."""
    config: dict = field(default_factory=dict)
    # Training metrics
    train_samples: int = 0
    test_samples: int = 0
    cv_accuracy: float = 0.0
    oob_score: float = 0.0
    # Trading metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_cents: int = 0
    sharpe_ratio: float = 0.0
    sharpe_label: str = "N/A"
    profit_factor: float = 0.0
    max_drawdown_cents: int = 0
    avg_edge: float = 0.0
    avg_log_return: float = 0.0
    # Details
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    signals_generated: int = 0
    signals_filtered: int = 0


class Backtester:
    """
    Runs the full strategy on historical data.

    1. Fetches settled markets from Kalshi
    2. Splits into train/test sets
    3. Trains the ensemble model on train set
    4. Replays the strategy on test set
    5. Computes full performance metrics
    """

    def __init__(self, kalshi: KalshiClient | None = None):
        """Initialize the backtester with an optional Kalshi client for data fetching.

        Args:
            kalshi: If provided, used to fetch settled markets. If None, settled_data
                    must be passed directly to run().
        """
        self.kalshi = kalshi
        self.model = PredictionModel(n_estimators=500)  # Fresh ensemble for each backtest
        self.tracker = PerformanceTracker()               # Tracks per-trade metrics

    def run(
        self,
        settled_data: list[dict] | None = None,
        cfg: BacktestConfig | None = None,
    ) -> BacktestResult:
        """Run a full backtest on historical settled market data.

        Steps:
          1. Fetch or receive settled market data with features and outcomes.
          2. Shuffle deterministically (seed=42) and split into train/test sets.
          3. Train the RF+GB ensemble on the training set.
          4. Replay the strategy on the test set: apply entry rules, simulate
             resolution, compute P&L, track MAE/MFE.
          5. Compile all metrics into a BacktestResult.

        Args:
            settled_data: Pre-fetched settled market data. If None, fetches from Kalshi.
            cfg: Backtest configuration. If None, uses defaults.

        Returns:
            BacktestResult with training metrics, trading metrics, trades, equity curve,
            and feature importance rankings.
        """
        cfg = cfg or BacktestConfig()
        result = BacktestResult(config={
            "initial_balance": cfg.initial_balance_cents,
            "max_bet": cfg.max_bet_cents,
            "entry_threshold": cfg.entry_threshold,
            "exit_threshold": cfg.exit_threshold,
            "min_confidence": cfg.min_confidence,
            "train_ratio": cfg.train_ratio,
        })

        # Get data
        if settled_data is None:
            if not self.kalshi:
                return BacktestResult(config={"error": "No data or Kalshi client provided"})
            fetcher = HistoricalDataFetcher(self.kalshi)
            settled_data = fetcher.fetch_settled_markets(limit=200)

        if len(settled_data) < 50:
            result.config["error"] = f"Only {len(settled_data)} settled markets found. Need 50+."
            return result

        # Shuffle with fixed seed for reproducibility, then split into train/test
        random.seed(42)
        data = list(settled_data)
        random.shuffle(data)

        split_idx = int(len(data) * cfg.train_ratio)
        train_data = data[:split_idx]
        test_data = data[split_idx:]

        result.train_samples = len(train_data)
        result.test_samples = len(test_data)

        # Train ensemble model
        train_features = [d["features"] for d in train_data]
        train_outcomes = [d["outcome"] for d in train_data]
        train_result = self.model.train_on_historical(train_features, train_outcomes)

        if isinstance(train_result, dict):
            result.cv_accuracy = train_result.get("cv_accuracy", 0)
            result.oob_score = train_result.get("oob_score", 0)

        # Replay strategy on test set
        balance = cfg.initial_balance_cents
        self.tracker = PerformanceTracker()

        for item in test_data:
            market: Market = item["market"]
            features: dict = item["features"]
            actual_outcome: int = item["outcome"]

            if market.volume < cfg.min_volume:
                continue

            # Model prediction
            model_prob = self.model.predict_probability(features)
            market_price = market.mid_price_yes / 100

            # Confidence check (guide: 70%+)
            confidence = abs(model_prob - 0.5) * 2
            if confidence < cfg.min_confidence:
                result.signals_filtered += 1
                continue

            # Entry rule (guide: market_price <= model_prob * 0.5)
            if model_prob > 0.5 and market_price <= model_prob * cfg.entry_threshold:
                side = "yes"
                entry_price = market_price
                edge = model_prob - market_price
            elif model_prob < 0.5 and (1 - market_price) <= (1 - model_prob) * cfg.entry_threshold:
                side = "no"
                entry_price = 1 - market_price
                edge = (1 - model_prob) - (1 - market_price)
            else:
                result.signals_filtered += 1
                continue

            result.signals_generated += 1

            # Position sizing
            bet_size = int(cfg.max_bet_cents * min(edge, 0.5) * confidence)
            bet_size = max(1, min(bet_size, cfg.max_bet_cents, balance))
            contracts = max(1, bet_size // max(int(entry_price * 100), 1))

            if contracts * int(entry_price * 100) > balance:
                continue

            # Simulate resolution
            if side == "yes":
                exit_price = 1.0 if actual_outcome == 1 else 0.0
                pnl = (exit_price - entry_price) * 100 * contracts
            else:
                exit_price = 0.0 if actual_outcome == 1 else 1.0
                pnl = ((1 - entry_price) - (1 - exit_price)) * 100 * contracts if actual_outcome == 0 else -entry_price * 100 * contracts

            # Calculate MAE/MFE (simplified for settled markets)
            mae = entry_price if (side == "yes" and actual_outcome == 0) or (side == "no" and actual_outcome == 1) else 0
            mfe = (1 - entry_price) if (side == "yes" and actual_outcome == 1) or (side == "no" and actual_outcome == 0) else 0

            self.tracker.record_trade(
                ticker=market.ticker,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                contracts=contracts,
                mae=mae,
                mfe=mfe,
                model_probability=model_prob,
                market_probability_at_entry=market_price,
            )

            balance += int(pnl)

        # Compile results
        metrics = self.tracker.get_metrics()
        result.total_trades = metrics.total_trades
        result.wins = metrics.wins
        result.losses = metrics.losses
        result.win_rate = metrics.win_rate
        result.total_pnl_cents = metrics.total_pnl_cents
        result.sharpe_ratio = metrics.sharpe_ratio
        result.sharpe_label = metrics.sharpe_label
        result.profit_factor = metrics.profit_factor
        result.max_drawdown_cents = metrics.max_drawdown_cents
        result.avg_edge = metrics.avg_edge
        result.avg_log_return = metrics.avg_log_return
        result.trades = self.tracker.get_trade_history()
        result.equity_curve = self.tracker.get_equity_curve()
        result.feature_importance = self.model.get_feature_importance()

        return result

    def parameter_sweep(
        self,
        settled_data: list[dict],
        entry_thresholds: list[float] | None = None,
        confidence_levels: list[float] | None = None,
    ) -> list[BacktestResult]:
        """Run backtests across a grid of parameter combinations to find optimal settings.

        Tests every combination of entry_threshold x min_confidence values. Each run
        uses a fresh model to avoid state leakage between configurations. Results are
        sorted by Sharpe Ratio (descending) — the guide's primary metric for strategy quality.

        Args:
            settled_data: Historical settled market data (shared across all runs).
            entry_thresholds: List of entry threshold values to test (default: 0.4 to 0.6).
            confidence_levels: List of confidence values to test (default: 0.60 to 0.80).

        Returns:
            List of BacktestResult objects sorted by Sharpe Ratio (best first).
        """
        entry_thresholds = entry_thresholds or [0.4, 0.45, 0.5, 0.55, 0.6]
        confidence_levels = confidence_levels or [0.60, 0.65, 0.70, 0.75, 0.80]

        results = []
        for entry in entry_thresholds:
            for conf in confidence_levels:
                cfg = BacktestConfig(
                    entry_threshold=entry,
                    min_confidence=conf,
                )
                # Reset model for each run
                self.model = PredictionModel(n_estimators=500)
                result = self.run(settled_data, cfg)
                results.append(result)

        # Sort by Sharpe Ratio (the guide's key metric)
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results


# ── Paper Trading Simulator ──────────────────────────────────────────────────

@dataclass
class PaperPosition:
    """A simulated (paper) position with MAE/MFE tracking.

    min_price_seen and max_price_seen are updated on every scan cycle
    to track how far the price moved against/in favor of the position
    before it was closed.
    """
    ticker: str                        # Kalshi market ticker
    side: str                          # "yes" or "no"
    entry_price: float                 # Entry price as decimal (0-1)
    contracts: int                     # Number of simulated contracts
    model_prob: float                  # Model's predicted probability at entry
    entry_time: str                    # ISO timestamp of simulated entry
    category: str = ""                 # Market category for per-category analytics
    min_price_seen: float = 1.0        # Lowest price seen since entry (for MAE calculation)
    max_price_seen: float = 0.0        # Highest price seen since entry (for MFE calculation)


class PaperTrader:
    """
    Paper trading simulator that uses live market data but
    simulates order fills without risking real money.

    Tracks full performance metrics, MAE/MFE, and Sharpe Ratio.
    """

    def __init__(self, db=None):
        """Initialize the paper trader with optional database persistence.

        Args:
            db: Optional Database instance for persisting state and trades to Supabase.
        """
        self.db = db                                       # Optional Supabase persistence
        self.balance_cents: int = 100_00                   # Starting paper balance ($100)
        self.positions: dict[str, PaperPosition] = {}      # Open positions keyed by ticker
        self.tracker = PerformanceTracker(db=db, mode="paper")  # Trade recording + metrics
        self.generator = RFSignalGenerator()               # RF+GB ensemble signal generator
        self.training_store = TrainingDataStore(db=db)      # Cumulative training data
        self.total_scans: int = 0                          # Number of scan cycles completed
        self.signals_seen: int = 0                         # Total signals generated across all scans

    def configure(self, balance_cents: int = 100_00):
        """Set starting balance."""
        self.balance_cents = balance_cents

    def add_funds(self, amount_cents: int) -> int:
        """Add demo funds to paper balance without resetting state."""
        self.balance_cents += amount_cents
        self.save_state()
        return self.balance_cents

    def scan_and_trade(self, events: list[Event]) -> dict:
        """Run a full paper trading scan cycle on live market data.

        Pipeline:
          1. Check all open positions for exit conditions (target hit, expiry, settlement).
          2. Generate new entry signals using the RF+GB ensemble.
          3. For each signal: skip if already positioned, check position limit, verify balance,
             simulate a fill at the ask price, and record the paper position.

        Args:
            events: List of Kalshi events with nested market data.

        Returns:
            Dict with entries (new positions), exits (closed positions), scan number,
            open position count, balance, and total trade count.
        """
        self.total_scans += 1

        # Check exits first
        exits = self._check_exits(events)

        # Generate new signals
        signals = self.generator.generate_signals(events)
        self.signals_seen += len(signals)

        # Execute entries
        entries = []
        for sig in signals:
            if sig.ticker in self.positions:
                continue
            if len(self.positions) >= config.max_open_positions:
                break

            cost = sig.recommended_size_cents
            if cost > self.balance_cents or cost <= 0:
                continue

            # Paper fill at the ask price
            entry_price = sig.market_probability
            contracts = max(1, cost // max(int(entry_price * 100), 1))

            # Find category from the signal's event
            sig_category = ""
            for ev in events:
                for m in ev.markets:
                    if m.ticker == sig.ticker:
                        sig_category = ev.category or m.category or ""
                        break

            self.positions[sig.ticker] = PaperPosition(
                ticker=sig.ticker,
                side=sig.side.value,
                entry_price=entry_price,
                contracts=contracts,
                model_prob=sig.fair_probability,
                entry_time=datetime.now(timezone.utc).isoformat(),
                category=sig_category,
                min_price_seen=entry_price,
                max_price_seen=entry_price,
            )
            self.balance_cents -= int(entry_price * 100 * contracts)
            entries.append({
                "ticker": sig.ticker,
                "side": sig.side.value,
                "entry_price": entry_price,
                "contracts": contracts,
                "model_prob": sig.fair_probability,
                "edge": sig.edge,
            })

        return {
            "scan_number": self.total_scans,
            "entries": entries,
            "exits": exits,
            "open_positions": len(self.positions),
            "balance_cents": self.balance_cents,
            "total_trades": self.tracker.get_metrics().total_trades,
        }

    def _check_exits(self, events: list[Event]) -> list[dict]:
        """Check all open paper positions for exit conditions and close those that trigger.

        Exit conditions (from the guide):
          - Target hit: current_price >= model_prob * 0.9 (90% of fair value reached).
          - Expiry approaching: days_to_expiry <= 7 (close before expiration risk).
          - Market settled: Kalshi resolved the market (definitive win/loss).

        Also updates MAE/MFE tracking for each open position on every check.

        Returns:
            List of exit dicts with ticker, reason, entry/exit prices, and P&L.
        """
        exits = []
        market_map = {}
        for event in events:
            for market in event.markets:
                market_map[market.ticker] = market

        tickers_to_close = []
        for ticker, pos in self.positions.items():
            market = market_map.get(ticker)
            if not market:
                continue

            current_price = market.mid_price_yes / 100

            # Track MAE/MFE
            pos.min_price_seen = min(pos.min_price_seen, current_price)
            pos.max_price_seen = max(pos.max_price_seen, current_price)

            # Exit rules from guide
            hit_target = current_price >= pos.model_prob * 0.9
            settled = market.status == "settled"

            # Time-based: check days to expiry
            days_left = 30
            if market.close_time:
                try:
                    close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
                    days_left = max(0, (close_dt - datetime.now(timezone.utc)).total_seconds() / 86400)
                except (ValueError, TypeError):
                    pass
            hit_expiry = days_left <= 7

            if hit_target or hit_expiry or settled:
                exit_price = current_price
                if settled:
                    exit_price = 1.0 if market.result == pos.side else 0.0

                mae = pos.entry_price - pos.min_price_seen if pos.side == "yes" else pos.max_price_seen - (1 - pos.entry_price)
                mfe = pos.max_price_seen - pos.entry_price if pos.side == "yes" else (1 - pos.entry_price) - pos.min_price_seen

                self.tracker.record_trade(
                    ticker=ticker,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    contracts=pos.contracts,
                    mae=max(0, mae),
                    mfe=max(0, mfe),
                    model_probability=pos.model_prob,
                    market_probability_at_entry=pos.entry_price,
                    entry_time=pos.entry_time,
                    category=pos.category,
                )

                if pos.side == "yes":
                    pnl = (exit_price - pos.entry_price) * 100 * pos.contracts
                    self.balance_cents += int(exit_price * 100 * pos.contracts)
                else:
                    pnl = (pos.entry_price - exit_price) * 100 * pos.contracts
                    self.balance_cents += int((1 - exit_price) * 100 * pos.contracts)

                reason = "Target" if hit_target else ("Expiry" if hit_expiry else "Settled")
                exits.append({
                    "ticker": ticker,
                    "reason": reason,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "pnl_cents": int(pnl),
                })
                tickers_to_close.append(ticker)

        for t in tickers_to_close:
            del self.positions[t]

        return exits

    def get_state(self) -> dict:
        """Get the complete paper trading state for the frontend Dashboard/Testing tabs.

        Returns a dict containing: balance, open positions, performance metrics,
        equity curve, trade history, scan count, signal count, and model training state.
        """
        metrics = self.tracker.get_metrics()
        return {
            "balance_cents": self.balance_cents,
            "open_positions": [
                {
                    "ticker": p.ticker,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "contracts": p.contracts,
                    "model_prob": p.model_prob,
                    "entry_time": p.entry_time,
                }
                for p in self.positions.values()
            ],
            "metrics": {
                "total_trades": metrics.total_trades,
                "wins": metrics.wins,
                "losses": metrics.losses,
                "win_rate": round(metrics.win_rate, 4),
                "total_pnl_cents": metrics.total_pnl_cents,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sharpe_label": metrics.sharpe_label,
                "profit_factor": metrics.profit_factor,
                "max_drawdown_cents": metrics.max_drawdown_cents,
                "avg_mae": metrics.avg_mae,
                "avg_mfe": metrics.avg_mfe,
            },
            "equity_curve": self.tracker.get_equity_curve(),
            "trades": self.tracker.get_trade_history(),
            "total_scans": self.total_scans,
            "signals_seen": self.signals_seen,
            "model_trained": self.generator.model.is_trained,
            "training_samples_count": self.training_store.count,
        }

    def train_model(self, settled_data: list[dict]) -> dict:
        """Train the model using historical data CUMULATIVELY.

        This is the key training method for the paper trader. Unlike a fresh backtest,
        it preserves all previously seen training samples and merges new ones:

          1. Add new settled market samples to TrainingDataStore (deduplicated by ticker).
          2. Train the RF+GB ensemble on ALL accumulated samples (not just the new batch).
          3. Log the training run to Supabase with sample counts and accuracy metrics.

        This cumulative approach means the model gets better over time as more markets
        settle, without ever losing previously learned data.

        Args:
            settled_data: List of dicts with 'market', 'features', and 'outcome' keys.

        Returns:
            Training result dict with: trained, samples, cv_accuracy, oob_score,
            total_cumulative_samples, new_samples_added.
        """
        # 1. Add new samples to the cumulative store (deduped by ticker)
        new_count = self.training_store.add_samples(settled_data)

        # 2. Train on ALL accumulated samples
        all_features, all_outcomes = self.training_store.get_features_and_outcomes()
        total_samples = len(all_features)

        logger.info(
            f"Training on {total_samples} cumulative samples "
            f"({new_count} new, {total_samples - new_count} existing)"
        )

        result = self.generator.model.train_on_historical(all_features, all_outcomes)

        # Add cumulative info to result
        if isinstance(result, dict):
            result["total_cumulative_samples"] = total_samples
            result["new_samples_added"] = new_count
            result["samples"] = total_samples  # Override with true count

        # Persist training run to DB
        if self.db and self.db.is_connected and isinstance(result, dict):
            try:
                self.db.insert_training_run(
                    samples=total_samples,
                    cv_accuracy=result.get("cv_accuracy", 0),
                    oob_score=result.get("oob_score", 0),
                    n_features=result.get("n_features", 106),
                    n_estimators=self.generator.model.n_estimators,
                    feature_importance=self.generator.model.get_feature_importance(),
                )
            except Exception:
                pass

        return result

    # ── Persistence ──────────────────────────────────────────────

    def save_state(self, path: Path | None = None):
        """Save paper trading state to JSON file and DB."""
        # Save to DB if connected
        if self.db and self.db.is_connected:
            try:
                positions_dict = {
                    ticker: {
                        "ticker": p.ticker, "side": p.side,
                        "entry_price": p.entry_price, "contracts": p.contracts,
                        "model_prob": p.model_prob, "entry_time": p.entry_time,
                        "min_price_seen": p.min_price_seen, "max_price_seen": p.max_price_seen,
                    }
                    for ticker, p in self.positions.items()
                }
                self.db.save_paper_state(
                    self.balance_cents, self.total_scans, self.signals_seen, positions_dict
                )
            except Exception:
                pass

        # Also save to JSON file as backup
        path = path or DATA_DIR / "paper_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "balance_cents": self.balance_cents,
            "total_scans": self.total_scans,
            "signals_seen": self.signals_seen,
            "positions": {
                ticker: {
                    "ticker": p.ticker,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "contracts": p.contracts,
                    "model_prob": p.model_prob,
                    "entry_time": p.entry_time,
                    "min_price_seen": p.min_price_seen,
                    "max_price_seen": p.max_price_seen,
                }
                for ticker, p in self.positions.items()
            },
            "trades": self.tracker.get_trade_history(),
            "equity_curve": self.tracker.get_equity_curve(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(state, indent=2, default=str))

    def load_state(self, path: Path | None = None) -> bool:
        """Load paper trading state from DB or JSON file. Returns True if loaded."""
        # Try DB first
        if self.db and self.db.is_connected:
            try:
                state = self.db.load_paper_state()
                if state:
                    self.balance_cents = state.get("balance_cents", 100_00)
                    self.total_scans = state.get("total_scans", 0)
                    self.signals_seen = state.get("signals_seen", 0)
                    self.positions = {}
                    for ticker, p in (state.get("positions") or {}).items():
                        self.positions[ticker] = PaperPosition(
                            ticker=p["ticker"], side=p["side"],
                            entry_price=p["entry_price"], contracts=p["contracts"],
                            model_prob=p["model_prob"], entry_time=p["entry_time"],
                            min_price_seen=p.get("min_price_seen", 1.0),
                            max_price_seen=p.get("max_price_seen", 0.0),
                        )
                    return True
            except Exception:
                pass

        # Fall back to JSON file
        path = path or DATA_DIR / "paper_state.json"
        if not path.exists():
            return False

        try:
            state = json.loads(path.read_text())
            self.balance_cents = state.get("balance_cents", 100_00)
            self.total_scans = state.get("total_scans", 0)
            self.signals_seen = state.get("signals_seen", 0)

            # Restore positions
            self.positions = {}
            for ticker, p in state.get("positions", {}).items():
                self.positions[ticker] = PaperPosition(
                    ticker=p["ticker"],
                    side=p["side"],
                    entry_price=p["entry_price"],
                    contracts=p["contracts"],
                    model_prob=p["model_prob"],
                    entry_time=p["entry_time"],
                    min_price_seen=p.get("min_price_seen", 1.0),
                    max_price_seen=p.get("max_price_seen", 0.0),
                )

            # Restore trade history
            self.tracker = PerformanceTracker()
            for t in state.get("trades", []):
                self.tracker.record_trade(
                    ticker=t["ticker"],
                    side=t["side"],
                    entry_price=t["entry_price"],
                    exit_price=t["exit_price"],
                    contracts=t.get("contracts", 1),
                    mae=t.get("mae", 0),
                    mfe=t.get("mfe", 0),
                    model_probability=t.get("model_probability", 0),
                    market_probability_at_entry=t.get("market_probability_at_entry", 0),
                    entry_time=t.get("entry_time", ""),
                    exit_time=t.get("exit_time", ""),
                )
            return True
        except (json.JSONDecodeError, KeyError):
            return False
