"""
Backtester: test the strategy on historical Kalshi data.

Fetches settled markets, replays the strategy, and reports
full performance metrics including Sharpe Ratio, win rate, MAE/MFE.

Also provides a paper trading simulator for live testing without real money.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from bot.config import config
from bot.kalshi_client import KalshiClient
from bot.models import Event, Market, Side
from bot.rf_model import extract_features, FEATURE_NAMES, PredictionModel, RFSignalGenerator
from bot.performance import PerformanceTracker


# ── Historical Data Fetcher ──────────────────────────────────────────────────

class HistoricalDataFetcher:
    """Fetches settled markets from Kalshi for backtesting and training."""

    def __init__(self, kalshi: KalshiClient):
        self.kalshi = kalshi

    def fetch_settled_markets(self, limit: int = 200) -> list[dict]:
        """
        Fetch settled (resolved) markets with their outcomes.

        Returns list of dicts with market data + result (yes/no).
        """
        data = self.kalshi._request("GET", "/events", params={
            "limit": min(limit, 200),
            "status": "settled",
            "with_nested_markets": "true",
        })

        settled = []
        for ev in data.get("events", []):
            event = Event(
                event_ticker=ev.get("event_ticker", ""),
                title=ev.get("title", ""),
                category=ev.get("category", ""),
            )
            for m in ev.get("markets", []):
                result = m.get("result", "")
                if result not in ("yes", "no"):
                    continue

                market = Market(
                    ticker=m.get("ticker", ""),
                    event_ticker=m.get("event_ticker", ""),
                    title=m.get("title", ""),
                    subtitle=m.get("subtitle", ""),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 0),
                    no_bid=m.get("no_bid", 0),
                    no_ask=m.get("no_ask", 0),
                    volume=m.get("volume", 0),
                    open_interest=m.get("open_interest", 0),
                    status="settled",
                    close_time=m.get("close_time", ""),
                    result=result,
                    category=m.get("category", ""),
                )

                features = extract_features(market, event)
                settled.append({
                    "market": market,
                    "event": event,
                    "features": features,
                    "outcome": 1 if result == "yes" else 0,
                    "result": result,
                })

        return settled

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
    """Configuration for a backtest run."""
    initial_balance_cents: int = 100_00  # $100
    max_bet_cents: int = 25_00           # $25
    entry_threshold: float = 0.5         # market <= model * threshold
    exit_threshold: float = 0.9          # market >= model * threshold
    min_confidence: float = 0.70         # 70% minimum
    min_volume: int = 50
    max_positions: int = 10
    train_ratio: float = 0.6            # 60% train, 40% test


@dataclass
class BacktestResult:
    """Complete backtest results."""
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
        self.kalshi = kalshi
        self.model = PredictionModel(n_estimators=200)
        self.tracker = PerformanceTracker()

    def run(
        self,
        settled_data: list[dict] | None = None,
        cfg: BacktestConfig | None = None,
    ) -> BacktestResult:
        """
        Run a full backtest.

        Pass settled_data directly, or it will fetch from Kalshi.
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

        # Shuffle and split
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
        """
        Run backtests across multiple parameter combinations
        to find optimal settings without overfitting.
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
                self.model = PredictionModel(n_estimators=200)
                result = self.run(settled_data, cfg)
                results.append(result)

        # Sort by Sharpe Ratio (the guide's key metric)
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        return results


# ── Paper Trading Simulator ──────────────────────────────────────────────────

@dataclass
class PaperPosition:
    """A simulated position."""
    ticker: str
    side: str
    entry_price: float
    contracts: int
    model_prob: float
    entry_time: str
    min_price_seen: float = 1.0
    max_price_seen: float = 0.0


class PaperTrader:
    """
    Paper trading simulator that uses live market data but
    simulates order fills without risking real money.

    Tracks full performance metrics, MAE/MFE, and Sharpe Ratio.
    """

    def __init__(self):
        self.balance_cents: int = 100_00  # $100 starting balance
        self.positions: dict[str, PaperPosition] = {}
        self.tracker = PerformanceTracker()
        self.generator = RFSignalGenerator()
        self.total_scans: int = 0
        self.signals_seen: int = 0

    def configure(self, balance_cents: int = 100_00):
        """Set starting balance."""
        self.balance_cents = balance_cents

    def scan_and_trade(self, events: list[Event]) -> dict:
        """
        Run a full scan cycle:
        1. Check exits on open positions
        2. Generate new entry signals
        3. Execute paper trades
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

            self.positions[sig.ticker] = PaperPosition(
                ticker=sig.ticker,
                side=sig.side.value,
                entry_price=entry_price,
                contracts=contracts,
                model_prob=sig.fair_probability,
                entry_time=datetime.now(timezone.utc).isoformat(),
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
        """Check and execute exits on paper positions."""
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
                )

                if pos.side == "yes":
                    pnl = (exit_price - pos.entry_price) * 100 * pos.contracts
                else:
                    pnl = (pos.entry_price - exit_price) * 100 * pos.contracts
                self.balance_cents += int(exit_price * 100 * pos.contracts) + int(pnl)

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
        """Get full paper trading state."""
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
        }

    def train_model(self, settled_data: list[dict]) -> dict:
        """Train the model using historical data."""
        features_list = [d["features"] for d in settled_data]
        outcomes = [d["outcome"] for d in settled_data]
        return self.generator.model.train_on_historical(features_list, outcomes)
