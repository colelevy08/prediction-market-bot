"""
Historical backtester, cumulative training data store, and paper trading simulator.

═══════════════════════════════════════════════════════════════════════════════
  CHAPTER 1: WHY WE TEST BEFORE TRADING WITH REAL MONEY
═══════════════════════════════════════════════════════════════════════════════

Imagine you invented a card-counting system for poker. Would you immediately sit
down at a high-stakes table and bet your life savings? Of course not — you would
first practice at home with fake chips, review thousands of past hands, and only
deploy real money once you're confident the system actually works.

That's exactly what this file does for our trading bot.

──────────────────────────────────────────────────────────────────────────────
BACKTESTING — Testing on history

A "backtest" replays your strategy on historical market data that has already
resolved (we know who won and who lost). We pretend we're standing in the past
and ask: "If I had used this strategy back then, would I have made money?"

Key advantages:
  - You can test years of market scenarios in seconds.
  - You can try hundreds of parameter combinations without risking a cent.
  - You can see exactly where the strategy would have succeeded or failed.

Key DANGERS (always keep these in mind):
  1. OVERFITTING: If you keep tweaking the strategy until it looks good on past
     data, you may have just memorized history rather than found a real edge.
     The strategy will appear to work perfectly on the data you tested, but
     fail completely on new data.
     Example: "I found that buying on Tuesdays in March works great!" — this
     is probably just noise in the historical data, not a real pattern.

  2. LOOK-AHEAD BIAS: Accidentally using information that wasn't available at
     the time of the decision. E.g., if you "enter a trade" using the closing
     price of a candle, but in real life you'd only know that price after the
     candle closes. This code carefully sorts data chronologically and uses
     only the information that would have been available at entry time.

  3. SURVIVORSHIP BIAS: Only analyzing markets that still exist or succeeded.
     If you only look at markets that had lots of volume, you're ignoring all
     the low-volume markets that might have behaved differently (or been traps).

──────────────────────────────────────────────────────────────────────────────
PAPER TRADING — Simulated live trading

"Paper trading" means running the full live strategy — looking at real current
prices, generating real signals — but NOT actually sending orders. Instead, we
pretend to execute the trade and track a fake account balance.

This is the bridge between backtesting and real trading. It answers the question:
"Does this strategy work on TODAY'S live data, with real spreads and timing?"

Paper trading catches issues that backtesting misses:
  - API delays (real data arrives slightly late)
  - Spread reality (the price you see isn't always the price you get)
  - Simultaneous position conflicts (can't always enter at the ideal moment)

──────────────────────────────────────────────────────────────────────────────
SLIPPAGE — The gap between what you expect and what you get

When you place an order in any market, there's often a small difference between
the price you thought you'd pay and the price you actually pay. This is called
"slippage." It happens because:
  - Other traders are also submitting orders at the same time.
  - The bid-ask spread means you always pay slightly more than the "fair" price.
  - By the time your order reaches the exchange, the market has moved slightly.

In this code, slippage is modeled as a fixed 1 cent (1¢) penalty per trade.
Over hundreds of trades, this adds up — a strategy that ignores slippage might
look profitable but actually lose money in practice.

──────────────────────────────────────────────────────────────────────────────
BID-ASK SPREAD — The invisible cost of every trade

Every market has two prices:
  - The BID:  the price a buyer is willing to pay.
  - The ASK:  the price a seller is willing to accept.
  - The MID:  the midpoint between them (the "fair" price).

The difference between bid and ask is called the "spread." When you buy, you pay
the ask (above mid). When you sell, you receive the bid (below mid). You
immediately start "underwater" by half the spread just by entering the trade.

For a Kalshi prediction market trading at 55¢ bid / 57¢ ask:
  - You'd pay 57¢ to enter.
  - You'd only get 55¢ if you had to exit immediately.
  - That's a 2¢ round-trip cost before you've done anything.

This code reconstructs realistic bid/ask spreads from trade history, rather
than using the post-settlement price of 0 or 100. That's critical — training
a model on "the price was 0 when it lost" is useless for learning entry signals.

──────────────────────────────────────────────────────────────────────────────
FILLS — When your order actually executes

A "fill" means your order was matched with a counterparty and executed. In liquid
markets (lots of buyers and sellers), you get filled quickly near the price you
wanted. In illiquid markets (few participants), you may not get filled at all, or
only at a much worse price.

For backtesting prediction markets, this matters because:
  - Low-volume markets may only show a theoretical price — no one may actually
    trade with you at that price.
  - This code filters out markets with fewer than 10 contracts traded for
    exactly this reason: if nobody trades there, our "profit" is theoretical.

──────────────────────────────────────────────────────────────────────────────
THIS MODULE CONTAINS FOUR MAJOR COMPONENTS:

1. TrainingDataStore:
   - Persists training samples (106-feature vectors + binary outcomes) to JSON + Supabase.
   - Samples are deduplicated by market ticker so each settled market is stored once.
   - Every training run builds cumulatively on all previously seen data rather than
     discarding old samples, so the model improves monotonically with more data.
   - Primary storage: Supabase table 'training_samples' (batch upsert, 500-row chunks).
   - Local cache: JSON file (data/training_samples.json) — ephemeral on Railway.

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

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

    ─────────────────────────────────────────────────────────────────────────
    WHY CUMULATIVE TRAINING MATTERS

    Most simple ML pipelines throw away old training data every time you
    retrain — they start fresh. This store takes the opposite approach:
    every settled market we've ever seen is kept permanently.

    Analogy: Imagine you're learning to predict weather. Would you forget all
    of last year's weather data before making tomorrow's forecast? Of course
    not. The more historical examples you've seen, the better your predictions.

    As more Kalshi markets settle over days and weeks, this store grows, and
    the model's accuracy improves continuously.

    ─────────────────────────────────────────────────────────────────────────
    WHAT IS A "TRAINING SAMPLE"?

    Each sample is a pair of:
      - features: a list of ~106 numbers describing one market at entry time
        (price, volume, volatility, spread, time-of-day, etc.)
      - outcome: 0 (market resolved NO) or 1 (market resolved YES)

    The model learns which feature patterns lead to YES outcomes vs NO outcomes.

    ─────────────────────────────────────────────────────────────────────────
    DEDUPLICATION

    Markets have unique "tickers" (like "KXBTC-15M-25000"). We use the ticker
    as the dictionary key so if we accidentally fetch the same settled market
    twice, we don't train on it twice. Training on duplicates would cause the
    model to "overweight" those examples — treating one market as if it were
    two — which distorts what the model learns.
    """

    def __init__(self, path: Path | None = None, db=None):
        self.path = path or DATA_DIR / "training_samples.json"
        self.db = db
        self.samples: dict[str, dict] = {}  # keyed by ticker for dedup
        self._load()

    def _load(self):
        """Load existing samples from DB (primary) or disk (fallback cache).

        This is called automatically when the store is created. It follows a
        two-level hierarchy:
          1. Try Supabase (the cloud database) first — this is the "source of
             truth" because it persists even if the server restarts or redeploys.
          2. Fall back to a local JSON file on disk — faster to read but will
             be lost if the server is restarted (Railway ephemeral filesystem).

        This pattern (cloud-primary, local-fallback) is common in production
        systems. It's like how your phone syncs contacts to iCloud: the cloud
        copy is authoritative, but the local copy lets you work offline.
        """
        _load_start = time.time()  # Fix 8: Track load time
        # Try DB first — authoritative source for cloud deployments
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
                    self._save_to_disk()  # Cache locally for faster restarts
                    return
            except Exception as e:
                logger.warning(f"TrainingDataStore: failed to load from DB: {e}")

        # Fallback to local JSON cache (useful during development)
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for s in data.get("samples", []):
                    ticker = s.get("ticker", "")
                    if ticker:
                        self.samples[ticker] = s
                logger.info(f"TrainingDataStore: loaded {len(self.samples)} samples from disk cache")
            except FileNotFoundError:
                logger.info("TrainingDataStore: no local cache file found")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"TrainingDataStore: failed to load from disk: {e}")

        # Fix 8: Log total load time
        _load_duration = time.time() - _load_start
        logger.info(f"TrainingDataStore: load completed in {_load_duration:.2f}s ({len(self.samples)} samples)")

    def add_samples(self, settled_data: list[dict]) -> dict:
        """
        Merge new training samples into the store, deduplicating by ticker.
        Checks for outlier feature values (beyond 3 standard deviations of existing data).

        Returns a dict with 'new_count' (number of new samples added) and
        'outlier_count' (number of samples with outlier features).

        ─────────────────────────────────────────────────────────────────────
        WHAT IS AN OUTLIER?

        An "outlier" is a data point that is extremely unusual — far outside
        the normal range. Outliers can distort what the model learns.

        We measure this using "standard deviations" (a measure of spread):
          - 68% of normal data falls within 1 standard deviation of the mean.
          - 95% falls within 2 standard deviations.
          - 99.7% falls within 3 standard deviations.

        If a feature value is more than 3 standard deviations from the mean,
        it's an outlier. We flag it but still keep it — we don't want to silently
        discard data, but we want to know when unusual samples come in.

        Example: If the average market volume is 500 contracts and one market
        had 50,000 contracts, that's an outlier. The model would need to learn
        whether that extreme volume is meaningful or just noise.
        """
        new_added = []
        now = datetime.now(timezone.utc).isoformat()
        outlier_count = 0

        # Compute feature statistics from existing samples for outlier detection
        # We're calculating the mean (average) and standard deviation for each
        # feature across all samples we've already stored.
        feature_stats = {}
        if self.samples:
            existing_features = [s["features"] for s in self.samples.values()]
            for feature_name in (list(existing_features[0].keys()) if existing_features else []):
                values = [f.get(feature_name, 0) for f in existing_features if isinstance(f.get(feature_name, 0), (int, float))]
                if len(values) >= 10:
                    mean = sum(values) / len(values)
                    variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
                    std = variance ** 0.5
                    if std > 0:
                        feature_stats[feature_name] = (mean, std)

        for item in settled_data:
            market = item.get("market")
            ticker = market.ticker if hasattr(market, "ticker") else item.get("ticker", "")
            if not ticker or ticker in self.samples:
                continue

            # Fix 5: Validate that features dict is not empty
            features = item.get("features", {})
            if not features:
                logger.debug(f"TrainingDataStore: skipping {ticker} — empty features dict")
                continue

            # Fix 6: Validate outcome is 0 or 1
            outcome = item.get("outcome", 0)
            if outcome not in (0, 1):
                logger.debug(f"TrainingDataStore: skipping {ticker} — invalid outcome {outcome}")
                continue
            is_outlier = False
            if feature_stats:
                for feat_name, (mean, std) in feature_stats.items():
                    val = features.get(feat_name, 0)
                    if isinstance(val, (int, float)) and abs(val - mean) > 3 * std:
                        is_outlier = True
                        break

            if is_outlier:
                outlier_count += 1
                logger.debug(f"TrainingDataStore: outlier sample flagged for {ticker}")
                # Flag but don't remove — still add to the store

            sample = {
                "ticker": ticker,
                "features": features,
                "outcome": outcome,
                "fetched_at": now,
                "is_outlier": is_outlier,
            }
            self.samples[ticker] = sample
            new_added.append(sample)

        if new_added:
            self._save_to_disk()
            self._save_to_db(new_added)
            logger.info(f"TrainingDataStore: added {len(new_added)} new samples (total: {len(self.samples)}), {outlier_count} outliers flagged")

        return {"new_count": len(new_added), "outlier_count": outlier_count}

    def get_all_samples(self) -> list[dict]:
        """Return all stored samples as a list of {features, outcome} dicts."""
        return list(self.samples.values())

    def get_features_and_outcomes(self) -> tuple[list[dict], list[int]]:
        """Return features list and outcomes list ready for model training.

        The ML model (Random Forest + Gradient Boosting) needs data in two
        parallel lists:
          - features: [[f1, f2, ...], [f1, f2, ...], ...]  — one list per market
          - outcomes: [0, 1, 1, 0, ...]                    — one 0/1 per market

        Think of it like two columns in a spreadsheet:
          Column A = "all the facts about this market at entry time"
          Column B = "did it resolve YES (1) or NO (0)?"

        The model tries to find which combinations of Column A facts reliably
        predict whether Column B will be 0 or 1.

        This method also checks that all samples have the same feature keys.
        If the code was updated to track new features, old samples might be
        missing those keys — this logs a warning so we notice the mismatch.
        """
        features = []
        outcomes = []
        # Fix 9: Validate consistent feature keys across all samples
        reference_keys = None
        for s in self.samples.values():
            feat = s["features"]
            if reference_keys is None:
                reference_keys = set(feat.keys())
            elif set(feat.keys()) != reference_keys:
                # Log mismatch but include sample anyway (model handles missing features)
                missing = reference_keys - set(feat.keys())
                extra = set(feat.keys()) - reference_keys
                if missing or extra:
                    logger.debug(f"Feature key mismatch for {s.get('ticker', '?')}: missing={missing}, extra={extra}")
            features.append(feat)
            outcomes.append(s["outcome"])
        return features, outcomes

    @property
    def count(self) -> int:
        return len(self.samples)

    def _save_to_disk(self):
        """Cache all samples to local JSON (best-effort, ephemeral on Railway)."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "total_count": len(self.samples),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "samples": list(self.samples.values()),
            }
            self.path.write_text(json.dumps(data, default=str))
        except OSError:
            pass  # Local cache is best-effort — DB is authoritative

    def _save_to_db(self, new_samples: list[dict]):
        """Batch-insert new samples to Supabase (best-effort).

        Expects clean sample dicts with ticker, features, outcome keys.
        """
        if not self.db or not self.db.is_connected:
            return
        try:
            self.db.insert_training_samples([
                {
                    "ticker": s["ticker"],
                    "features": s.get("features", {}),
                    "outcome": s.get("outcome", 0),
                }
                for s in new_samples
            ])
            # Fix 7: Log success with count
            logger.info(f"TrainingDataStore: saved {len(new_samples)} samples to DB")
        except Exception as e:
            logger.warning(f"TrainingDataStore: DB save failed: {e}")


# ── Historical Data Fetcher ──────────────────────────────────────────────────

class HistoricalDataFetcher:
    """Fetches settled markets from Kalshi for backtesting and training.

    ─────────────────────────────────────────────────────────────────────────
    THE CORE CHALLENGE: RECONSTRUCTING PRE-SETTLEMENT PRICES

    After a Kalshi market settles, the YES price becomes either $1.00 (if YES
    won) or $0.00 (if NO won). This is the final settled price — useless for
    training a model, because the model doesn't trade at settlement, it trades
    while the market is still open.

    Imagine training someone to predict sports outcomes by only showing them
    the final score. They'd learn nothing about how to read the game in progress.

    This class fetches the last 50 actual trades that occurred before settlement,
    then reconstructs what the bid and ask prices would have looked like during
    the trading period. The model then trains on those realistic "in-game" prices.

    ─────────────────────────────────────────────────────────────────────────
    PAGINATION

    Kalshi's API returns markets in pages (like a search engine showing 200
    results per page). To get 1,000 settled markets, you need to request page 1,
    then use the "cursor" (a bookmark) from that response to request page 2, etc.

    This fetcher handles pagination automatically, stopping when it has enough
    samples or runs out of pages.
    """

    def __init__(self, kalshi: KalshiClient):
        self.kalshi = kalshi

    def fetch_settled_markets(self, limit: int = 200, series_tickers: list[str] | None = None) -> list[dict]:
        """
        Fetch settled (resolved) markets with their outcomes.
        Reconstructs pre-settlement features from trade history so the model
        can learn from realistic price/volume data instead of post-settlement zeros.
        Paginates through multiple pages to get up to `limit` samples.

        Args:
            limit: Maximum number of samples to return.
            series_tickers: If provided, fetch only markets from these series
                (e.g. ["KXBTC15M", "KXETH15M"]). Makes one paginated pass per
                series so the training set is focused on the markets we actually
                trade.  When None, fetches all settled events (original behaviour).

        For each settled market, this method:
          1. Fetches the last 50 trades to reconstruct a realistic entry price.
          2. Computes a synthetic bid/ask spread from the standard deviation of
             those trades (more volatile trades → wider spread).
          3. Skips markets with no price data or very low volume (< 10 contracts),
             because you likely couldn't have actually traded those markets.
          4. Builds a Market object and extracts features (same 106-number vector
             used by the live bot), then returns them paired with the outcome.

        The 'limit' parameter is a safety cap, not a request size. The method may
        return fewer samples if not enough qualifying markets are found.
        """
        # When series_tickers is given we run one fetch-loop per series so we
        # get a balanced, crypto-focused dataset.  Otherwise fall back to the
        # original single-loop behaviour that fetches all event types.
        if series_tickers:
            all_settled: list[dict] = []
            per_series = max(50, limit // len(series_tickers))
            for series in series_tickers:
                all_settled.extend(
                    self._fetch_settled_for_series(series, per_series)
                )
            logger.info(
                f"fetch_settled_markets (series={series_tickers}): "
                f"{len(all_settled)} total samples"
            )
            return all_settled[:limit]

        settled = []
        cursor = None
        page_size = min(limit, 200)
        _fetch_start = time.time()  # Fix 4: Track total duration

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

                    # Fix 2: Skip markets with volume < 10 (too illiquid)
                    if isinstance(volume, (int, float)) and volume < 10:
                        continue

                    last_price = _dollars_to_cents(m.get("last_price_dollars")) or 0
                    prev_price = _dollars_to_cents(m.get("previous_price_dollars")) or 0

                    # ── Reconstruct pre-settlement prices from trade history ──────
                    # We fetch the last 50 actual trades that happened in this market
                    # before it settled. These give us real prices that traders paid,
                    # which is far more useful than the final 0/100 settlement price.
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

                    # ── Reconstruct a realistic bid/ask spread ───────────────────
                    # The spread is estimated from how much the trade prices varied.
                    # np.std() is the "standard deviation" — a measure of how spread
                    # out the prices were. More variation → wider spread → less liquid
                    # market → harder to get a clean fill.
                    #
                    # Fallback chain (from best to worst data quality):
                    #   1. Use real trade history (most accurate)
                    #   2. Use the last known price + a fixed 4¢ spread
                    #   3. Use the price from the previous period + 6¢ spread
                    #   4. Skip the market entirely (no price data at all)
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
                        # Skip markets with no price data — we can't train on nothing
                        continue

                    yes_bid = max(1, avg_price - spread // 2)
                    yes_ask = min(99, avg_price + spread // 2)

                    # Fix 3: Skip markets with yes_bid == yes_ask (no spread = no opportunity)
                    if yes_bid == yes_ask:
                        continue

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

                    # Fix 1: Log progress every 500 markets fetched
                    if len(settled) % 500 == 0:
                        logger.info(f"fetch_settled_markets: {len(settled)} markets fetched so far...")

            # Check for next page cursor
            cursor = data.get("cursor")
            if not cursor:
                break

        # Fix 4: Log total duration
        _fetch_duration = time.time() - _fetch_start
        logger.info(f"fetch_settled_markets: completed {len(settled)} markets in {_fetch_duration:.1f}s")
        return settled[:limit]

    def _fetch_settled_for_series(self, series_ticker: str, limit: int = 500) -> list[dict]:
        """Fetch settled markets for a single Kalshi series (e.g. 'KXBTC15M')."""
        settled = []
        cursor = None
        page_size = min(200, limit)
        _fetch_start = time.time()

        while len(settled) < limit:
            params = {
                "limit": page_size,
                "status": "settled",
                "with_nested_markets": "true",
                "series_ticker": series_ticker,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                data = self.kalshi._request("GET", "/events", params=params)
            except Exception as e:
                logger.warning(f"_fetch_settled_for_series({series_ticker}): API error {e}")
                break

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
                    if isinstance(volume, (int, float)) and volume < 10:
                        continue
                    last_price = _dollars_to_cents(m.get("last_price_dollars")) or 0
                    prev_price = _dollars_to_cents(m.get("previous_price_dollars")) or 0
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
                        continue
                    yes_bid = max(1, avg_price - spread // 2)
                    yes_ask = min(99, avg_price + spread // 2)
                    if yes_bid == yes_ask:
                        continue
                    no_bid = max(1, 100 - yes_ask)
                    no_ask = min(99, 100 - yes_bid)
                    market = Market(
                        ticker=ticker,
                        event_ticker=m.get("event_ticker", ""),
                        title=m.get("title", ""),
                        subtitle=m.get("subtitle", ""),
                        yes_bid=yes_bid, yes_ask=yes_ask,
                        no_bid=no_bid, no_ask=no_ask,
                        volume=int(volume) if isinstance(volume, (int, float)) else 0,
                        open_interest=_parse_fp(m.get("open_interest_fp")) or m.get("open_interest", 0) or 0,
                        status="settled",
                        close_time=m.get("close_time", ""),
                        result=result,
                        category=m.get("category", ""),
                        last_price=last_price,
                        prev_price=prev_price,
                    )
                    history = [{"yes_mid": p, "volume": volume} for p in trade_prices] if trade_prices else None
                    features = extract_features(market, event, history)
                    settled.append({
                        "market": market, "event": event, "features": features,
                        "outcome": 1 if result == "yes" else 0,
                        "result": result, "n_trades": len(trade_prices),
                    })
                    if len(settled) >= limit:
                        break
                if len(settled) >= limit:
                    break

            cursor = data.get("cursor")
            if not cursor:
                break

        _dur = time.time() - _fetch_start
        logger.info(f"_fetch_settled_for_series({series_ticker}): {len(settled)} samples in {_dur:.1f}s")
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
    """Configuration parameters for a single backtest run.

    These mirror the guide's recommended settings and can be swept via
    Backtester.parameter_sweep() to find optimal values.

    ─────────────────────────────────────────────────────────────────────────
    A @dataclass is Python shorthand for a simple data-holding class. Instead
    of writing an __init__ method by hand, Python auto-generates one from the
    field definitions below. Each field has a default value, so you can create
    a BacktestConfig() with defaults and override only what you need.

    Think of this like a settings form where every field is pre-filled.

    ─────────────────────────────────────────────────────────────────────────
    ABOUT entry_threshold AND min_confidence

    The bot only enters a trade when TWO conditions are both true:

      1. The model is CONFIDENT enough (min_confidence = 0.65 means the model
         is at least 65% sure the market will go one direction).

      2. There's enough EDGE — the market price is cheap relative to what the
         model thinks the fair probability is (entry_threshold controls this).

    These two knobs prevent trading on weak or borderline signals. The
    parameter_sweep() function tests many combinations to find which values
    maximize the Sharpe Ratio on historical data.
    """
    initial_balance_cents: int = 100_00  # Starting paper balance ($100)
    max_bet_cents: int = 25_00           # Maximum bet per trade ($25)
    entry_threshold: float = 0.93         # Entry rule: market_price <= model_prob * threshold (~7% undervalued)
    exit_threshold: float = 0.9          # Exit rule: market_price >= model_prob * threshold
    min_confidence: float = 0.65         # Minimum model confidence to enter
    min_volume: int = 50                 # Skip markets with < 50 contracts traded
    max_positions: int = 10              # Max simultaneous open positions
    train_ratio: float = 0.6            # Train/test split (60% train, 40% test)
    slippage_cents: int = 1              # Fix 11: Slippage per trade in cents (default 1c)
    commission_cents: int = 0            # Fix 13: Commission per trade in cents (default 0)


@dataclass
class BacktestResult:
    """Complete backtest results including training metrics, trading metrics, and details.

    ─────────────────────────────────────────────────────────────────────────
    GLOSSARY OF METRICS

    These are the numbers that tell you whether a strategy is actually good.
    Think of them as the "report card" for a backtest.

    cv_accuracy:
        Cross-validation accuracy. During training, we hold out a small slice
        of training data as a "mini test" to check accuracy. If CV accuracy is
        70%, the model correctly predicted 70% of those held-out examples.

    oob_score:
        "Out-of-bag" score, a Random Forest specialty. When building each tree
        in the forest, some training samples are randomly left out ("out of bag").
        Those left-out samples are used to estimate accuracy without a separate
        test set. It's a second opinion on model quality.

    sharpe_ratio:
        The gold standard metric for strategy quality. It answers:
        "How much return do I get per unit of risk?"
        Formula: average_daily_return / std_deviation_of_daily_returns × √365

        - Above 1.0: Decent. The strategy makes consistent positive returns.
        - Above 2.0: Very good. Used by professional fund managers as a target.
        - Above 3.0: Excellent. Hard to achieve consistently.
        - Below 0:   The strategy loses money.

        WHY NOT JUST USE TOTAL PROFIT?
        Because a strategy that makes $1,000 steadily over 100 trades is far
        better than one that swings between +$5,000 and -$4,000 before ending
        at +$1,000. The Sharpe Ratio captures this stability vs. volatility.

    profit_factor:
        Total gross profit / total gross loss.
        - Above 1.0: Makes more than it loses (profitable overall).
        - 2.0 means: for every $1 lost, the strategy earns $2. Very good.
        - Below 1.0: Losing strategy.

    max_drawdown_cents:
        The worst peak-to-trough loss during the test period. If the account
        grew to $1,000, then fell to $700, the drawdown is $300.
        This answers: "What's the worst losing streak I'd have had to survive?"

    avg_edge:
        The average difference between the model's fair probability and the
        market price at entry. Edge = model_prob - market_price.
        If the model says 70% chance but the market price is 55¢, edge = 15%.
        Positive edge means you're buying cheaper than fair value, on average.

    time_in_market_pct:
        What % of calendar days had at least one open position. Low values
        mean the strategy is selective (good). High values mean you're always
        in a trade (more exposure to market swings).

    equity_curve:
        A list of the account balance after every trade. Plotting this as a
        line chart shows whether the account grew smoothly, erratically, or
        declined. A smooth upward slope is what we want.
    """
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
    # Real-time duration metrics
    calendar_duration_days: int = 0
    avg_hold_time_hours: float = 0.0
    trades_per_day: float = 0.0
    annualized_return_pct: float = 0.0
    annualized_sharpe: float = 0.0
    time_in_market_pct: float = 0.0
    # Details
    trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)
    signals_generated: int = 0
    signals_filtered: int = 0
    best_trade_pnl: int = 0          # Fix 17: Best individual trade P&L
    worst_trade_pnl: int = 0         # Fix 17: Worst individual trade P&L
    avg_trade_duration_hours: float = 0.0  # Fix 18: Average trade duration


class Backtester:
    """
    Runs the full strategy on historical data.

    1. Fetches settled markets from Kalshi
    2. Splits into train/test sets
    3. Trains the ensemble model on train set
    4. Replays the strategy on test set
    5. Computes full performance metrics

    ─────────────────────────────────────────────────────────────────────────
    THE TRAIN/TEST SPLIT — Preventing Look-Ahead Bias

    We cannot train the model on the same data we use to test it. If we did,
    the model would simply "memorize" the answers — like giving a student the
    exam answers during the study session and then using the same exam.

    Instead, we split the historical data chronologically:
      - First 60% of markets → TRAINING SET (the model learns from these)
      - Last 40% of markets  → TEST SET (strategy is replayed on these, with
                                 the model having never seen them)

    Sorting by close_time BEFORE splitting is critical. If we split randomly,
    the model might see markets from December during training, then be tested on
    markets from June — effectively having "future knowledge" of what happened
    in June when it learned in December. This is look-ahead bias.

    ─────────────────────────────────────────────────────────────────────────
    THE STRATEGY REPLAY

    During the test phase, for each historical market, we:
      1. Ask the model: "What probability does this market resolve YES?"
      2. Check: Is the model confident enough? (>= min_confidence)
      3. Check: Is there enough edge? (market price cheap vs. model's estimate)
      4. If both pass, "enter" the trade (subtract cost from balance).
      5. Apply the actual historical outcome to determine profit/loss.

    This replay is strictly sequential — no peeking ahead in the test data.
    """

    def __init__(self, kalshi: KalshiClient | None = None):
        """Initialize the backtester with an optional Kalshi client for data fetching.

        Args:
            kalshi: If provided, used to fetch settled markets. If None, settled_data
                    must be passed directly to run().
        """
        self.kalshi = kalshi
        # Fix 10: Validate kalshi client is connected if provided
        if kalshi is not None and not getattr(kalshi, 'session', None):
            logger.warning("Backtester: Kalshi client provided but may not be connected")
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
        # Fix 15: Log start of backtest with config summary
        logger.info(
            f"Backtest starting: entry_thresh={cfg.entry_threshold}, confidence={cfg.min_confidence}, "
            f"balance={cfg.initial_balance_cents}c, slippage={cfg.slippage_cents}c, commission={cfg.commission_cents}c"
        )
        _backtest_start = time.time()
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

        if len(settled_data) < 5:
            result.config["error"] = f"Only {len(settled_data)} settled markets found. Need 5+."
            return result

        if len(settled_data) < 50:
            result.config["warning"] = f"Only {len(settled_data)} settled markets found. Results may be unreliable (recommend 50+)."

        # Sort by close_time to avoid look-ahead bias, then split chronologically.
        # "Look-ahead bias" means accidentally using future information while making
        # a past decision. If market A closed in December and market B closed in
        # January, the model must not be allowed to "learn" from B while being
        # "tested" on A. Sorting and then slicing in order prevents this.
        data = list(settled_data)
        data.sort(key=lambda x: x.get('close_time', ''))

        # 60% of the data goes to training, 40% to testing (out-of-sample evaluation).
        # This ratio is a commonly-used default — enough data to train a reasonably
        # good model while leaving a meaningful test set.
        split_idx = int(len(data) * 0.6)
        train_data = data[:split_idx]   # Earlier markets — the model learns from these
        test_data = data[split_idx:]     # Later markets — we pretend to trade these

        result.train_samples = len(train_data)
        result.test_samples = len(test_data)

        # Train ensemble model
        train_features = [d["features"] for d in train_data]
        train_outcomes = [d["outcome"] for d in train_data]
        train_result = self.model.train_on_historical(train_features, train_outcomes)

        if isinstance(train_result, dict):
            result.cv_accuracy = train_result.get("cv_accuracy", 0)
            result.oob_score = train_result.get("oob_score", 0)

        # ── Data leakage prevention ──────────────────────────────────────────────
        # This is one of the most important correctness checks in the entire backtest.
        #
        # "Data leakage" occurs when information from the future accidentally
        # influences decisions made in the past. It makes backtest results look
        # far better than they actually are in real life.
        #
        # The history cache stores recent price movements for each market ticker.
        # Some features (like "is the price trending up?") are computed from this
        # cache. If the cache was built using training data, then during testing
        # the model would already know what happened in the earlier period — giving
        # it an unfair advantage that wouldn't exist in real trading.
        #
        # Solution: clear the cache completely before the test phase, then rebuild
        # it sequentially from test data only, one market at a time, in order.
        _test_history_cache: dict[str, list[dict]] = {}

        # Replay strategy on test set
        balance = cfg.initial_balance_cents
        self.tracker = PerformanceTracker()

        for item in test_data:
            market: Market = item["market"]
            actual_outcome: int = item["outcome"]

            if market.volume < cfg.min_volume:
                continue

            # Build history sequentially from test data only (prevents leakage)
            ticker = market.ticker
            if ticker not in _test_history_cache:
                _test_history_cache[ticker] = []
            _test_history_cache[ticker].append({
                "yes_mid": market.mid_price_yes,
                "volume": market.volume,
            })
            if len(_test_history_cache[ticker]) > 100:
                _test_history_cache[ticker] = _test_history_cache[ticker][-100:]

            # Re-extract features using test-only history to prevent data leakage
            event: Event = item["event"]
            features = extract_features(market, event, _test_history_cache.get(ticker))

            # Model prediction
            model_prob = self.model.predict_probability(features)
            market_price = market.mid_price_yes / 100

            # ── Confidence gate ─────────────────────────────────────────────────
            # confidence = max(model_prob, 1 - model_prob) converts the model's
            # probability into a "how sure are we?" number.
            #
            # Example: model_prob = 0.72 → confidence = max(0.72, 0.28) = 0.72
            #          model_prob = 0.35 → confidence = max(0.35, 0.65) = 0.65
            #
            # Both of those pass a 0.65 threshold. A model_prob of exactly 0.50
            # (complete uncertainty) would produce confidence = 0.50 and be filtered.
            # This prevents the bot from trading when it genuinely doesn't know.
            confidence = max(model_prob, 1 - model_prob)
            if confidence < cfg.min_confidence:
                # Too uncertain — skip this market. Count it as a filtered signal.
                result.signals_filtered += 1
                continue

            # ── Edge gate (entry threshold) ─────────────────────────────────────
            # We only enter a trade if the market price is cheap relative to the
            # model's fair probability estimate.
            #
            # For YES trades:
            #   Enter if market_price <= model_prob × (1 - entry_threshold)
            #   With entry_threshold=0.07, that means market_price <= model_prob × 0.93
            #   e.g., model says 70% → enter only if market price is ≤ 65¢
            #   We're buying something worth 70¢ for 65¢ or less — positive edge.
            #
            # For NO trades:
            #   Mirror image — enter if the NO side is cheap relative to (1 - model_prob).
            #   We check if the NO price (1 - market_price) is cheap vs our estimate of P(NO).
            #
            # entry_price always stores the YES market price for consistent P&L calc.
            # The NO side costs (1 - entry_price) per contract.
            if model_prob > 0.5 and market_price <= model_prob * (1 - cfg.entry_threshold):
                side = "yes"
                entry_price = market_price
                edge = model_prob - market_price
            elif model_prob < 0.5 and (1 - market_price) <= (1 - model_prob) * (1 - cfg.entry_threshold):
                side = "no"
                entry_price = market_price  # Store YES price; NO cost = 1 - market_price
                edge = (1 - model_prob) - (1 - market_price)
            else:
                # Not enough edge — the market is fairly priced or too expensive.
                result.signals_filtered += 1
                continue

            result.signals_generated += 1

            # ── Kelly-optimal position sizing ────────────────────────────────────
            # The Kelly Criterion is a mathematical formula for how much to bet.
            # It answers: "Given my edge and the odds, what fraction of my bankroll
            # should I risk to maximize long-term growth?"
            #
            # Kelly formula: f = (b × p - q) / b
            #   where: b = odds (how much you win per dollar risked)
            #          p = probability of winning
            #          q = probability of losing = 1 - p
            #
            # Full Kelly can be too aggressive (large swings), so we scale it down
            # by config.kelly_fraction (typically 0.25 = "quarter Kelly").
            #
            # We also scale by volatility: more volatile features → smaller bet.
            # This is like buying less insurance during a hurricane vs. calm weather.
            cost_per_contract = int(entry_price * 100) if side == "yes" else int((1 - entry_price) * 100)
            if cost_per_contract <= 0:
                continue
            cost_per_contract = max(cost_per_contract, 1)
            market_cost = entry_price if side == "yes" else (1 - entry_price)
            if market_cost <= 0.01 or market_cost >= 0.99:
                continue
            b = (1.0 - market_cost) / market_cost
            win_prob = model_prob if side == "yes" else (1 - model_prob)
            kelly_f = max(0, (b * win_prob - (1 - win_prob)) / b) if b > 0 else 0
            kelly_f *= config.kelly_fraction
            vol = max(features.get("volatility", 0.05), 0.01)
            vol_scalar = min(1.0, 0.05 / vol)
            bet_size = int(kelly_f * vol_scalar * balance)
            bet_size = max(1, min(bet_size, cfg.max_bet_cents, balance))
            contracts = max(1, bet_size // cost_per_contract)

            if contracts * cost_per_contract > balance:
                continue

            # ── Apply slippage ───────────────────────────────────────────────────
            # Slippage makes the backtest more realistic by modeling the fact that
            # you never get the exact price you see on screen. When you buy, the
            # actual fill is slightly higher. When you sell (or the market settles),
            # the effective exit price is slightly lower.
            #
            # Example with 1¢ slippage:
            #   You see a YES market at 55¢. You pay 56¢ (entry worsened by 1¢).
            #   Market settles YES at $1.00. You effectively receive 99¢ (exit worsened by 1¢).
            #   Net: paid 56¢, received 99¢ → profit of 43¢ instead of ideal 45¢.
            slippage_frac = cfg.slippage_cents / 100.0
            entry_price_adj = min(1.0, entry_price + slippage_frac)  # Entry worsened by slippage

            # In backtesting, the "exit price" is the settlement value: 1.0 if YES won,
            # 0.0 if NO won. Real markets resolve to exactly 100¢ or 0¢.
            # We still apply a small slippage to the exit to be conservative.
            exit_price = 1.0 if actual_outcome == 1 else 0.0
            exit_price_adj = max(0.0, exit_price - slippage_frac)  # Exit worsened by slippage

            if side == "yes":
                pnl = (exit_price_adj - entry_price_adj) * 100 * contracts
            else:
                pnl = (entry_price_adj - exit_price_adj) * 100 * contracts

            # Fix 14: Subtract commission from each trade P&L
            pnl -= cfg.commission_cents

            # ── MAE and MFE ─────────────────────────────────────────────────────
            # MAE = Maximum Adverse Excursion: how far the position moved AGAINST you.
            # MFE = Maximum Favorable Excursion: how far the position moved FOR you.
            #
            # These metrics reveal how much risk you had to endure to earn a profit.
            # A winning trade with a large MAE means you were almost stopped out
            # before recovering — that's a risky "survivor." A strategy that has
            # consistent wins with small MAE is much safer.
            #
            # In backtesting with settled markets (no intermediate prices available),
            # we use simplified estimates:
            #   YES trade wins:   MAE=0 (it won, so price ended up),  MFE=1-entry (full gain)
            #   YES trade loses:  MAE=entry (it lost everything),      MFE=0
            #   NO trade wins:    MAE=0,                               MFE=entry (the NO cost was recovered)
            #   NO trade loses:   MAE=1-entry (NO cost lost),          MFE=0
            #
            # For YES: MAE = entry_price (full loss if loses), MFE = 1 - entry_price (full gain if wins)
            # For NO: MAE = 1 - entry_price (NO cost if loses), MFE = entry_price (profit if wins)
            if side == "yes":
                mae = entry_price if actual_outcome == 0 else 0
                mfe = (1 - entry_price) if actual_outcome == 1 else 0
            else:
                mae = (1 - entry_price) if actual_outcome == 1 else 0
                mfe = entry_price if actual_outcome == 0 else 0

            self.tracker.record_trade(
                ticker=market.ticker,
                side=side,
                entry_price=entry_price_adj,
                exit_price=exit_price_adj,
                contracts=contracts,
                mae=mae,
                mfe=mfe,
                model_probability=model_prob,
                market_probability_at_entry=market_price,
            )

            # Deduct cost and add settlement proceeds
            balance -= cost_per_contract * contracts
            if side == "yes":
                # YES settlement: receive exit_price_adj * 100 per contract (slippage-adjusted)
                balance += round(exit_price_adj * 100) * contracts
            else:
                # NO settlement: receive (1 - exit_price_adj) * 100 per contract (slippage-adjusted)
                balance += round((1 - exit_price_adj) * 100) * contracts

        # Fix 16: Handle case where no trades generated
        if not self.tracker.trades:
            result.config["warning"] = "No trades generated with the given parameters"
            logger.info(f"Backtest completed in {time.time() - _backtest_start:.1f}s — no trades generated")
            return result

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

        # Fix 17: Track best/worst individual trade P&L
        result.best_trade_pnl = metrics.best_trade_pnl
        result.worst_trade_pnl = metrics.worst_trade_pnl

        # ── Real-time duration analysis ──
        # Parse close_time from each traded market to estimate calendar duration.
        # Entry time is estimated as close_time minus 24h (typical market lifespan);
        # exit time is the market's close_time (settlement).
        trade_timestamps: list[tuple[datetime, datetime]] = []
        for item in test_data:
            market = item["market"]
            if not market.close_time:
                continue
            # Only include markets that were actually traded
            if market.ticker not in {t.get("ticker") for t in result.trades}:
                continue
            try:
                close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
                # Estimate entry as 24h before close (typical short-duration market)
                entry_dt = close_dt - timedelta(hours=24)
                trade_timestamps.append((entry_dt, close_dt))
            except (ValueError, AttributeError):
                continue

        if trade_timestamps:
            trade_timestamps.sort(key=lambda t: t[0])
            first_entry = trade_timestamps[0][0]
            last_exit = max(t[1] for t in trade_timestamps)
            duration = last_exit - first_entry
            duration_days = max(duration.days, 1)

            result.calendar_duration_days = duration_days

            # Average hold time per trade (entry to exit). Fix 18: Also set avg_trade_duration_hours
            hold_hours = [(t[1] - t[0]).total_seconds() / 3600 for t in trade_timestamps]
            result.avg_hold_time_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0
            result.avg_trade_duration_hours = result.avg_hold_time_hours

            # Trades per day
            result.trades_per_day = len(trade_timestamps) / duration_days if duration_days > 0 else 0.0

            # Annualized return: scale P&L to 365 days
            if cfg.initial_balance_cents > 0 and duration_days > 0:
                total_return = result.total_pnl_cents / cfg.initial_balance_cents
                annualization_factor = 365.0 / duration_days
                result.annualized_return_pct = total_return * annualization_factor * 100

            # Annualized Sharpe: daily Sharpe * sqrt(365)
            result.annualized_sharpe = round(result.sharpe_ratio * math.sqrt(365), 2)

            # Time in market: % of calendar days with at least one open position
            active_days: set = set()
            for entry_dt, exit_dt in trade_timestamps:
                current = entry_dt.date()
                end = exit_dt.date()
                while current <= end:
                    active_days.add(current.toordinal())
                    current = current + timedelta(days=1)

            total_calendar_days_range = (last_exit.date() - first_entry.date()).days + 1
            result.time_in_market_pct = (len(active_days) / total_calendar_days_range * 100) if total_calendar_days_range > 0 else 0.0

        # Fix 15: Log end of backtest
        logger.info(
            f"Backtest completed in {time.time() - _backtest_start:.1f}s: "
            f"{result.total_trades} trades, Sharpe={result.sharpe_ratio}, "
            f"P&L={result.total_pnl_cents}c, WR={result.win_rate:.1%}"
        )
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

        ─────────────────────────────────────────────────────────────────────
        WHAT IS A PARAMETER SWEEP?

        A "grid search" or "sweep" means testing every combination of two or
        more settings to see which combination performs best. It's like trying
        every combination of oven temperature and baking time to find the
        perfect cookie recipe.

        Example with 5 entry thresholds × 5 confidence levels:
          entry_threshold = [0.4, 0.45, 0.5, 0.55, 0.6]
          confidence_level = [0.60, 0.65, 0.70, 0.75, 0.80]

          That's 5 × 5 = 25 complete backtests, each using different settings.
          The one with the best Sharpe Ratio wins.

        ─────────────────────────────────────────────────────────────────────
        THE OVERFITTING DANGER IN PARAMETER SWEEPS

        WARNING: Running hundreds of combinations and picking the best one is
        a form of overfitting. You might be finding parameters that are "lucky"
        on this specific historical dataset, not parameters that reflect a
        genuine market edge.

        To mitigate this:
          1. Each run uses a fresh, independently-trained model.
          2. We use Sharpe Ratio rather than raw profit — Sharpe penalizes
             volatility, so lucky-but-inconsistent results score poorly.
          3. Walk-forward testing (see walk_forward()) provides additional
             validation by testing on multiple independent time windows.
        """
        entry_thresholds = entry_thresholds or [0.4, 0.45, 0.5, 0.55, 0.6]
        confidence_levels = confidence_levels or [0.60, 0.65, 0.70, 0.75, 0.80]

        results = []
        for entry in entry_thresholds:
            for conf in confidence_levels:
                # Fix 20: Skip combinations where entry_threshold > exit_threshold (invalid)
                if entry > 0.9:  # exit_threshold defaults to 0.9
                    logger.debug(f"Sweep: skipping entry={entry} > exit_threshold=0.9")
                    continue

                # Fix 19: Log each parameter combination tested
                logger.info(f"Sweep: testing entry_threshold={entry}, min_confidence={conf}")
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

    # ── Task 11: Monte Carlo Simulation ───────────────────────────────────

    def monte_carlo_sim(
        self,
        trades: list[dict] | None = None,
        n_simulations: int = 1000,
    ) -> dict:
        """Randomly reorder the trade sequence and compute distribution of outcomes.

        Args:
            trades: List of trade dicts with pnl_cents. If None, uses self.tracker.trades.
            n_simulations: Number of random permutations to simulate.

        Returns:
            Dict with percentile distributions for final P&L, max drawdown, and Sharpe ratio.

        ─────────────────────────────────────────────────────────────────────
        WHAT IS MONTE CARLO SIMULATION?

        Named after the famous casino in Monaco, Monte Carlo simulation answers
        the question: "How would my strategy perform under different possible
        orderings of the same trades?"

        In a real backtest, the trades happened in a specific order — some won
        early, some lost late, etc. But what if the losing trades had come first?
        What if all your big wins clustered in one week and all your losses in
        another? The order of trades affects things like drawdowns significantly.

        Monte Carlo simulation takes your actual trades and shuffles them into
        1,000 different random orders. For each shuffle, it computes final P&L,
        max drawdown, and Sharpe Ratio. This gives you a DISTRIBUTION of outcomes
        rather than a single number.

        From this distribution you can ask:
          - "In the worst 5% of scenarios, how much would I have lost?" (p5)
          - "What's the typical outcome?" (p50, the median)
          - "In the best 5% of scenarios, how much would I have gained?" (p95)

        A strategy whose p5 (worst case) is still profitable is much more robust
        than one whose typical result looks good but occasionally destroys the
        account.

        ─────────────────────────────────────────────────────────────────────
        PERCENTILE NOTATION

        p5  = 5th percentile  → 5% of scenarios were worse than this
        p25 = 25th percentile → 25% of scenarios were worse
        p50 = 50th percentile → the median (half were better, half worse)
        p75 = 75th percentile → 75% were worse (i.e., you were in top 25%)
        p95 = 95th percentile → only 5% of scenarios did better than this

        Think of it like test scores: "p50 = 70 points" means half the class
        scored below 70 and half scored above.
        """
        if trades is None:
            trade_list = self.tracker.trades
        else:
            trade_list = trades

        if len(trade_list) < 5:
            return {"error": "Need at least 5 trades for Monte Carlo simulation"}

        # Extract PnL and return arrays
        pnls = []
        simple_returns = []
        for t in trade_list:
            if isinstance(t, dict):
                pnl = t.get("pnl_cents", 0)
                entry = t.get("entry_price", 0.5)
                exit_p = t.get("exit_price", 0.5)
                side = t.get("side", "yes")
            else:
                pnl = t.pnl_cents
                entry = t.entry_price
                exit_p = t.exit_price
                side = t.side
            pnls.append(pnl)
            if side == "yes":
                cost = entry
                payout = exit_p
            else:
                cost = 1 - entry
                payout = 1 - exit_p
            if cost > 0:
                simple_returns.append((payout - cost) / cost)
            else:
                simple_returns.append(0.0)

        pnl_arr = np.array(pnls)   # Profit/loss for each trade, in cents
        ret_arr = np.array(simple_returns)  # Percentage return for each trade
        # Seed the random number generator for reproducibility.
        # Using seed=42 means the same 1,000 shuffles happen every time you run this.
        # This is important — if every run produced different shuffles, the results
        # would look different each time, making it hard to compare runs.
        rng = np.random.RandomState(42)

        final_pnls = []
        max_drawdowns = []
        sharpe_ratios = []
        # The "risk-free rate" is the return you'd get by doing nothing (e.g., a savings
        # account). Here it's 5% annually, divided by 365 to get a daily rate.
        # The Sharpe Ratio measures excess return above this baseline.
        risk_free_daily = 0.05 / 365

        for _ in range(n_simulations):
            idx = rng.permutation(len(pnl_arr))
            shuffled_pnl = pnl_arr[idx]
            shuffled_ret = ret_arr[idx]

            # Final P&L
            final_pnls.append(float(np.sum(shuffled_pnl)))

            # Max drawdown
            cumsum = np.cumsum(shuffled_pnl)
            peak = np.maximum.accumulate(cumsum)
            drawdowns = peak - cumsum
            max_drawdowns.append(float(np.max(drawdowns)) if len(drawdowns) > 0 else 0)

            # Sharpe ratio
            mean_ret = float(np.mean(shuffled_ret))
            std_ret = float(np.std(shuffled_ret, ddof=1)) if len(shuffled_ret) > 1 else 0
            sharpe = (mean_ret - risk_free_daily) / std_ret if std_ret > 0 else 0
            sharpe_ratios.append(sharpe)

        def _percentiles(arr):
            arr_np = np.array(arr)
            return {
                "p5": round(float(np.percentile(arr_np, 5)), 2),
                "p25": round(float(np.percentile(arr_np, 25)), 2),
                "p50": round(float(np.percentile(arr_np, 50)), 2),
                "p75": round(float(np.percentile(arr_np, 75)), 2),
                "p95": round(float(np.percentile(arr_np, 95)), 2),
                "mean": round(float(np.mean(arr_np)), 2),
                "std": round(float(np.std(arr_np)), 2),
            }

        return {
            "n_simulations": n_simulations,
            "n_trades": len(pnls),
            "final_pnl": _percentiles(final_pnls),
            "max_drawdown": _percentiles(max_drawdowns),
            "sharpe_ratio": _percentiles(sharpe_ratios),
            "actual_final_pnl": int(np.sum(pnl_arr)),
            "actual_sharpe": round(
                (float(np.mean(ret_arr)) - risk_free_daily) / max(float(np.std(ret_arr, ddof=1)), 0.001), 2
            ),
        }

    # ── Task 12: Walk-Forward Optimization ────────────────────────────────

    def walk_forward(
        self,
        settled_data: list[dict],
        n_windows: int = 4,
        train_months: int = 3,
        test_months: int = 1,
        cfg: BacktestConfig | None = None,
    ) -> dict:
        """Split data into rolling windows and compute out-of-sample performance.

        More robust than a single train/test split. Each window trains on
        `train_months` of data and tests on the next `test_months`.

        Args:
            settled_data: Historical settled market data, ideally sorted by time.
            n_windows: Number of rolling windows.
            train_months: Months of data per training window.
            test_months: Months of data per test window.
            cfg: Backtest configuration.

        Returns:
            Dict with per-window and aggregate out-of-sample metrics.

        ─────────────────────────────────────────────────────────────────────
        WHAT IS WALK-FORWARD TESTING?

        Walk-forward testing is the most rigorous form of backtesting. Instead
        of one train/test split, you slide a window through time repeatedly.

        Imagine you have 12 months of data:

          Window 1:
            Train on months  1–3  →  Test on month  4
          Window 2:
            Train on months  1–4  →  Test on month  5
          Window 3:
            Train on months  1–5  →  Test on month  6
          ... and so on.

        Each test period is truly "out of sample" — the model has never seen it.
        But unlike a single split, you get MULTIPLE independent test results.

        WHY IS THIS BETTER THAN ONE SPLIT?

        With one 60/40 split, you might get lucky or unlucky with which 40% you
        happened to test on. Walk-forward gives you several test periods, each
        showing how the strategy performed on fresh, unseen data.

        If a strategy is genuinely good, it should work across multiple windows.
        If it only works in one specific period, that's a red flag — the strategy
        may have just gotten lucky in that one period.

        The "consistency" score in the output (fraction of windows with Sharpe > 0)
        is the key metric: a strategy that's profitable in 4 out of 4 windows is
        far more trustworthy than one that's profitable in 6 out of 10.
        """
        cfg = cfg or BacktestConfig()

        if len(settled_data) < 50:
            return {"error": f"Only {len(settled_data)} samples. Need 50+."}

        # Split data into n_windows + 1 equal chunks
        total_chunks = n_windows + 1
        chunk_size = len(settled_data) // total_chunks
        if chunk_size < 20:
            return {"error": f"Chunk size {chunk_size} too small. Need more data or fewer windows."}

        window_results = []
        all_oos_trades = []

        for i in range(n_windows):
            train_start = i * chunk_size
            train_end = train_start + chunk_size * (total_chunks - n_windows + i)
            # Ensure at least chunk_size training samples
            train_end = min(train_end, len(settled_data) - chunk_size)
            train_end = max(train_end, train_start + chunk_size)
            test_start = train_end
            test_end = min(test_start + chunk_size, len(settled_data))

            if test_end <= test_start or train_end <= train_start:
                continue

            train_data = settled_data[train_start:train_end]
            test_data = settled_data[test_start:test_end]

            if len(train_data) < 30 or len(test_data) < 5:
                continue

            # Train model on this window's training data
            model = PredictionModel(n_estimators=500)
            train_features = [d["features"] for d in train_data]
            train_outcomes = [d["outcome"] for d in train_data]
            train_result = model.train_on_historical(train_features, train_outcomes)

            if not train_result.get("trained", False):
                continue

            # Test on out-of-sample data
            tracker = PerformanceTracker()
            trades_in_window = 0

            for item in test_data:
                market = item["market"]
                actual_outcome = item["outcome"]
                if market.volume < cfg.min_volume:
                    continue

                event = item["event"]
                features = extract_features(market, event)
                model_prob = model.predict_probability(features)
                market_price = market.mid_price_yes / 100
                confidence = max(model_prob, 1 - model_prob)

                if confidence < cfg.min_confidence:
                    continue

                if model_prob > 0.5 and market_price <= model_prob * (1 - cfg.entry_threshold):
                    side = "yes"
                    entry_price = market_price
                elif model_prob < 0.5 and (1 - market_price) <= (1 - model_prob) * (1 - cfg.entry_threshold):
                    side = "no"
                    entry_price = market_price
                else:
                    continue

                exit_price = 1.0 if actual_outcome == 1 else 0.0
                if side == "yes":
                    mae = entry_price if actual_outcome == 0 else 0
                    mfe = (1 - entry_price) if actual_outcome == 1 else 0
                else:
                    mae = (1 - entry_price) if actual_outcome == 1 else 0
                    mfe = entry_price if actual_outcome == 0 else 0

                tracker.record_trade(
                    ticker=market.ticker, side=side,
                    entry_price=entry_price, exit_price=exit_price,
                    contracts=1, mae=mae, mfe=mfe,
                    model_probability=model_prob, market_probability_at_entry=market_price,
                )
                trades_in_window += 1

            metrics = tracker.get_metrics()
            window_results.append({
                "window": i + 1,
                "train_size": len(train_data),
                "test_size": len(test_data),
                "trades": metrics.total_trades,
                "win_rate": round(metrics.win_rate, 4),
                "total_pnl_cents": metrics.total_pnl_cents,
                "sharpe_ratio": metrics.sharpe_ratio,
                "profit_factor": metrics.profit_factor,
                "cv_accuracy": train_result.get("cv_accuracy", 0),
            })
            all_oos_trades.extend(tracker.trades)

        if not window_results:
            return {"error": "No valid windows produced results"}

        # Aggregate out-of-sample metrics
        total_trades = sum(w["trades"] for w in window_results)
        avg_win_rate = sum(w["win_rate"] * w["trades"] for w in window_results) / max(total_trades, 1)
        total_pnl = sum(w["total_pnl_cents"] for w in window_results)
        avg_sharpe = sum(w["sharpe_ratio"] for w in window_results) / len(window_results)
        sharpe_std = float(np.std([w["sharpe_ratio"] for w in window_results])) if len(window_results) > 1 else 0

        return {
            "n_windows": len(window_results),
            "windows": window_results,
            "aggregate": {
                "total_trades": total_trades,
                "avg_win_rate": round(avg_win_rate, 4),
                "total_pnl_cents": total_pnl,
                "avg_sharpe": round(avg_sharpe, 2),
                "sharpe_std": round(sharpe_std, 2),
                "consistency": round(
                    sum(1 for w in window_results if w["sharpe_ratio"] > 0) / len(window_results), 2
                ),
            },
        }


# ── Paper Trading Simulator ──────────────────────────────────────────────────

@dataclass
class PaperPosition:
    """A simulated (paper) position with MAE/MFE tracking.

    min_price_seen and max_price_seen are updated on every scan cycle
    to track how far the price moved against/in favor of the position
    before it was closed.

    ─────────────────────────────────────────────────────────────────────────
    WHAT IS A PAPER POSITION?

    In real trading, a "position" is money you currently have at risk in a
    market. A "paper position" is the same concept but with no real money.

    Think of it like keeping score on paper: "I would have bought 10 contracts
    of this market at 55¢. It moved to 70¢. On paper I'm up $1.50."

    Paper positions are tracked with all the same details as real positions:
    - Which market (ticker)
    - Which direction we bet (yes or no)
    - How much we paid (entry_price)
    - How many contracts we bought (contracts)
    - The model's conviction at entry (model_prob)
    - When we entered (entry_time)

    The min_price_seen / max_price_seen fields are updated on every scan cycle
    to track MAE and MFE across the position's lifetime.
    """
    ticker: str                        # Kalshi market ticker (e.g., "KXBTC-15M-25000")
    side: str                          # "yes" = betting YES wins; "no" = betting NO wins
    entry_price: float                 # Entry price as decimal (0.0 to 1.0, where 0.55 = 55¢)
    contracts: int                     # Number of simulated contracts (each contract pays $1 if correct)
    model_prob: float                  # Model's predicted probability at entry
    entry_time: str                    # ISO timestamp of simulated entry
    category: str = ""                 # Market category for per-category analytics
    entry_edge: float = 0.0            # Edge at entry (for stop-loss/take-profit scaling)
    min_price_seen: float = 1.0        # Lowest price seen since entry (for MAE calculation)
    max_price_seen: float = 0.0        # Highest price seen since entry (for MFE calculation)


class PaperTrader:
    """
    Paper trading simulator that uses live market data but
    simulates order fills without risking real money.

    Tracks full performance metrics, MAE/MFE, and Sharpe Ratio.

    ─────────────────────────────────────────────────────────────────────────
    HOW PAPER TRADING WORKS IN THIS BOT

    Every 1–3 seconds, the bot runs a "scan cycle":
      1. Look at all currently open paper positions.
         - Has any of them hit its exit target?
         - Has any market settled (resolved)?
         - Should we cut losses on any?
         → Close those positions and record the P&L.

      2. Generate new entry signals.
         - Fetch live prices from Kalshi.
         - Compute features for each market.
         - Ask the model: "Which markets are worth trading?"
         → For good signals: open a new paper position.

    This runs continuously in the background. Over days and weeks, a track
    record builds up showing whether the strategy would have been profitable
    with real money.

    ─────────────────────────────────────────────────────────────────────────
    WHY PAPER TRADE BEFORE GOING LIVE?

    Backtesting on historical data has a crucial weakness: the historical data
    is "clean." It doesn't include:
      - API latency (how long it takes for data to arrive)
      - Market impact (your order moving the price)
      - Model errors (wrong predictions in real-time conditions)
      - Edge cases (markets that briefly show incorrect data)

    Paper trading exposes all of these issues in a live environment without
    financial consequences. It's the final validation step before deploying
    real capital.

    ─────────────────────────────────────────────────────────────────────────
    PERSISTENCE: SURVIVING CRASHES

    If the server crashes or restarts, all paper trading state is saved to:
      - Supabase (cloud database) — the authoritative record
      - Local JSON file — fast read on startup

    This means paper trading state survives restarts. Without this, a server
    crash during the night would wipe out weeks of paper trading history.
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
        self._dirty: bool = False                          # Tracks whether state changed since last save
        self.model_version: int = 0                        # Incrementing version number for trained models (Task 41)

    def configure(self, balance_cents: int = 100_00):
        """Set starting balance. Fix 24: Validates balance > 0."""
        if balance_cents <= 0:
            logger.warning(f"configure: balance_cents must be > 0, got {balance_cents}. Using default 10000.")
            balance_cents = 100_00
        self.balance_cents = balance_cents

    def add_funds(self, amount_cents: int) -> int | dict:
        """Add demo funds to paper balance without resetting state. Fix 25: Validates amount > 0."""
        if amount_cents <= 0:
            logger.warning(f"add_funds: amount must be > 0, got {amount_cents}")
            return {"error": f"Amount must be > 0, got {amount_cents}", "balance_cents": self.balance_cents}
        self.balance_cents += amount_cents
        self._dirty = True
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

        # Fix 23: Handle case where generator is None
        if self.generator is None:
            logger.error("scan_and_trade: generator is None, cannot generate signals")
            return {
                "scan_number": self.total_scans,
                "entries": [],
                "exits": [],
                "open_positions": len(self.positions),
                "balance_cents": self.balance_cents,
                "total_trades": self.tracker.get_metrics().total_trades,
                "error": "Signal generator not initialized",
            }

        # Check exits first
        import time as _t
        _t0 = _t.time()
        exits = self._check_exits(events)
        _t1 = _t.time()

        # Generate new signals (pass position count for dynamic quality threshold)
        signals = self.generator.generate_signals(events, n_positions=len(self.positions))
        self._last_signals = signals  # Cache for live trading to reuse
        _t2 = _t.time()
        logger.info(f"[scan_and_trade] exits: {_t1-_t0:.1f}s, signals: {_t2-_t1:.1f}s, {len(events)} events")
        self.signals_seen += len(signals)

        # Mark dirty if there were exits or signals found
        if exits or signals:
            self._dirty = True

        # Build ticker -> category map once to avoid O(signals * events * markets) lookup
        ticker_category_map: dict[str, str] = {}
        for ev in events:
            for m in ev.markets:
                ticker_category_map[m.ticker] = ev.category or m.category or ""

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
            # entry_price = YES market price (used for tracking and P&L)
            entry_price = sig.market_probability
            # Cost per contract: YES side pays entry_price, NO side pays (1 - entry_price)
            cost_per_contract = int(entry_price * 100) if sig.side == Side.YES else int((1 - entry_price) * 100)
            cost_per_contract = max(cost_per_contract, 1)
            contracts = max(1, cost // cost_per_contract)

            # Find category from the pre-built map
            sig_category = ticker_category_map.get(sig.ticker, "")

            self._dirty = True
            self.positions[sig.ticker] = PaperPosition(
                ticker=sig.ticker,
                side=sig.side.value,
                entry_price=entry_price,
                contracts=contracts,
                model_prob=sig.fair_probability,
                entry_time=datetime.now(timezone.utc).isoformat(),
                category=sig_category,
                entry_edge=sig.edge,
                min_price_seen=entry_price,
                max_price_seen=entry_price,
            )
            self.balance_cents -= cost_per_contract * contracts
            entries.append({
                "ticker": sig.ticker,
                "side": sig.side.value,
                "entry_price": entry_price,
                "contracts": contracts,
                "model_prob": sig.fair_probability,
                "edge": sig.edge,
            })

        # Fix 21: Log total events/markets scanned
        total_markets = sum(len(ev.markets) for ev in events) if events else 0
        logger.info(f"Scan #{self.total_scans}: {len(events)} events, {total_markets} markets scanned")

        # Fix 22: Log entries and exits separately
        if entries:
            logger.info(f"Scan #{self.total_scans}: {len(entries)} entries — {[e['ticker'] for e in entries]}")
        if exits:
            logger.info(f"Scan #{self.total_scans}: {len(exits)} exits — {[e['ticker'] for e in exits]}")

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
        ticker_to_event = {}
        for event in events:
            for market in event.markets:
                market_map[market.ticker] = market
                ticker_to_event[market.ticker] = event

        tickers_to_close = []
        for ticker, pos in self.positions.items():
            market = market_map.get(ticker)
            if not market:
                continue

            current_price = market.mid_price_yes / 100

            # Track MAE/MFE
            pos.min_price_seen = min(pos.min_price_seen, current_price)
            pos.max_price_seen = max(pos.max_price_seen, current_price)

            settled = market.status == "settled"

            # Time-based: check days to expiry
            days_left = 30
            if market.close_time:
                try:
                    close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
                    days_left = max(0, (close_dt - datetime.now(timezone.utc)).total_seconds() / 86400)
                except (ValueError, TypeError):
                    pass

            # ── Multi-leg exit logic ─────────────────────────────────────────────
            # There are four reasons to close a paper position:
            #   1. STOP-LOSS: price moved too far against us → cut the loss
            #   2. TAKE-PROFIT: price moved far enough in our favor → lock in the gain
            #   3. MODEL DISAGREEMENT: model now strongly disagrees with the position
            #   4. EXPIRY: the market closes tomorrow, close now to avoid timing risk
            #
            # The stop and take-profit distances are proportional to the original edge:
            # if we had a large edge entering, we give the position more room to breathe.
            entry_edge = max(pos.entry_edge, 0.05)
            stop_distance = min(entry_edge * 0.6, 0.15)    # Stop if price moves 60% of edge against us
            take_profit_distance = entry_edge * 0.7          # Target if price moves 70% of edge for us

            # As the market approaches expiry, tighten both stops.
            # A market with 2 days left doesn't have much room to recover from a bad move.
            # We scale the distances down proportionally to remaining time.
            if days_left <= 14:
                time_urgency = max(0.2, days_left / 14.0)
                stop_distance *= time_urgency
                take_profit_distance *= time_urgency

            # Re-evaluate the model on the current state of this market.
            # If the model has changed its mind (e.g., new price data shifted the signals),
            # we can exit early rather than holding a position the model no longer supports.
            # This "model disagreement" exit is a live feedback loop that backtesting
            # alone cannot capture — it's one of the key advantages of paper trading.
            event_for_pos = ticker_to_event.get(ticker)
            history = self.generator.history_cache.get(ticker)
            features = extract_features(market, event_for_pos or Event(event_ticker="", title=""), history)
            model_prob_now = self.generator.model.predict_probability(features)

            reason = None
            if pos.side == "yes":
                if current_price <= pos.entry_price - stop_distance:
                    reason = "Stop-loss"
                elif current_price >= pos.entry_price + take_profit_distance:
                    reason = "Target"
                elif model_prob_now < 0.45:
                    reason = "Model disagreement"
            else:
                if current_price >= pos.entry_price + stop_distance:
                    reason = "Stop-loss"
                elif current_price <= pos.entry_price - take_profit_distance:
                    reason = "Target"
                elif model_prob_now > 0.55:
                    reason = "Model disagreement"

            if days_left <= 1 and reason is None:
                reason = "Expiry"

            if reason or settled:
                # exit_price = YES market price at exit (consistent with entry_price)
                exit_price = current_price
                if settled:
                    # Settlement: YES price goes to 1.0 if YES wins, 0.0 if NO wins
                    exit_price = 1.0 if market.result == "yes" else 0.0

                # MAE/MFE: tracked relative to entry_price (YES price)
                if pos.side == "yes":
                    mae = pos.entry_price - pos.min_price_seen  # Price dropped below entry
                    mfe = pos.max_price_seen - pos.entry_price  # Price rose above entry
                else:
                    mae = pos.max_price_seen - pos.entry_price  # Price rose (bad for NO)
                    mfe = pos.entry_price - pos.min_price_seen  # Price dropped (good for NO)

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

                # Feed edge tracker for adaptive sizing
                realized_return = pnl / max(pos.contracts * 100, 1)
                self.generator.edge_tracker.record(pos.entry_edge, realized_return)

                if not reason:
                    reason = "Settled"
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
            "model_version": self.model_version,
            "last_scan_time": datetime.now(timezone.utc).isoformat() if self.total_scans > 0 else None,  # Fix 28
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

        ─────────────────────────────────────────────────────────────────────
        THE VIRTUOUS CYCLE

        As the bot paper trades, markets settle and new training samples accumulate:

          Week 1: 50 settled markets → model trained on 50 samples, moderate accuracy
          Week 2: 200 more settle   → model trained on 250 samples, better accuracy
          Week 3: 300 more settle   → model trained on 550 samples, even better

        Unlike a snapshot approach (throwing away old data and retraining from scratch),
        this cumulative approach is like a person who learns from every experience
        they've ever had, rather than forgetting everything before last week.

        The model_version counter increments on every training run so you can
        track which model generated which trading signals in the history.
        """
        # Fix 29: Validate settled data is non-empty before proceeding
        if not settled_data:
            logger.warning("train_model: settled_data is empty, nothing to train on")
            return {"trained": False, "error": "No settled data provided", "samples": 0}

        # 1. Add new samples to the cumulative store (deduped by ticker)
        add_result = self.training_store.add_samples(settled_data)
        new_count = add_result["new_count"] if isinstance(add_result, dict) else add_result
        outlier_count = add_result.get("outlier_count", 0) if isinstance(add_result, dict) else 0

        # 2. Train on ALL accumulated samples
        all_features, all_outcomes = self.training_store.get_features_and_outcomes()
        total_samples = len(all_features)

        logger.info(
            f"Training on {total_samples} cumulative samples "
            f"({new_count} new, {total_samples - new_count} existing, {outlier_count} outliers)"
        )

        result = self.generator.model.train_on_historical(all_features, all_outcomes)

        # Task 41: Increment model version on each training run
        self.model_version += 1

        # Add cumulative info to result (Fix 30: Include train/test split info)
        if isinstance(result, dict):
            result["total_cumulative_samples"] = total_samples
            result["new_samples_added"] = new_count
            result["outlier_count"] = outlier_count
            result["samples"] = total_samples  # Override with true count
            result["model_version"] = self.model_version
            # Fix 30: Include train/test split info for transparency
            result["train_samples"] = total_samples
            result["test_samples_estimate"] = int(total_samples * 0.2)  # ~20% used for CV
            result["training_features_count"] = len(all_features[0]) if all_features and all_features[0] else 0

        # Persist training run to DB with version number
        if self.db and self.db.is_connected and isinstance(result, dict):
            try:
                self.db.insert_training_run(
                    samples=total_samples,
                    cv_accuracy=result.get("cv_accuracy", 0),
                    oob_score=result.get("oob_score", 0),
                    n_features=result.get("n_features", 106),
                    n_estimators=self.generator.model.n_estimators,
                    feature_importance=self.generator.model.get_feature_importance(),
                    model_version=self.model_version,
                )
            except Exception:
                pass

        return result

    # ── Persistence ──────────────────────────────────────────────

    def save_state(self, path: Path | None = None, force: bool = False):
        """Save paper trading state to JSON file and DB.

        Only writes if state has changed (dirty flag) or force=True.
        Resets the dirty flag after a successful save.
        """
        if not self._dirty and not force:
            return
        # Save to DB if connected
        if self.db and self.db.is_connected:
            try:
                positions_dict = {
                    ticker: {
                        "ticker": p.ticker, "side": p.side,
                        "entry_price": p.entry_price, "contracts": p.contracts,
                        "model_prob": p.model_prob, "entry_time": p.entry_time,
                        "min_price_seen": p.min_price_seen, "max_price_seen": p.max_price_seen,
                        "entry_edge": p.entry_edge, "category": p.category,
                    }
                    for ticker, p in self.positions.items()
                }
                self.db.save_paper_state(
                    self.balance_cents, self.total_scans, self.signals_seen, positions_dict
                )
            except Exception:
                pass

        # Also save to local JSON as cache (best-effort, ephemeral on Railway)
        try:
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
                        "entry_edge": p.entry_edge,
                        "category": p.category,
                    }
                    for ticker, p in self.positions.items()
                },
                "trades": self.tracker.get_trade_history(),
                "equity_curve": self.tracker.get_equity_curve(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            path.write_text(json.dumps(state, indent=2, default=str))
        except OSError:
            pass  # Local cache is best-effort — DB is authoritative

        # Fix 26: Log success with balance and position count
        logger.info(f"Paper state saved: balance={self.balance_cents}c, positions={len(self.positions)}, scans={self.total_scans}")
        self._dirty = False

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
                            entry_edge=p.get("entry_edge", 0.0),
                            category=p.get("category", ""),
                        )
                    # Fix 27: Log which source was used
                    logger.info(f"Paper state loaded from DB: balance={self.balance_cents}c, positions={len(self.positions)}")
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
                    entry_edge=p.get("entry_edge", 0.0),
                    category=p.get("category", ""),
                )

            # Restore trade history into tracker WITHOUT re-inserting into DB.
            # We pass db=None to avoid duplicating rows on every server restart.
            self.tracker = PerformanceTracker(db=None, mode="paper")
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
            # Now set the real db so future trades get persisted
            self.tracker.db = self.db
            # Fix 27: Log which source was used (file)
            logger.info(f"Paper state loaded from file: balance={self.balance_cents}c, positions={len(self.positions)}")
            return True
        except (json.JSONDecodeError, KeyError):
            return False
