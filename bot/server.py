"""
FastAPI REST server for the prediction market trading bot.

This module is the central backend that powers the React + Vite frontend.
It exposes 32+ REST endpoints organized into logical groups:

  - Status & Config:    GET /api/status, PATCH /api/config
  - Market Data:        GET /api/events, GET /api/market/{ticker}
  - Portfolio & Orders: GET /api/portfolio, POST /api/trade, etc.
  - Signals:            POST /api/scan, GET /api/signals
  - Performance:        GET /api/performance, GET /api/model/features
  - Backtesting:        POST /api/backtest, POST /api/backtest/sweep
  - Paper Trading:      GET /api/paper, POST /api/paper/scan, POST /api/paper/train
  - Auto-Scan/Trade:    POST /api/autoscan, POST /api/autotrade
  - Arbitrage:          GET /api/arbitrage
  - History (Supabase): GET /api/history/trades, /scans, /performance, /training
  - Notifications:      POST /api/notifications/test
  - Webhook:            POST /api/webhook (external trigger for scans/retrains)
  - Export:             GET /api/export/trades (CSV download)

Connects to:
  - Kalshi Trade API v2 (via KalshiClient) for market data and order execution
  - Supabase (via Database) for persistent storage of trades, scans, training data
  - APScheduler for background auto-scanning (every 60s) and scheduled retraining
  - Slack/Discord webhooks (via Notifier) for trade alerts

Deployment: Railway (backend), with the React frontend on Vercel calling these endpoints.

The server maintains in-memory shared state (module-level globals) for:
  - Client instances (kalshi, dk, rf_generator, etc.)
  - A cache dict (_cache) holding latest scan results, trade logs, and scheduler state
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from bot.config import config
from bot.kalshi_client import KalshiClient
from bot.draftkings_client import DraftKingsClient
from bot.rf_model import RFSignalGenerator, extract_features, FEATURE_NAMES, PredictionModel
from bot.analyzer import MarketAnalyzer
from bot.risk_manager import RiskManager
from bot.performance import PerformanceTracker
from bot.arbitrage import detect_arbitrage
from bot.backtester import Backtester, BacktestConfig, HistoricalDataFetcher, PaperTrader
from bot.database import Database
from bot.notifier import Notifier
from bot.models import OrderRequest, Side, TradingSignal

logger = logging.getLogger("predictionbot")


# ── Shared State ─────────────────────────────────────────────────────────────
# Module-level globals hold all singleton service instances. They are initialized
# in the lifespan() context manager when the FastAPI app starts, and cleaned up
# when it shuts down. Using globals here (rather than dependency injection) keeps
# the endpoint handlers simple and avoids passing state through every function.

kalshi: KalshiClient | None = None          # Kalshi Trade API v2 client
dk: DraftKingsClient | None = None          # DraftKings scraper for arbitrage
rf_generator: RFSignalGenerator | None = None  # RF+GB ensemble signal generator
ai_analyzer: MarketAnalyzer | None = None   # Claude AI analyzer (optional)
risk_manager: RiskManager | None = None     # Pre-trade risk validation
performance: PerformanceTracker | None = None  # Live trade performance tracking
paper_trader: PaperTrader | None = None     # Paper trading simulator
db: Database | None = None                  # Supabase persistence layer
scheduler: AsyncIOScheduler | None = None   # APScheduler for background jobs
notifier: Notifier | None = None            # Slack/Discord webhook notifications

# In-memory cache for the latest scan results and trade logs.
# This avoids re-scanning Kalshi on every frontend poll. The cache is updated
# by _auto_scan_job() (background) and run_scan() (on-demand).
# Logs are trimmed to 500 entries max to prevent unbounded memory growth.
_cache: dict[str, Any] = {
    "events": [],               # Latest event data from Kalshi
    "signals": [],              # Latest validated trading signals (RF + AI)
    "exit_signals": [],         # Latest exit signals for open positions
    "arbitrage": [],            # Latest arbitrage opportunities
    "last_scan": None,          # ISO timestamp of most recent scan
    "auto_scan_enabled": False, # Whether the 60s background scan loop is running
    "auto_trade_enabled": False,# Whether live orders are auto-executed (requires auto_scan)
    "auto_scan_log": [],        # Combined scan log for the Settings page
    "paper_trade_log": [],      # Paper (shadow) trade entries/exits
    "live_trade_log": [],       # Live trade entries/exits on Kalshi
}


async def _auto_scan_job():
    """Background job that scans ALL of Kalshi for edges (runs every 60 seconds).

    This is the main workhorse of the bot when auto-scan is enabled. It:
    1. Fetches every open event from Kalshi via cursor pagination (~5000 events, ~41000 markets).
    2. Runs the paper trader's scan_and_trade() to generate signals and simulate fills.
    3. If auto_trade is also enabled, generates live RF signals, checks exits on real
       positions, and places real orders through the Kalshi API.
    4. Logs all activity to _cache for the frontend to display.
    5. Persists paper state and scan logs to Supabase.

    Maintains separate logs for shadow (paper) and live trades so the UI can show both.
    All exceptions are caught at the top level to prevent the scheduler from crashing.
    """
    if not kalshi or not config.validate_kalshi():
        return

    try:
        scan_start = datetime.now(timezone.utc)
        now = scan_start.isoformat()

        # Fetch ALL events from Kalshi (paginated, ~5000+ events)
        events = kalshi.get_all_events()
        total_events = len(events)
        total_markets = sum(len(e.markets) for e in events)

        logger.info(f"Full Kalshi scan: {total_events} events, {total_markets} markets")

        # ── Shadow (paper) trading scan ──
        # Paper trading always runs when the model is trained. It simulates trades
        # using real market data without placing actual orders on Kalshi.
        paper_entries = []
        paper_exits = []
        if paper_trader and paper_trader.generator.model.is_trained:
            result = paper_trader.scan_and_trade(events)
            paper_trader.save_state()
            paper_entries = result.get("entries", [])
            paper_exits = result.get("exits", [])
            open_pos = result.get("open_positions", 0)

            # Log each shadow trade individually
            for entry in paper_entries:
                _cache["paper_trade_log"].append({
                    "time": now,
                    "action": "entry",
                    "ticker": entry.get("ticker", ""),
                    "side": entry.get("side", ""),
                    "price": entry.get("entry_price", 0),
                    "contracts": entry.get("contracts", 1),
                    "model_prob": entry.get("model_prob", 0),
                    "edge": entry.get("edge", 0),
                })
            for ex in paper_exits:
                _cache["paper_trade_log"].append({
                    "time": now,
                    "action": "exit",
                    "ticker": ex.get("ticker", ""),
                    "side": ex.get("side", ""),
                    "entry_price": ex.get("entry_price", 0),
                    "exit_price": ex.get("exit_price", 0),
                    "pnl_cents": ex.get("pnl_cents", 0),
                })

            scan_entry = {
                "time": now, "type": "paper",
                "events_scanned": total_events,
                "markets_scanned": total_markets,
                "entries": len(paper_entries), "exits": len(paper_exits),
                "open_positions": open_pos,
            }
            _cache["auto_scan_log"].append(scan_entry)
            if db:
                db.insert_scan_log("paper", total_markets, len(paper_entries), len(paper_exits), open_pos)

        # ── Live trading scan (only when auto_trade is explicitly enabled) ──
        # Live trading is a separate opt-in: auto_scan must be on AND auto_trade
        # must be explicitly toggled. This two-step design prevents accidental
        # real-money trades when the user only wants paper trading.
        if _cache.get("auto_trade_enabled") and rf_generator and risk_manager:
            rf_signals = rf_generator.generate_signals(events)
            portfolio = kalshi.get_portfolio_summary()
            positions = kalshi.get_positions()
            exit_signals = rf_generator.check_exits(events, positions)

            # Auto-execute exit signals
            for sig in exit_signals:
                try:
                    order = OrderRequest(
                        ticker=sig.ticker,
                        side=Side(sig.side.value),
                        price_cents=int(sig.market_probability * 100),
                        count=1,
                        action="sell",
                    )
                    kalshi.place_order(order)
                    risk_manager.record_trade()
                    _cache["live_trade_log"].append({
                        "time": now, "action": "exit",
                        "ticker": sig.ticker, "side": sig.side.value,
                        "price": int(sig.market_probability * 100),
                    })
                    logger.info(f"Auto-exit: {sig.ticker}")
                except Exception as e:
                    logger.error(f"Auto-exit failed for {sig.ticker}: {e}")

            # Auto-execute entry signals that pass risk check
            for sig in rf_signals:
                allowed, reason = risk_manager.check_signal(sig, portfolio)
                if not allowed:
                    continue
                try:
                    order = risk_manager.build_order(sig)
                    kalshi.place_order(order)
                    risk_manager.record_trade()
                    _cache["live_trade_log"].append({
                        "time": now, "action": "entry",
                        "ticker": sig.ticker, "side": sig.side.value,
                        "price": order.price_cents,
                        "count": order.count,
                        "edge": round(sig.edge, 4),
                    })
                    if notifier and notifier.is_configured:
                        notifier.notify_trade_entry(sig.ticker, sig.side.value, sig.market_probability, order.price_cents)
                    logger.info(f"Auto-entry: {sig.ticker} ({sig.side.value})")
                except Exception as e:
                    logger.error(f"Auto-entry failed for {sig.ticker}: {e}")

            _cache["auto_scan_log"].append({
                "time": now, "type": "live",
                "events_scanned": total_events,
                "markets_scanned": total_markets,
                "signals": len(rf_signals),
                "exits": len(exit_signals),
            })

        # Update cache
        _cache["last_scan"] = now
        elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
        logger.info(f"Scan complete: {total_events} events, {total_markets} markets in {elapsed:.1f}s | paper: +{len(paper_entries)}/-{len(paper_exits)}")

        # Trim logs to prevent unbounded memory growth — keep only the most recent 500 entries
        for key in ("auto_scan_log", "paper_trade_log", "live_trade_log"):
            if len(_cache[key]) > 500:
                _cache[key] = _cache[key][-500:]

    except Exception as e:
        logger.error(f"Auto-scan failed: {e}")


async def _retrain_job():
    """Scheduled job to retrain the paper trader's model on fresh settled market data.

    Runs on the cron schedule defined by RETRAIN_DAYS and RETRAIN_HOUR config
    (default: Mon/Wed/Fri at 3am UTC). Fetches up to 1000 settled markets from
    Kalshi, merges them into the cumulative TrainingDataStore, and retrains the
    RF+GB ensemble. Sends a notification on completion and logs the run to Supabase.
    """
    if not kalshi or not config.validate_kalshi() or not paper_trader:
        return
    try:
        fetcher = HistoricalDataFetcher(kalshi)
        settled = fetcher.fetch_settled_markets(limit=1000)
        result = paper_trader.train_model(settled)  # Cumulative — merges with existing data
        total = result.get("total_cumulative_samples", len(settled))
        new = result.get("new_samples_added", len(settled))
        logger.info(f"Scheduled retrain: {total} total samples ({new} new), {result}")
        if notifier and notifier.is_configured:
            notifier.notify_retrain(total, result.get("cv_accuracy", 0))
        if db:
            db.insert_training_run(
                samples=len(settled),
                cv_accuracy=result.get("cv_accuracy", 0),
                oob_score=result.get("oob_score", 0),
                n_features=result.get("n_features", 0),
                n_estimators=result.get("n_estimators", 0),
                feature_importance=result.get("feature_importance", {}),
            )
    except Exception as e:
        logger.error(f"Scheduled retrain failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager: initializes all services on startup, cleans up on shutdown.

    Startup sequence:
      1. Initialize Database (Supabase), KalshiClient, DraftKingsClient, RFSignalGenerator,
         RiskManager, PerformanceTracker, PaperTrader, and Notifier.
      2. Restore paper trader state from Supabase (or local JSON fallback).
      3. Optionally initialize MarketAnalyzer if Anthropic API key is configured.
      4. Start APScheduler with a cron trigger for model retraining.

    Shutdown sequence:
      1. Stop the APScheduler.
      2. Save paper trader state to persist positions and metrics.
      3. Close all HTTP client connections (Notifier, KalshiClient, DraftKingsClient).
    """
    global kalshi, dk, rf_generator, ai_analyzer, risk_manager, performance, paper_trader, db, scheduler, notifier
    db = Database()
    kalshi = KalshiClient()
    dk = DraftKingsClient()
    rf_generator = RFSignalGenerator()
    risk_manager = RiskManager()
    performance = PerformanceTracker(db=db, mode="live")
    paper_trader = PaperTrader(db=db)
    paper_trader.load_state()  # Restore from DB or disk
    notifier = Notifier()

    if config.validate_anthropic():
        ai_analyzer = MarketAnalyzer()

    # Start APScheduler for background jobs (auto-scan and model retraining)
    scheduler = AsyncIOScheduler()

    # Add retrain schedule (default: Mon/Wed/Fri at 3am UTC)
    # The cron trigger uses APScheduler's day_of_week format (mon,wed,fri)
    if config.retrain_days:
        day_map = {"mon": "mon", "tue": "tue", "wed": "wed", "thu": "thu", "fri": "fri", "sat": "sat", "sun": "sun"}
        days = ",".join(day_map.get(d.strip().lower(), d.strip()) for d in config.retrain_days.split(","))
        scheduler.add_job(
            _retrain_job,
            CronTrigger(day_of_week=days, hour=config.retrain_hour),
            id="retrain",
            replace_existing=True,
        )

    scheduler.start()

    yield

    # Shutdown
    if scheduler:
        scheduler.shutdown(wait=False)
    if paper_trader:
        paper_trader.save_state()  # Persist on shutdown
    if notifier:
        notifier.close()
    if kalshi:
        kalshi.close()
    if dk:
        dk.close()


# ── FastAPI App Initialization ────────────────────────────────────────────────

app = FastAPI(title="Prediction Market Bot", version="1.0.0", lifespan=lifespan)

# CORS middleware allows the React frontend (on Vercel/localhost:5173) to call
# this backend (on Railway/localhost:8000). allow_origins=["*"] is permissive;
# tighten this in production to the actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────────────
# Pydantic models for endpoint request bodies. These define the JSON schema
# that the frontend sends to each endpoint.

class ScanRequest(BaseModel):
    """Request body for POST /api/scan."""
    use_ai: bool = False       # If true, also run Claude AI analysis alongside RF model


class TradeRequest(BaseModel):
    """Request body for POST /api/trade (manual order placement)."""
    ticker: str                # Kalshi market ticker (e.g., "KXBTC-24MAR14-T100000")
    side: str = "yes"          # "yes" or "no"
    price_cents: int = 50      # Limit price in cents (1-99)
    count: int = 1             # Number of contracts


class ConfigUpdate(BaseModel):
    """Request body for PATCH /api/config (partial config update)."""
    max_bet_amount_cents: int | None = None
    min_edge_threshold: float | None = None
    max_daily_loss_cents: int | None = None
    max_open_positions: int | None = None
    max_events_to_analyze: int | None = None
    kelly_fraction: float | None = None


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Get bot status, connection state, configuration, and model info.

    Returns a comprehensive status object used by the frontend Dashboard tab to
    display connection indicators, config values, model state, paper trader state,
    and auto-scan/trade toggle states.
    """
    return {
        "kalshi_connected": config.validate_kalshi(),
        "anthropic_connected": config.validate_anthropic(),
        "supabase_connected": db.is_connected if db else False,
        "environment": "demo" if config.kalshi_use_demo else "production",
        "config": {
            "max_bet_amount_cents": config.max_bet_amount_cents,
            "min_edge_threshold": config.min_edge_threshold,
            "max_daily_loss_cents": config.max_daily_loss_cents,
            "max_open_positions": config.max_open_positions,
            "max_events_to_analyze": config.max_events_to_analyze,
            "kelly_fraction": config.kelly_fraction,
        },
        "model": {
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "n_estimators": rf_generator.model.n_estimators if rf_generator else 0,
            "is_trained": rf_generator.model.is_trained if rf_generator else False,
        },
        "paper_trader": {
            "initialized": paper_trader is not None,
            "model_trained": paper_trader.generator.model.is_trained if paper_trader else False,
            "training_samples": paper_trader.training_store.count if paper_trader else 0,
            "balance_cents": paper_trader.balance_cents if paper_trader else 0,
            "open_positions": len(paper_trader.positions) if paper_trader else 0,
            "total_scans": paper_trader.total_scans if paper_trader else 0,
        },
        "auto_scan_enabled": _cache.get("auto_scan_enabled", False),
        "auto_trade_enabled": _cache.get("auto_trade_enabled", False),
        "last_scan": _cache.get("last_scan"),
    }


@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio summary."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    try:
        summary = kalshi.get_portfolio_summary()
        return summary.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/positions")
async def get_positions():
    """Get open positions enriched with current market prices and unrealized P&L.

    For each position, fetches the current market data to calculate the live
    unrealized P&L based on the difference between current ask price and entry price.
    """
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    try:
        positions = kalshi.get_positions()
        result = []
        for pos in positions:
            market = kalshi.get_market(pos.ticker)
            pos_dict = pos.model_dump()
            if market:
                pos_dict["current_price_cents"] = market.yes_ask if pos.side == "yes" else market.no_ask
                pos_dict["market_title"] = market.title
                pos_dict["unrealized_pnl_cents"] = (
                    (pos_dict["current_price_cents"] - pos.avg_price_cents) * pos.quantity
                    if pos.side == "yes" else
                    (pos.avg_price_cents - pos_dict["current_price_cents"]) * pos.quantity
                )
            result.append(pos_dict)
        return {"positions": result}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/scan")
async def run_scan(req: ScanRequest):
    """Run an on-demand market scan triggered by the frontend Signals tab.

    Pipeline:
      1. Fetch ALL events from Kalshi using cursor-based pagination.
      2. Generate entry signals using the RF+GB ensemble model.
      3. Optionally generate additional signals using Claude AI (if use_ai=true).
      4. Check open positions for exit signals (target hit or expiry approaching).
      5. Validate all signals through the RiskManager (balance, limits, edge checks).
      6. Cache results for the GET /api/signals endpoint.

    Returns signal counts and the full list of validated signals for the frontend.
    """
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")

    try:
        # Fetch ALL events from Kalshi
        events = kalshi.get_all_events()
        total_markets = sum(len(e.markets) for e in events)

        # Generate RF signals (guide logic)
        rf_signals = rf_generator.generate_signals(events) if rf_generator else []

        # Optionally enhance with Claude AI
        ai_signals = []
        if req.use_ai and ai_analyzer:
            ai_signals = ai_analyzer.analyze_events(events)

        # Check exit signals for open positions
        positions = kalshi.get_positions()
        exit_signals = rf_generator.check_exits(events, positions) if rf_generator else []

        # Get portfolio for risk checks
        portfolio = kalshi.get_portfolio_summary()

        # Validate signals through risk manager
        validated_signals = []
        for sig in rf_signals:
            allowed, reason = risk_manager.check_signal(sig, portfolio)
            sig_dict = sig.model_dump()
            sig_dict["risk_check"] = {"allowed": allowed, "reason": reason}
            sig_dict["source"] = "random_forest"
            validated_signals.append(sig_dict)

        for sig in ai_signals:
            allowed, reason = risk_manager.check_signal(sig, portfolio)
            sig_dict = sig.model_dump()
            sig_dict["risk_check"] = {"allowed": allowed, "reason": reason}
            sig_dict["source"] = "claude_ai"
            validated_signals.append(sig_dict)

        # Cache results
        from datetime import datetime, timezone
        _cache["events"] = [e.model_dump() for e in events]
        _cache["signals"] = validated_signals
        _cache["exit_signals"] = [s.model_dump() for s in exit_signals]
        _cache["last_scan"] = datetime.now(timezone.utc).isoformat()

        return {
            "events_scanned": len(events),
            "markets_scanned": total_markets,
            "rf_signals": len(rf_signals),
            "ai_signals": len(ai_signals),
            "exit_signals": len(exit_signals),
            "signals": validated_signals,
            "exit_signals_data": [s.model_dump() for s in exit_signals],
            "scan_time": _cache["last_scan"],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/signals")
async def get_cached_signals():
    """Get the most recent scan results."""
    return {
        "signals": _cache.get("signals", []),
        "exit_signals": _cache.get("exit_signals", []),
        "last_scan": _cache.get("last_scan"),
    }


@app.get("/api/events")
async def get_events(limit: int = 20):
    """Get top events from Kalshi."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    try:
        events = kalshi.get_events(limit=limit)
        return {"events": [e.model_dump() for e in events]}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/market/{ticker}")
async def get_market(ticker: str):
    """Get detailed market data with RF model analysis for a specific ticker.

    Returns the raw market data plus model analysis including: model probability,
    edge, whether entry/exit signals are active, all 106 features, and the
    entry/exit price thresholds computed from the guide's rules.
    """
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")

    market = kalshi.get_market(ticker)
    if not market:
        raise HTTPException(404, f"Market {ticker} not found")

    # Get parent event for feature extraction
    from bot.models import Event
    event = Event(event_ticker=market.event_ticker, title=market.title)

    features = extract_features(market, event)
    model_prob = rf_generator.model.predict_probability(features) if rf_generator else 0.5
    market_price = market.mid_price_yes / 100

    # Guide entry/exit checks:
    # Entry: market is trading at half or less of model's fair value (2x undervalued)
    # Exit: market has corrected to 90% of model's fair value
    entry_signal = market_price <= model_prob * 0.5
    exit_signal = market_price >= model_prob * 0.9

    return {
        "market": market.model_dump(),
        "analysis": {
            "model_probability": round(model_prob, 4),
            "market_probability": round(market_price, 4),
            "edge": round(model_prob - market_price, 4),
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "entry_threshold": round(model_prob * 0.5, 4),
            "exit_threshold": round(model_prob * 0.9, 4),
            "features": {k: round(v, 4) for k, v in features.items()},
        },
    }


@app.post("/api/trade")
async def place_trade(req: TradeRequest):
    """Place a trade on Kalshi."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")

    order = OrderRequest(
        ticker=req.ticker,
        side=Side(req.side),
        price_cents=req.price_cents,
        count=req.count,
    )

    try:
        result = kalshi.place_order(order)
        risk_manager.record_trade()
        return result.model_dump()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.delete("/api/order/{order_id}")
async def cancel_order(order_id: str):
    """Cancel an open order."""
    if not kalshi:
        raise HTTPException(400, "Kalshi not connected")
    success = kalshi.cancel_order(order_id)
    return {"cancelled": success}


@app.get("/api/orders")
async def get_orders():
    """Get open orders."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    try:
        orders = kalshi.get_open_orders()
        return {"orders": orders}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Performance Endpoints ────────────────────────────────────────────────────

@app.get("/api/performance")
async def get_performance():
    """Get performance metrics (Sharpe Ratio, win rate, etc.)."""
    if not performance:
        return {}
    metrics = performance.get_metrics()
    return {
        "metrics": {
            "total_trades": metrics.total_trades,
            "wins": metrics.wins,
            "losses": metrics.losses,
            "win_rate": round(metrics.win_rate, 4),
            "total_pnl_cents": metrics.total_pnl_cents,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sharpe_label": metrics.sharpe_label,
            "avg_log_return": metrics.avg_log_return,
            "avg_mae": metrics.avg_mae,
            "avg_mfe": metrics.avg_mfe,
            "avg_edge": metrics.avg_edge,
            "best_trade_pnl": metrics.best_trade_pnl,
            "worst_trade_pnl": metrics.worst_trade_pnl,
            "profit_factor": metrics.profit_factor,
            "max_drawdown_cents": metrics.max_drawdown_cents,
        },
        "equity_curve": performance.get_equity_curve(),
        "trades": performance.get_trade_history(),
    }


@app.get("/api/model/features")
async def get_feature_importance():
    """Get Random Forest feature importance rankings."""
    if not rf_generator:
        return {"features": {}, "feature_names": FEATURE_NAMES}
    importance = rf_generator.model.get_feature_importance()
    return {
        "features": importance,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "is_trained": rf_generator.model.is_trained,
    }


# ── Arbitrage Endpoints ─────────────────────────────────────────────────────

@app.get("/api/arbitrage")
async def scan_arbitrage():
    """Scan for cross-platform arbitrage opportunities."""
    if not kalshi or not dk:
        raise HTTPException(400, "Clients not initialized")

    try:
        events = kalshi.get_all_events()
        kalshi_markets = [m for e in events for m in e.markets if m.status == "open"]
        dk_markets = dk.get_prediction_markets()

        opportunities = detect_arbitrage(kalshi_markets, dk_markets)

        return {
            "kalshi_markets": len(kalshi_markets),
            "dk_markets": len(dk_markets),
            "opportunities": [
                {
                    "kalshi_ticker": o.kalshi_ticker,
                    "dk_market_id": o.dk_market_id,
                    "title": o.title,
                    "kalshi_yes_price": o.kalshi_yes_price,
                    "dk_yes_price": o.dk_yes_price,
                    "spread_pct": o.spread_pct,
                    "recommended_action": o.recommended_action,
                }
                for o in opportunities
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.patch("/api/config")
async def update_config(update: ConfigUpdate):
    """Update trading configuration."""
    if update.max_bet_amount_cents is not None:
        config.max_bet_amount_cents = update.max_bet_amount_cents
    if update.min_edge_threshold is not None:
        config.min_edge_threshold = update.min_edge_threshold
    if update.max_daily_loss_cents is not None:
        config.max_daily_loss_cents = update.max_daily_loss_cents
    if update.max_open_positions is not None:
        config.max_open_positions = update.max_open_positions
    if update.max_events_to_analyze is not None:
        config.max_events_to_analyze = update.max_events_to_analyze
    if update.kelly_fraction is not None:
        config.kelly_fraction = update.kelly_fraction
    return {"status": "updated", "config": {
        "max_bet_amount_cents": config.max_bet_amount_cents,
        "min_edge_threshold": config.min_edge_threshold,
        "max_daily_loss_cents": config.max_daily_loss_cents,
        "max_open_positions": config.max_open_positions,
        "max_events_to_analyze": config.max_events_to_analyze,
        "kelly_fraction": config.kelly_fraction,
    }}


# ── Backtest Endpoints ──────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    """Request body for POST /api/backtest (single backtest run)."""
    initial_balance_cents: int = 100_00   # Starting paper balance ($100)
    max_bet_cents: int = 25_00            # Max bet per trade ($25)
    entry_threshold: float = 0.5          # Entry: market <= model * threshold
    exit_threshold: float = 0.9           # Exit: market >= model * threshold
    min_confidence: float = 0.70          # Minimum model confidence to trade
    min_volume: int = 50                  # Minimum market volume to consider
    train_ratio: float = 0.6             # 60% train / 40% test split
    max_markets: int = 200                # Max settled markets to fetch from Kalshi


class SweepRequest(BaseModel):
    """Request body for POST /api/backtest/sweep (parameter optimization grid search)."""
    entry_thresholds: list[float] = [0.4, 0.45, 0.5, 0.55, 0.6]   # Entry threshold values to test
    confidence_levels: list[float] = [0.60, 0.65, 0.70, 0.75, 0.80]  # Confidence values to test
    max_markets: int = 200                # Max settled markets to fetch


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    """Run a backtest on historical settled markets."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")

    try:
        fetcher = HistoricalDataFetcher(kalshi)
        settled = fetcher.fetch_settled_markets(limit=req.max_markets)

        bt = Backtester(kalshi)
        cfg = BacktestConfig(
            initial_balance_cents=req.initial_balance_cents,
            max_bet_cents=req.max_bet_cents,
            entry_threshold=req.entry_threshold,
            exit_threshold=req.exit_threshold,
            min_confidence=req.min_confidence,
            min_volume=req.min_volume,
            train_ratio=req.train_ratio,
        )
        result = bt.run(settled, cfg)

        return {
            "config": result.config,
            "train_samples": result.train_samples,
            "test_samples": result.test_samples,
            "cv_accuracy": round(result.cv_accuracy, 4),
            "oob_score": round(result.oob_score, 4),
            "total_trades": result.total_trades,
            "wins": result.wins,
            "losses": result.losses,
            "win_rate": round(result.win_rate, 4),
            "total_pnl_cents": result.total_pnl_cents,
            "sharpe_ratio": result.sharpe_ratio,
            "sharpe_label": result.sharpe_label,
            "profit_factor": result.profit_factor,
            "max_drawdown_cents": result.max_drawdown_cents,
            "avg_edge": round(result.avg_edge, 4),
            "avg_log_return": round(result.avg_log_return, 4),
            "signals_generated": result.signals_generated,
            "signals_filtered": result.signals_filtered,
            "trades": result.trades[:50],  # Limit for UI
            "equity_curve": result.equity_curve,
            "feature_importance": result.feature_importance,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/backtest/sweep")
async def run_parameter_sweep(req: SweepRequest):
    """Run parameter sweep to find optimal settings."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")

    try:
        fetcher = HistoricalDataFetcher(kalshi)
        settled = fetcher.fetch_settled_markets(limit=req.max_markets)

        bt = Backtester(kalshi)
        results = bt.parameter_sweep(settled, req.entry_thresholds, req.confidence_levels)

        return {
            "total_combinations": len(results),
            "results": [
                {
                    "config": r.config,
                    "total_trades": r.total_trades,
                    "win_rate": round(r.win_rate, 4),
                    "sharpe_ratio": r.sharpe_ratio,
                    "total_pnl_cents": r.total_pnl_cents,
                    "profit_factor": r.profit_factor,
                    "max_drawdown_cents": r.max_drawdown_cents,
                }
                for r in results[:25]  # Top 25 by Sharpe
            ],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Paper Trading Endpoints ─────────────────────────────────────────────────

class PaperTradeConfig(BaseModel):
    """Request body for POST /api/paper/configure (reset paper trader)."""
    balance_cents: int = 100_00  # Starting balance in cents ($100)


class AddFundsRequest(BaseModel):
    """Request body for POST /api/paper/add-funds (top up without resetting)."""
    amount_cents: int  # Amount to add to paper balance


@app.get("/api/paper")
async def get_paper_state():
    """Get paper trading state."""
    if not paper_trader:
        return {}
    return paper_trader.get_state()


@app.post("/api/paper/configure")
async def configure_paper(req: PaperTradeConfig):
    """Reset and configure paper trader."""
    global paper_trader
    paper_trader = PaperTrader(db=db)
    paper_trader.configure(req.balance_cents)
    return paper_trader.get_state()


@app.post("/api/risk/reset-daily")
async def reset_daily_pnl():
    """Reset the daily P&L counter on the risk manager."""
    if risk_manager:
        risk_manager.reset_daily()
    return {"status": "reset", "daily_pnl_cents": 0, "trades_today": 0}


@app.post("/api/paper/add-funds")
async def add_paper_funds(req: AddFundsRequest):
    """Add demo funds to paper trading balance without resetting state."""
    if not paper_trader:
        raise HTTPException(400, "Paper trader not initialized")
    if req.amount_cents <= 0:
        raise HTTPException(400, "Amount must be positive")
    new_balance = paper_trader.add_funds(req.amount_cents)
    return {"balance_cents": new_balance, "added_cents": req.amount_cents}


@app.post("/api/paper/scan")
async def paper_scan():
    """Run a paper trading scan cycle."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    if not paper_trader:
        raise HTTPException(400, "Paper trader not initialized")

    try:
        events = kalshi.get_all_events()
        result = paper_trader.scan_and_trade(events)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/paper/train")
async def paper_train():
    """Train the paper trader's model on historical settled market data.

    Fetches up to 1000 settled markets from Kalshi, extracts features from
    trade history, and trains the RF+GB ensemble cumulatively (new samples are
    merged with all previously seen data, deduplicated by ticker).
    """
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    if not paper_trader:
        raise HTTPException(400, "Paper trader not initialized")

    try:
        fetcher = HistoricalDataFetcher(kalshi)
        settled = fetcher.fetch_settled_markets(limit=1000)
        result = paper_trader.train_model(settled)
        return {
            "status": "trained",
            "samples_fetched": len(settled),
            "total_cumulative_samples": result.get("total_cumulative_samples", len(settled)),
            "new_samples_added": result.get("new_samples_added", len(settled)),
            **result,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Auto-Scan / Auto-Trade Scheduler ──────────────────────────────────────

class AutoScanConfig(BaseModel):
    """Request body for POST /api/autoscan (toggle background scanning)."""
    enabled: bool = True           # Enable or disable the background scan loop
    interval_seconds: int = 60     # Seconds between scans (default: scan every 60s)


class AutoTradeConfig(BaseModel):
    """Request body for POST /api/autotrade (toggle live order execution)."""
    enabled: bool = True  # When true AND auto_scan is on, bot places real Kalshi orders


@app.post("/api/autoscan")
async def toggle_auto_scan(req: AutoScanConfig):
    """Enable/disable automatic background scanning every N seconds.

    When enabled, adds an APScheduler interval job that runs _auto_scan_job()
    every interval_seconds (default 60). When disabled, removes the job.
    The job scans all ~41000 Kalshi markets and runs paper + optionally live trading.
    """
    global scheduler
    if not scheduler:
        raise HTTPException(500, "Scheduler not initialized")

    job_id = "auto_scan"
    existing = scheduler.get_job(job_id)

    if req.enabled:
        if existing:
            scheduler.reschedule_job(job_id, trigger="interval", seconds=req.interval_seconds)
        else:
            scheduler.add_job(
                _auto_scan_job,
                "interval",
                seconds=req.interval_seconds,
                id=job_id,
                replace_existing=True,
            )
        _cache["auto_scan_enabled"] = True
    else:
        if existing:
            scheduler.remove_job(job_id)
        _cache["auto_scan_enabled"] = False

    return {
        "auto_scan_enabled": _cache["auto_scan_enabled"],
        "interval_seconds": req.interval_seconds,
    }


@app.post("/api/autotrade")
async def toggle_auto_trade(req: AutoTradeConfig):
    """Enable/disable automatic live trading (requires auto-scan to be on)."""
    _cache["auto_trade_enabled"] = req.enabled
    return {
        "auto_trade_enabled": _cache["auto_trade_enabled"],
        "auto_scan_enabled": _cache.get("auto_scan_enabled", False),
    }


@app.get("/api/autoscan/status")
async def get_auto_scan_status():
    """Get auto-scan scheduler status and recent log."""
    return {
        "auto_scan_enabled": _cache.get("auto_scan_enabled", False),
        "auto_trade_enabled": _cache.get("auto_trade_enabled", False),
        "last_scan": _cache.get("last_scan"),
        "log": _cache.get("auto_scan_log", [])[-20:],
    }


@app.get("/api/trades/shadow")
async def get_shadow_trade_log(limit: int = 200):
    """Get the shadow (paper) trade log — every entry/exit the bot made."""
    log = _cache.get("paper_trade_log", [])
    # Also include historical trades from the paper trader itself
    trades = []
    if paper_trader:
        for t in paper_trader.tracker.get_trade_history():
            trades.append({
                "time": t.get("exit_time", t.get("entry_time", "")),
                "action": "closed",
                "ticker": t.get("ticker", ""),
                "side": t.get("side", ""),
                "entry_price": t.get("entry_price", 0),
                "exit_price": t.get("exit_price", 0),
                "pnl_cents": t.get("pnl_cents", 0),
                "won": t.get("won", False),
                "log_return": t.get("log_return", 0),
                "category": t.get("category", ""),
            })
    return {
        "live_log": log[-limit:],
        "completed_trades": trades[-limit:],
        "total_shadow_trades": len(trades),
        "open_positions": [
            {
                "ticker": p.ticker,
                "side": p.side,
                "entry_price": p.entry_price,
                "contracts": p.contracts,
                "model_prob": p.model_prob,
                "entry_time": p.entry_time,
            }
            for p in (paper_trader.positions.values() if paper_trader else [])
        ],
    }


@app.get("/api/trades/live")
async def get_live_trade_log(limit: int = 200):
    """Get the live trade log — every real entry/exit executed on Kalshi."""
    log = _cache.get("live_trade_log", [])
    # Also include historical trades from the live performance tracker
    trades = []
    if performance:
        for t in performance.get_trade_history():
            trades.append({
                "time": t.get("exit_time", t.get("entry_time", "")),
                "action": "closed",
                "ticker": t.get("ticker", ""),
                "side": t.get("side", ""),
                "entry_price": t.get("entry_price", 0),
                "exit_price": t.get("exit_price", 0),
                "pnl_cents": t.get("pnl_cents", 0),
                "won": t.get("won", False),
                "log_return": t.get("log_return", 0),
                "category": t.get("category", ""),
            })
    return {
        "live_log": log[-limit:],
        "completed_trades": trades[-limit:],
        "total_live_trades": len(trades),
    }


# ── Database History Endpoints ─────────────────────────────────────────────

@app.get("/api/history/trades")
async def get_trade_history(mode: str = "paper", limit: int = 100):
    """Get persistent trade history from Supabase."""
    if not db or not db.is_connected:
        return {"trades": [], "source": "none"}
    return {"trades": db.get_trades(mode=mode, limit=limit), "source": "supabase"}


@app.get("/api/history/scans")
async def get_scan_history(limit: int = 50):
    """Get scan log history from Supabase."""
    if not db or not db.is_connected:
        return {"scans": [], "source": "none"}
    return {"scans": db.get_scan_logs(limit=limit), "source": "supabase"}


@app.get("/api/history/performance")
async def get_performance_history(mode: str = "paper", limit: int = 50):
    """Get performance snapshots over time from Supabase."""
    if not db or not db.is_connected:
        return {"snapshots": [], "source": "none"}
    return {"snapshots": db.get_performance_history(mode=mode, limit=limit), "source": "supabase"}


@app.get("/api/history/training")
async def get_training_history(limit: int = 10):
    """Get model training history from Supabase."""
    if not db or not db.is_connected:
        return {"runs": [], "source": "none"}
    return {"runs": db.get_training_history(limit=limit), "source": "supabase"}


# ── Webhook Receiver ──────────────────────────────────────────────────────

class WebhookPayload(BaseModel):
    """Request body for POST /api/webhook (external trigger from CI/cron/etc)."""
    action: str = "scan"   # "scan" to trigger a market scan, "retrain" to retrain the model
    params: dict = {}      # Optional parameters for the action (currently unused)


@app.post("/api/webhook")
async def receive_webhook(
    payload: WebhookPayload,
    x_webhook_secret: str | None = Header(None),
):
    """External webhook to trigger scans or retrains."""
    if config.webhook_secret and x_webhook_secret != config.webhook_secret:
        raise HTTPException(403, "Invalid webhook secret")

    if payload.action == "scan":
        if not kalshi or not config.validate_kalshi():
            raise HTTPException(400, "Kalshi not connected")
        events = kalshi.get_all_events()
        if paper_trader and paper_trader.generator.model.is_trained:
            result = paper_trader.scan_and_trade(events)
            return {"action": "scan", "result": result}
        return {"action": "scan", "result": "model not trained"}

    elif payload.action == "retrain":
        await _retrain_job()
        return {"action": "retrain", "status": "completed"}

    raise HTTPException(400, f"Unknown action: {payload.action}")


# ── Notification Endpoints ────────────────────────────────────────────────

@app.post("/api/notifications/test")
async def test_notifications():
    """Send a test notification to all configured channels."""
    if not notifier or not notifier.is_configured:
        return {"status": "not_configured", "channels": []}
    result = notifier.send_test()
    return {"status": "sent", **result}


@app.get("/api/notifications/config")
async def get_notification_config():
    """Get notification configuration status."""
    return {
        "slack_configured": bool(config.slack_webhook_url),
        "discord_configured": bool(config.discord_webhook_url),
        "channels": notifier.channels if notifier else [],
    }


# ── CSV Export ────────────────────────────────────────────────────────────

@app.get("/api/export/trades")
async def export_trades_csv(source: str = "paper"):
    """Export trade history as a CSV file download.

    Generates a CSV with all trade fields (ticker, side, entry/exit prices,
    P&L, log return, MAE/MFE, etc.) and returns it as a streaming response
    with a Content-Disposition header for browser download.
    """
    trades = []
    if source == "paper" and paper_trader:
        trades = paper_trader.tracker.get_trade_history()
    elif source == "live" and performance:
        trades = performance.get_trade_history()

    if not trades:
        raise HTTPException(404, "No trades to export")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=trades[0].keys())
    writer.writeheader()
    writer.writerows(trades)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=trades_{source}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.csv"},
    )


# ── Category P&L / Heatmap ───────────────────────────────────────────────

@app.get("/api/performance/by-category")
async def get_performance_by_category(source: str = "paper"):
    """Get P&L breakdown by market category."""
    if source == "paper" and paper_trader:
        return {"categories": paper_trader.tracker.get_metrics_by_category()}
    elif source == "live" and performance:
        return {"categories": performance.get_metrics_by_category()}
    return {"categories": {}}


@app.get("/api/portfolio/heatmap")
async def get_portfolio_heatmap():
    """Get portfolio positions grouped by category for heatmap visualization.

    Returns paper trading positions organized by market category (politics, crypto,
    sports, etc.) with position count and value data for the frontend heatmap chart.
    """
    if not paper_trader:
        return {"categories": {}}

    from collections import defaultdict
    cats = defaultdict(lambda: {"positions": [], "total_pnl_cents": 0, "count": 0})

    for ticker, pos in paper_trader.positions.items():
        cat = pos.category or "uncategorized"
        entry_val = pos.entry_price * 100 * pos.contracts
        cats[cat]["positions"].append({
            "ticker": ticker,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "contracts": pos.contracts,
            "category": cat,
            "entry_value_cents": int(entry_val),
        })
        cats[cat]["count"] += 1

    return {"categories": dict(cats)}


# ── Trade Notes ───────────────────────────────────────────────────────────

class TradeNotesRequest(BaseModel):
    """Request body for PATCH /api/trade/{index}/notes (trade journal annotation)."""
    notes: str  # Free-text notes to attach to a specific trade


@app.patch("/api/trade/{trade_index}/notes")
async def update_trade_notes(trade_index: int, req: TradeNotesRequest, source: str = "paper"):
    """Update notes on a specific trade by index."""
    tracker = None
    if source == "paper" and paper_trader:
        tracker = paper_trader.tracker
    elif source == "live" and performance:
        tracker = performance

    if not tracker:
        raise HTTPException(400, "Tracker not available")

    if tracker.update_trade_notes(trade_index, req.notes):
        return {"status": "updated", "trade_index": trade_index}
    raise HTTPException(404, f"Trade index {trade_index} not found")


# ── Retrain Schedule ──────────────────────────────────────────────────────

class RetrainScheduleRequest(BaseModel):
    """Request body for POST /api/retrain/schedule (update cron schedule)."""
    days: str = "mon,wed,fri"  # Comma-separated day abbreviations for APScheduler CronTrigger
    hour: int = 3              # Hour (UTC) to run the retrain job


@app.get("/api/retrain/schedule")
async def get_retrain_schedule():
    """Get current retrain schedule."""
    job = scheduler.get_job("retrain") if scheduler else None
    return {
        "days": config.retrain_days,
        "hour": config.retrain_hour,
        "active": job is not None,
        "next_run": str(job.next_run_time) if job else None,
    }


@app.post("/api/retrain/schedule")
async def update_retrain_schedule(req: RetrainScheduleRequest):
    """Update the retrain schedule."""
    if not scheduler:
        raise HTTPException(500, "Scheduler not initialized")

    config.retrain_days = req.days
    config.retrain_hour = req.hour

    day_map = {"mon": "mon", "tue": "tue", "wed": "wed", "thu": "thu", "fri": "fri", "sat": "sat", "sun": "sun"}
    days = ",".join(day_map.get(d.strip().lower(), d.strip()) for d in req.days.split(","))

    job = scheduler.get_job("retrain")
    if job:
        scheduler.reschedule_job("retrain", trigger=CronTrigger(day_of_week=days, hour=req.hour))
    else:
        scheduler.add_job(
            _retrain_job,
            CronTrigger(day_of_week=days, hour=req.hour),
            id="retrain",
            replace_existing=True,
        )

    return {"days": req.days, "hour": req.hour, "status": "updated"}


@app.post("/api/retrain/now")
async def retrain_now():
    """Trigger an immediate model retrain."""
    await _retrain_job()
    return {"status": "retrain_complete"}
