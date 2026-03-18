"""FastAPI server providing REST endpoints for the React UI."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
from bot.models import OrderRequest, Side, TradingSignal


# ── Shared State ─────────────────────────────────────────────────────────────

kalshi: KalshiClient | None = None
dk: DraftKingsClient | None = None
rf_generator: RFSignalGenerator | None = None
ai_analyzer: MarketAnalyzer | None = None
risk_manager: RiskManager | None = None
performance: PerformanceTracker | None = None
paper_trader: PaperTrader | None = None

# Cache for latest scan results
_cache: dict[str, Any] = {
    "events": [],
    "signals": [],
    "exit_signals": [],
    "arbitrage": [],
    "last_scan": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kalshi, dk, rf_generator, ai_analyzer, risk_manager, performance, paper_trader
    kalshi = KalshiClient()
    dk = DraftKingsClient()
    rf_generator = RFSignalGenerator()
    risk_manager = RiskManager()
    performance = PerformanceTracker()
    paper_trader = PaperTrader()
    if config.validate_anthropic():
        ai_analyzer = MarketAnalyzer()
    yield
    if kalshi:
        kalshi.close()
    if dk:
        dk.close()


app = FastAPI(title="Prediction Market Bot", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──────────────────────────────────────────────────

class ScanRequest(BaseModel):
    max_events: int = 20
    use_ai: bool = False  # Use Claude in addition to RF model


class TradeRequest(BaseModel):
    ticker: str
    side: str = "yes"
    price_cents: int = 50
    count: int = 1


class ConfigUpdate(BaseModel):
    max_bet_amount_cents: int | None = None
    min_edge_threshold: float | None = None
    max_daily_loss_cents: int | None = None
    max_open_positions: int | None = None


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Get bot status and configuration."""
    return {
        "kalshi_connected": config.validate_kalshi(),
        "anthropic_connected": config.validate_anthropic(),
        "environment": "demo" if config.kalshi_use_demo else "production",
        "config": {
            "max_bet_amount_cents": config.max_bet_amount_cents,
            "min_edge_threshold": config.min_edge_threshold,
            "max_daily_loss_cents": config.max_daily_loss_cents,
            "max_open_positions": config.max_open_positions,
            "max_events_to_analyze": config.max_events_to_analyze,
        },
        "model": {
            "n_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "n_estimators": rf_generator.model.n_estimators if rf_generator else 0,
            "is_trained": rf_generator.model.is_trained if rf_generator else False,
        },
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
    """Get open positions with current prices."""
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
    """
    Run market scan: fetch events, generate RF signals,
    optionally run AI analysis, check exits.
    """
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")

    try:
        # Fetch events
        events = kalshi.get_events(limit=req.max_events)
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
    """Get detailed market data with model analysis."""
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

    # Guide entry/exit checks
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
        events = kalshi.get_events(limit=config.max_events_to_analyze)
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
    return {"status": "updated", "config": {
        "max_bet_amount_cents": config.max_bet_amount_cents,
        "min_edge_threshold": config.min_edge_threshold,
        "max_daily_loss_cents": config.max_daily_loss_cents,
        "max_open_positions": config.max_open_positions,
    }}


# ── Backtest Endpoints ──────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    initial_balance_cents: int = 100_00
    max_bet_cents: int = 25_00
    entry_threshold: float = 0.5
    exit_threshold: float = 0.9
    min_confidence: float = 0.70
    min_volume: int = 50
    train_ratio: float = 0.6
    max_markets: int = 200


class SweepRequest(BaseModel):
    entry_thresholds: list[float] = [0.4, 0.45, 0.5, 0.55, 0.6]
    confidence_levels: list[float] = [0.60, 0.65, 0.70, 0.75, 0.80]
    max_markets: int = 200


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
    balance_cents: int = 100_00


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
    paper_trader = PaperTrader()
    paper_trader.configure(req.balance_cents)
    return paper_trader.get_state()


@app.post("/api/paper/scan")
async def paper_scan():
    """Run a paper trading scan cycle."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    if not paper_trader:
        raise HTTPException(400, "Paper trader not initialized")

    try:
        events = kalshi.get_events(limit=config.max_events_to_analyze)
        result = paper_trader.scan_and_trade(events)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/paper/train")
async def paper_train():
    """Train the paper trader's model on historical data."""
    if not kalshi or not config.validate_kalshi():
        raise HTTPException(400, "Kalshi not connected")
    if not paper_trader:
        raise HTTPException(400, "Paper trader not initialized")

    try:
        fetcher = HistoricalDataFetcher(kalshi)
        settled = fetcher.fetch_settled_markets(limit=200)
        result = paper_trader.train_model(settled)
        return {"status": "trained", "samples": len(settled), **result}
    except Exception as e:
        raise HTTPException(500, str(e))
