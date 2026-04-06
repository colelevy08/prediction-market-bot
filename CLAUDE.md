# Prediction Market Bot — Claude Code Guide

## What This Project Is
A fully-automated trading bot for Kalshi prediction markets. Trades BTC/ETH/XRP/SOL/DOGE/BNB/HYPE 15-minute binary crypto markets using live price signals + Kelly criterion sizing. Runs **fully local** for minimum latency to Kalshi API.

**Current mode: PRODUCTION** (`KALSHI_USE_DEMO=false` in `.env`) — real money.

---

## Running the Server

```bash
./botctl start
```

Or directly:
```bash
source .venv/bin/activate
uvicorn bot.server:app --reload --port 8000
```

Server runs at **http://localhost:8000**. Health check: `GET /api/status`

Frontend runs at **http://localhost:5173**:
```bash
cd frontend && npm run dev
```

---

## Bot CLI (./botctl)

The `./botctl` script is the fastest way to interact. Always use this first.

```bash
./botctl status          # Server health, mode, uptime, balance
./botctl start           # Start local server (background, logs → /tmp/predbot.log)
./botctl stop            # Kill local server
./botctl scan            # Trigger a manual RF scan
./botctl autoscan on     # Start background auto-scan loop
./botctl autoscan off    # Stop background auto-scan loop
./botctl autotrade on    # Enable live order execution
./botctl autotrade off   # Disable live order execution
./botctl positions       # Current open positions on Kalshi
./botctl portfolio       # Balance + all positions
./botctl trades          # Recent live trade log (last 20)
./botctl signals         # Latest trading signals from last scan
./botctl paper           # Paper trading stats
./botctl performance     # P&L, win rate, Sharpe ratio
./botctl arbitrage       # Cross-platform arbitrage opportunities
./botctl logs            # Tail server log
./botctl stop-trading    # Emergency stop (both loops)
```

---

## Key API Endpoints

All at `http://localhost:8000`.

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/api/status` | Health, mode, uptime, balance, flags |
| GET | `/api/portfolio` | Balance + open positions |
| GET | `/api/signals` | Latest signals from last scan |
| POST | `/api/scan` | Trigger manual scan |
| POST | `/api/autoscan` | `{"enabled": true/false}` |
| POST | `/api/autotrade` | `{"enabled": true/false}` |
| GET | `/api/history/trades` | Full trade history |
| GET | `/api/performance` | P&L, win rate, Sharpe ratio |
| GET | `/api/arbitrage` | Kalshi vs DraftKings opportunities |
| GET | `/api/paper` | Paper trading positions and log |
| POST | `/api/backtest` | Run historical backtest |
| GET | `/api/model/features` | RF model feature importance |
| PATCH | `/api/config` | Update trading config at runtime |
| GET | `/api/export/trades` | Download trade history as CSV |
| GET | `/api/crypto/monitor` | BTC/ETH/XRP markets by timeframe |
| GET | `/api/liquidity` | Per-coin fill rates + suggested bet cap (monitor when balance > $3K) |
| GET | `/api/rewards` | Kalshi VIP LP score + estimated maker reward payout (cached, updated every ~5 min) |

---

## Architecture

```
bot/
├── server.py        — FastAPI app (all API endpoints, background loops)
├── crypto_feed.py   — Live prices (CoinGecko) + funding rates (Binance)
├── rf_model.py      — Random Forest + Gradient Boosting ensemble (108 features)
├── analyzer.py      — Claude AI overlay for market analysis
├── risk_manager.py  — Pre-trade safety gates (Kelly, drawdown, position limits)
├── kalshi_client.py — Kalshi Trade API v2 client
├── config.py        — All settings from .env (singleton)
├── backtester.py    — Historical backtesting + paper trading
├── performance.py   — P&L tracking + Sharpe/win rate
├── database.py      — Supabase persistence layer
├── notifier.py      — Slack/Discord trade alerts
└── models.py        — Pydantic data models

frontend/            — React + Vite dashboard (localhost:5173)
  vite.config.js     — Proxies /api/* → localhost:8000
  .env.local         — VITE_API_URL=http://localhost:8000
```

---

## Crypto Trading Strategy (Active)

Trades Kalshi 15-minute BTC/ETH/XRP/SOL/DOGE/BNB/HYPE binary markets.

**Signal (W90):**
`fair_prob = sigmoid(0.90 × logit(p_price) + 0.10 × logit(market_prob))`
- `p_price`: t-distribution probability from live price momentum (CoinGecko)
- `market_prob`: Kalshi mid-price as probability

**Direction gate (WP55):** YES if `fair_prob ≥ 0.55`, NO if `fair_prob ≤ 0.45`, dead zone 0.45–0.55

**Entry requires:** edge ≥ 11¢ (`round(fair_prob×100) - ask_price ≥ 11`)

**Order type:** IOC limit order, price ceiling = `fair_cents - 4` (7¢ sweep range)

**Sizing:** K35 Kelly base × vol_scale × spread_adj × corr_deflator, capped at 10% of balance

**No stop-loss, no DCA, no hedging** — hold to settlement (confirmed optimal by shadow data)

**Triple-fill protection:** 5s post-fill cooldown + `actual_qty > 0` gate before retry

---

## Key Config (.env)

- `KALSHI_USE_DEMO=false` — **PRODUCTION, real money**
- `MAX_BET_AMOUNT_CENTS=1500` — floor bet size; dynamic cap = max(this, 10% of balance)
- `MAX_DAILY_LOSS_CENTS=3440` — ~$34 daily loss limit (scales via MAX_DRAWDOWN_PCT at runtime)
- `MAX_OPEN_POSITIONS=5` — max simultaneous open positions
- `AUTO_SCAN_INTERVAL=0.15` — 0.15s scan; portfolio cached at 1s → ~75ms avg decision latency
- `AUTO_FETCH_INTERVAL=1` — 1s REST fetch for portfolio; WS streams prices continuously
- `KELLY_FRACTION=0.15` — Kelly base (K35 live = 35% due to additional scaling)
- `LIVE_DEPLOY_PCT=1.00` — 100% capital deployment
- **Auto-shutdown floor:** $100 total portfolio (cash + open position cost basis)

---

## Common Tasks

```bash
# Start everything (autoscan ON by default; autotrade must be enabled manually)
./botctl start && ./botctl autotrade on

# Check what the bot is doing
./botctl trades && ./botctl positions

# Shadow tournament snapshot
python3 /tmp/shadow_snap.py

# Emergency stop
./botctl stop-trading
```

## Emergency Floor

If total portfolio value (cash + open positions) drops below **$100**, stop both loops:
```bash
curl -X POST http://localhost:8000/api/autotrade -H "Content-Type: application/json" -d '{"enabled": false}'
curl -X POST http://localhost:8000/api/autoscan  -H "Content-Type: application/json" -d '{"enabled": false}'
```
