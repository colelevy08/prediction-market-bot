<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Claude_AI-Sonnet_4-CC785C?style=for-the-badge&logo=anthropic&logoColor=white" />
  <img src="https://img.shields.io/badge/Kalshi-API_v2-00C853?style=for-the-badge" />
</p>

# PredictionBot

> **AI-powered prediction market trading system** that combines machine learning, live web research, and quantitative analysis to find and trade mispriced contracts on [Kalshi](https://kalshi.com).

PredictionBot scans **46,000+ markets** every 60 seconds using a three-layer intelligence stack:

1. **ML Ensemble** — Random Forest + Gradient Boosting model with 108 engineered features scores every market in <0.1s
2. **Claude AI + Web Search** — Feeds real market data to Claude Sonnet, which searches the web for breaking news, polls, and scores to find edges the model misses
3. **Quant Risk Engine** — Kelly-optimal sizing, drawdown control, correlation limits, and multi-leg exits protect capital

The system runs 24/7 with paper trading, live trading, email/Slack/Discord alerts, and a premium React dashboard deployed on Vercel + Railway.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Vercel)                               │
│   React 18 · Vite · Tailwind CSS · Recharts · 7 Dashboard Pages        │
│   Stock ticker · Glassmorphism UI · Real-time SSE updates               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ REST API + SSE
┌───────────────────────────────▼─────────────────────────────────────────┐
│                         BACKEND (Railway)                               │
│   FastAPI · 50+ endpoints · APScheduler · asyncio                       │
│                                                                         │
│   ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│   │  RF+GB Model  │  │  Claude AI Agent  │  │   Risk Manager         │   │
│   │  108 features │  │  Web search (3x)  │  │   Kelly sizing         │   │
│   │  Ensemble     │──│  Real market data │──│   Drawdown control     │   │
│   │  predictions  │  │  News synthesis   │  │   Position limits      │   │
│   └──────────────┘  └──────────────────┘  └────────────────────────┘   │
│                                                                         │
│   ┌──────────────┐  ┌──────────────────┐  ┌────────────────────────┐   │
│   │  Kalshi API   │  │  Supabase (PG)   │  │   Notifications        │   │
│   │  RSA-PSS auth │  │  Trade history   │  │   Email · Slack        │   │
│   │  Orders/Data  │  │  Scan logs       │  │   Discord              │   │
│   └──────────────┘  └──────────────────┘  └────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

| Layer | Technology |
|-------|-----------|
| **ML** | scikit-learn (RF + GB ensemble), 108 features, StandardScaler, CalibratedClassifierCV |
| **AI** | Anthropic Claude Sonnet 4 with web search tool — researches live news per market |
| **Backend** | FastAPI, uvicorn, APScheduler, httpx, Pydantic v2 |
| **Frontend** | React 18, Vite, Tailwind CSS, Recharts, SSE real-time updates |
| **Database** | Supabase PostgreSQL (optional) — persistent trade history, scans, snapshots |
| **Auth** | RSA-PSS signed requests to Kalshi Trade API v2 |
| **Notifications** | Email (SMTP), Slack webhooks, Discord webhooks |
| **Deploy** | Railway (backend), Vercel (frontend), auto-deploy on `git push` |

---

## How It Works

### The Three-Layer Signal Pipeline

```
46,000 markets → [RF Model: 0.1s] → Top candidates → [Claude AI + Web Search] → Signals → [Risk Engine] → Trades
```

**Layer 1: ML Pre-Filter (108 Features)**

The ensemble model extracts 108 quantitative features across 8 categories and scores every open market:

| Category | Features | Examples |
|----------|----------|---------|
| Price | 18 | Bid/ask, log odds, implied probability, price buckets |
| Volume & Liquidity | 14 | Log volume, OI ratio, dollar volume, intensity |
| Time Decay | 14 | Days/hours to expiry, theta proxy, urgency |
| Orderbook | 12 | Bid/ask pressure, imbalance, microprice |
| Efficiency | 10 | Overround, vig, arbitrage spread, dislocation |
| Momentum | 8 | Price momentum, mean reversion, conviction |
| Historical | 14 | Trend strength, volatility, volume trend, skew |
| Interactions | 16 | Price x volume, edge x liquidity, risk-adjusted edge |

**Layer 2: Claude AI Deep Research**

Markets with the highest edge potential are sent to Claude Sonnet with:
- All 108 model features and the RF prediction
- Live orderbook data (bid/ask pressure, imbalance, microprice)
- Price history and momentum indicators
- Volume trends and liquidity analysis

Claude then **searches the web** (up to 3 Google searches per market) for:
- Breaking news, latest polls, scores, announcements
- Expert forecasts and consensus estimates
- Upcoming catalysts that could move the market

It synthesizes the quantitative data + web research + world knowledge into a calibrated probability estimate.

**Layer 3: Risk Engine**

Every signal passes through:
- **Kelly-optimal sizing** with volatility, liquidity, and edge-decay scaling
- **Drawdown control** — scales down during losing streaks
- **Correlation limits** — max 3 positions per category
- **Multi-leg exits** — stop-loss, take-profit, time decay, model disagreement

---

## Quick Start

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Node.js 18+**
- **Kalshi account** with API access ([demo.kalshi.co](https://demo.kalshi.co) for testing)
- **Anthropic API key** (optional — enables AI + web search)

### 1. Clone and Install

```bash
git clone https://github.com/colelevy08/prediction-market-bot.git
cd prediction-market-bot

# Python backend
python -m venv .venv
source .venv/bin/activate
pip install -e .

# React frontend
cd frontend && npm install && cd ..
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [Environment Variables](#environment-variables)).

### 3. Launch

```bash
# Terminal 1 — Backend
source .venv/bin/activate
uvicorn bot.server:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open **http://localhost:5173**

---

## Dashboard Pages

The frontend has **7 pages** accessible via top navigation or keyboard shortcuts (`1`–`7`).

### Dashboard

The command center. Shows strategy overview, key metrics, equity curve, feature importance, and **Top Opportunities** — the highest-edge signals from the latest scan with one-click trading.

| Stat | Description |
|------|-------------|
| **Balance** | Available cash for trading |
| **Positions** | Currently open market positions |
| **Win Rate** | Profitable trades / total trades with W/L breakdown |
| **Sharpe** | Risk-adjusted returns: `(Rp - Rf) / σ` |
| **P&L** | Total profit/loss across all resolved trades |
| **Profit Factor** | Gross profit / gross loss (>1 = profitable) |

Features: live ticker bar, near-miss opportunities, gradient equity curve, auto-refresh every 30s.

### Signals

Signal discovery and trade execution. Run scans, filter by source (RF/AI), search by ticker, and execute trades.

- **RF Scan** — Fast model-only scan across all markets
- **RF + AI** — Full pipeline with Claude web research
- Entry signals table with model prob, market price, edge, confidence, Kelly size
- Exit signals with trigger reason (stop-loss, take-profit, time decay, model flip)

### Portfolio

Live Kalshi account view: cash balance, open positions with unrealized P&L, pending orders, and a **category heatmap** showing concentration risk at a glance.

### Markets

Browse all ~46,000 live Kalshi markets. Search, filter by status, sort by volume/spread/OI. Click **Analyze** on any market to see the full 108-feature breakdown with entry/exit signal indicators.

### Performance

Comprehensive analytics: win rate, Sharpe ratio, profit factor, max drawdown, MAE/MFE analysis, equity curve, per-trade P&L chart, category breakdown, and full trade history with CSV export.

### Testing

Two modes:
- **Backtester** — Train on settled markets, test on holdout. Parameter sweep across 25 entry/confidence combinations ranked by Sharpe.
- **Shadow Trading** — Paper trade with real market data and virtual funds. Auto-scan every 60s, track performance, train models — all risk-free.

### Settings

Control panel: connection status, trading parameters (max bet, min edge, Kelly fraction), auto-scan/auto-trade toggles, model retrain schedule, notification config, and trade logs.

---

## Trading Strategy

### Entry Rules

```python
entry_threshold = 0.93   # Market must be ~7% undervalued vs model
min_edge       = 0.03   # 3% minimum edge
min_confidence = 0.65   # 65% model confidence

if market_price <= model_prob * entry_threshold:
    if edge >= min_edge and confidence >= min_confidence:
        size = kelly_optimal_size(edge, confidence, volatility, liquidity)
        buy(side, size)
```

### Exit Rules (Multi-Leg)

| Trigger | Condition | Purpose |
|---------|-----------|---------|
| **Stop-loss** | Drawdown > 60% of entry edge | Cap losses |
| **Take-profit** | Gain > 70% of entry edge | Lock gains early |
| **Time decay** | Tighten stops as expiry approaches | Reduce theta risk |
| **Model disagreement** | Model flips direction | New info invalidated thesis |
| **Trailing stop** | Price drops >30% from high-water mark | Protect profits |
| **Force exit** | < 1 day to expiry | Avoid settlement risk |

### Position Sizing (Kelly Criterion)

```
b = (1 - market_cost) / market_cost     # payout odds
f* = (b·p - q) / b                       # optimal Kelly fraction
size = f* × kelly_fraction × vol_scalar × liq_scalar × edge_scalar × bankroll
```

Scaling factors reduce size for volatile markets, illiquid spreads, and when the model's edge predictions have been overestimating realized edge.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KALSHI_API_KEY_ID` | Yes | — | API key from Kalshi settings |
| `KALSHI_PRIVATE_KEY_PATH` | Yes* | `./kalshi_private_key.pem` | Path to PEM key (local) |
| `KALSHI_PRIVATE_KEY` | Yes* | — | Raw PEM content (cloud deploy) |
| `KALSHI_USE_DEMO` | No | `true` | `true` = demo, `false` = real money |
| `ANTHROPIC_API_KEY` | No | — | Enables Claude AI + web search |
| `SUPABASE_URL` | No | — | Supabase project URL |
| `SUPABASE_KEY` | No | — | Supabase service role key |
| `MAX_BET_AMOUNT_CENTS` | No | `2500` | Max per-trade wager ($25) |
| `MIN_EDGE_THRESHOLD` | No | `0.03` | Minimum edge to trade (3%) |
| `ENTRY_THRESHOLD` | No | `0.93` | Entry rule: price ≤ model × 0.93 |
| `MIN_CONFIDENCE` | No | `0.65` | Minimum model confidence |
| `MAX_DAILY_LOSS_CENTS` | No | `10000` | Daily stop-loss ($100) |
| `MAX_OPEN_POSITIONS` | No | `10` | Max simultaneous positions |
| `KELLY_FRACTION` | No | `0.5` | Half-Kelly (safer) |
| `NOTIFICATION_EMAIL` | No | — | Email for trade alerts |
| `SMTP_HOST` | No | `smtp.gmail.com` | SMTP server |
| `SMTP_PORT` | No | `587` | SMTP port |
| `SMTP_USER` | No | — | SMTP username |
| `SMTP_PASS` | No | — | SMTP password / app password |
| `SLACK_WEBHOOK_URL` | No | — | Slack alerts |
| `DISCORD_WEBHOOK_URL` | No | — | Discord alerts |
| `WEBHOOK_SECRET` | No | — | External webhook auth |
| `RETRAIN_DAYS` | No | `mon,wed,fri` | Auto-retrain schedule |
| `RETRAIN_HOUR` | No | `3` | Retrain hour (UTC) |

*One of `KALSHI_PRIVATE_KEY_PATH` or `KALSHI_PRIVATE_KEY` required.

---

## API Reference

<details>
<summary><strong>50+ REST endpoints</strong> — click to expand</summary>

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| Status | `/api/status` | GET | Bot status, connections, config |
| Portfolio | `/api/portfolio` | GET | Balance and account summary |
| Positions | `/api/positions` | GET | Open positions with P&L |
| Scanning | `/api/scan` | POST | Run market scan (RF + AI) |
| Signals | `/api/signals` | GET | Cached scan results |
| Events | `/api/events` | GET | Top Kalshi events by volume |
| Market | `/api/market/{ticker}` | GET | Full 108-feature analysis |
| Trading | `/api/trade` | POST | Place a trade on Kalshi |
| Orders | `/api/orders` | GET | Open orders |
| Cancel | `/api/order/{id}` | DELETE | Cancel pending order |
| Performance | `/api/performance` | GET | Metrics + equity curve |
| By Category | `/api/performance/by-category` | GET | P&L by market category |
| Features | `/api/model/features` | GET | Feature importances |
| Arbitrage | `/api/arbitrage` | GET | Cross-platform arb scan |
| Config | `/api/config` | PATCH | Update trading params |
| Backtest | `/api/backtest` | POST | Historical backtest |
| Sweep | `/api/backtest/sweep` | POST | Parameter sweep (25 combos) |
| Paper | `/api/paper` | GET | Paper trading state |
| Paper Config | `/api/paper/configure` | POST | Reset paper trader |
| Paper Funds | `/api/paper/add-funds` | POST | Add demo funds |
| Paper Scan | `/api/paper/scan` | POST | Run paper scan cycle |
| Paper Train | `/api/paper/train` | POST | Train model |
| Risk Reset | `/api/risk/reset-daily` | POST | Reset daily P&L |
| Auto-Scan | `/api/autoscan` | POST | Toggle auto-scan |
| Auto-Trade | `/api/autotrade` | POST | Toggle live trading |
| Scan Status | `/api/autoscan/status` | GET | Scheduler status + log |
| Shadow Trades | `/api/trades/shadow` | GET | Paper trade log |
| Live Trades | `/api/trades/live` | GET | Live trade log |
| History | `/api/history/trades` | GET | Persistent trades (Supabase) |
| Scan History | `/api/history/scans` | GET | Scan log (Supabase) |
| Perf History | `/api/history/performance` | GET | Performance snapshots |
| Training Hist | `/api/history/training` | GET | Model training runs |
| Webhook | `/api/webhook` | POST | External trigger (auth) |
| Notif Test | `/api/notifications/test` | POST | Test notifications |
| Notif Config | `/api/notifications/config` | GET | Notification status |
| Export | `/api/export/trades` | GET | CSV download |
| Heatmap | `/api/portfolio/heatmap` | GET | Category heatmap |
| Notes | `/api/trade/{i}/notes` | PATCH | Trade notes |
| Retrain | `/api/retrain/schedule` | GET/POST | Retrain schedule |
| Retrain Now | `/api/retrain/now` | POST | Immediate retrain |
| SSE | `/api/events/stream` | GET | Real-time event stream |

</details>

---

## Database (Supabase)

Optional but recommended — persists everything across deploys.

| Table | Purpose |
|-------|---------|
| `trades` | Every paper + live trade with full metrics |
| `scan_logs` | Scan results with signal counts |
| `paper_state` | Paper trader balance, positions, scan count |
| `performance_snapshots` | Periodic Sharpe, equity, win rate snapshots |
| `model_training_runs` | CV accuracy, OOB score, feature importance |

Setup: Create project at [supabase.com](https://supabase.com) → SQL Editor → run `supabase/schema.sql` → add URL + key to `.env`.

---

## Deployment

### Backend → Railway

```bash
railway login && railway init && railway up
```

Add `.env` variables in Railway dashboard. For the private key, paste full PEM into `KALSHI_PRIVATE_KEY`. Auto-deploys on `git push`.

### Frontend → Vercel

```bash
npx vercel
```

Set `VITE_API_URL` to your Railway URL. Auto-deploys on `git push`.

---

## Project Structure

```
prediction-market-bot/
├── bot/
│   ├── server.py            # FastAPI API (50+ endpoints, SSE, APScheduler)
│   ├── rf_model.py          # RF+GB ensemble, 108 features, signal generation
│   ├── analyzer.py          # Claude AI with web search + real market data
│   ├── backtester.py        # Backtesting, parameter sweep, paper trader
│   ├── kalshi_client.py     # Kalshi Trade API v2 (RSA-PSS auth)
│   ├── risk_manager.py      # Kelly sizing, drawdown, position limits
│   ├── performance.py       # Sharpe, log returns, MAE/MFE, equity curves
│   ├── notifier.py          # Email (SMTP) + Slack + Discord notifications
│   ├── database.py          # Supabase persistence layer
│   ├── arbitrage.py         # Cross-platform arbitrage detection
│   ├── draftkings_client.py # DraftKings data for cross-platform arb
│   ├── models.py            # Pydantic v2 data models
│   ├── config.py            # Environment configuration
│   └── main.py              # CLI entry point
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app — 7 tabs, ticker bar, theme toggle
│   │   ├── api.js           # API client with timeout + error handling
│   │   ├── index.css        # Tailwind + glassmorphism + animations
│   │   └── components/
│   │       ├── Dashboard.jsx      # Command center + top opportunities
│   │       ├── Signals.jsx        # Signal discovery + trade execution
│   │       ├── Portfolio.jsx      # Positions, orders, category heatmap
│   │       ├── Markets.jsx        # Browse 46K+ markets with analysis
│   │       ├── Performance.jsx    # Analytics, equity curves, trade history
│   │       ├── Testing.jsx        # Backtester + paper trading
│   │       ├── Settings.jsx       # Config, automation, notifications
│   │       ├── MarqueeText.jsx    # Stock ticker scrolling text
│   │       ├── Toast.jsx          # Notification toasts
│   │       ├── Tooltip.jsx        # Info tooltips
│   │       ├── Sparkline.jsx      # Inline mini charts
│   │       ├── Skeleton.jsx       # Loading placeholders
│   │       ├── ProgressBar.jsx    # Animated progress bars
│   │       ├── CommandPalette.jsx # Cmd+K command palette
│   │       ├── KeyboardHelp.jsx   # Keyboard shortcuts overlay
│   │       └── NotificationCenter.jsx
│   ├── package.json
│   └── vite.config.js
├── supabase/schema.sql      # Database schema
├── .env.example             # Environment template
├── pyproject.toml           # Python project config
├── requirements.txt         # Python dependencies
├── vercel.json              # Vercel deployment config
├── Procfile                 # Railway start command
└── railway.toml             # Railway config
```

---

## Risk Disclaimer

This bot is for **educational and research purposes**. Prediction market trading involves real financial risk. Always start with **demo mode** (`KALSHI_USE_DEMO=true`) and validate with **shadow trading** before committing real money. Past performance does not guarantee future results.

---

## License

MIT
