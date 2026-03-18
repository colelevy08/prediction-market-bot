# PREDICTIONBOT

**AI-powered prediction market trading bot for [Kalshi](https://kalshi.com) and [DraftKings](https://draftkings.com)**

Built on the Random Forest strategy from [@noisyb0y1's guide](https://x.com/noisyb0y1/status/2033856891181220265) — 106 engineered features, ensemble ML model, and strict entry/exit rules designed to find 2x undervalued markets.

```
┌─────────────────────────────────────────────────────┐
│  PREDICTIONBOT                          DEMO  SCAN  │
│  Dashboard  Signals  Portfolio  Markets  Testing     │
│─────────────────────────────────────────────────────│
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ BALANCE  │ │ SHARPE   │ │ WIN RATE │            │
│  │ $124.50  │ │ 1.82     │ │ 67.3%    │            │
│  └──────────┘ └──────────┘ └──────────┘            │
│                                                     │
│  ████████████████░░░░  Equity Curve                 │
│  ██████████░░░░░░░░░░  Feature Importance           │
└─────────────────────────────────────────────────────┘
```

---

## Architecture

```
Frontend (React + Vite + Tailwind)  ←→  Backend (FastAPI + Python)  ←→  Kalshi API
         Vercel                              Railway                  DraftKings
                                              ↕
                                      Supabase (PostgreSQL)
```

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn (Random Forest + Gradient Boosting), NumPy |
| Backend | FastAPI, uvicorn, APScheduler, httpx, Pydantic |
| Frontend | React 18, Vite, Tailwind CSS, Recharts |
| AI | Anthropic Claude (Sonnet 4.6) |
| Database | Supabase (PostgreSQL) — optional persistent storage |
| Auth | RSA-PSS signed API requests |
| Deploy | Railway (backend), Vercel (frontend), auto-deploy on git push |

---

## How It Works

### The Strategy

The bot uses a **Random Forest + Gradient Boosting ensemble** to predict market outcomes, then trades only when it finds extreme mispricings.

```
1. PREDICT    →  Ensemble model estimates true probability
2. FILTER     →  Only enter when market is 2x undervalued
3. TRADE      →  Buy undervalued contracts on Kalshi
4. MONITOR    →  Track MAE/MFE during position lifetime
5. EXIT       →  Sell when price converges to model or 7 days to expiry
6. EVALUATE   →  Sharpe Ratio, log returns, profit factor
```

### Entry/Exit Rules (from the guide)

```python
# Entry — buy when market is massively underpriced
if market_price <= model_probability * 0.5:
    buy()

# Exit — sell when price converges toward fair value
if market_price >= model_probability * 0.9:
    sell()

# Time stop — exit if running out of time
if days_to_expiry <= 7:
    sell()
```

### The Model

| Component | Details |
|-----------|---------|
| **Random Forest** | 200 trees, `sqrt(features)` per split, OOB scoring |
| **Gradient Boosting** | 150 estimators, 0.05 learning rate |
| **Ensemble Weight** | 60% RF + 40% GB |
| **Features** | 106 across 8 categories |
| **Min Confidence** | 70%+ model certainty required |

### 106 Features (8 Categories)

| Category | Count | Examples |
|----------|-------|---------|
| Price | 18 | bid/ask spreads, log odds, midpoint, price buckets |
| Volume & Liquidity | 14 | log volume, OI ratio, dollar volume, intensity |
| Time Decay | 14 | days/hours to expiry, theta proxy, time urgency |
| Orderbook | 12 | bid/ask pressure, microprice, weighted midpoint |
| Market Efficiency | 10 | overround, vig, arb spread, dislocation score |
| Momentum | 8 | price momentum proxy, mean reversion, conviction |
| Historical | 14 | trend strength, volatility, volume trend, skew |
| Interactions | 16 | price×volume, edge×liquidity, risk-adjusted edge |

### Performance Metrics

```
Sharpe Ratio = (Rp - Rf) / σ
  < 1   →  Bad
  1-2   →  Good
  > 2   →  Excellent

Log Return = ln(P1 / P0)    # Correct for big moves, additive
MAE = max drawdown during trade lifetime
MFE = max gain during trade lifetime
```

---

## Quick Start

### Prerequisites

- **Python 3.11+** (we recommend 3.12)
- **Node.js 18+**
- **Kalshi account** with API access ([demo.kalshi.co](https://demo.kalshi.co) for testing)
- **Anthropic API key** (optional — enables AI+RF hybrid mode)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/prediction-market-bot.git
cd prediction-market-bot

# Python backend
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e .

# React frontend
cd frontend
npm install
cd ..
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Required — get from demo.kalshi.co/settings/api
KALSHI_API_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=./kalshi-private-key.pem
KALSHI_USE_DEMO=true        # Start with demo, switch to false for real money

# Optional — enables Claude AI hybrid analysis
ANTHROPIC_API_KEY=sk-ant-...

# Risk management
MAX_BET_AMOUNT_CENTS=2500   # $25 max per trade
MIN_EDGE_THRESHOLD=0.08     # 8% minimum edge
MAX_DAILY_LOSS_CENTS=10000  # $100 daily loss limit
MAX_OPEN_POSITIONS=10

# Optional — persistent database (survives restarts/redeploys)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
```

### 3. Launch

Open **two terminals**:

```bash
# Terminal 1 — Backend API
source .venv/bin/activate
uvicorn bot.server:app --reload --port 8000
```

```bash
# Terminal 2 — Frontend UI
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Using the Bot

### Dashboard
Overview of strategy, balance, win rate, Sharpe Ratio, equity curve, and feature importance chart.

### Signals
Click **Scan** to analyze live markets. The bot shows entry/exit signals with model probability, market price, edge, confidence, and recommended position size. One-click **Trade** button for approved signals.

### Portfolio
Live view of your Kalshi balance, open positions with unrealized P&L, and open orders with cancel buttons.

### Markets
Browse all live Kalshi markets sorted by volume. Click **Analyze** on any market to see the full 106-feature breakdown, entry/exit rule evaluation, and model vs. market probability.

### Performance
Sharpe Ratio, win rate, MAE/MFE, equity curve, P&L per trade chart, and full trade history with log returns.

### Testing

Two modes for validating the strategy without risking real money:

#### Backtester
- Fetches settled markets from Kalshi history
- Trains the model on 60% of data, tests on 40%
- Reports full metrics: Sharpe, win rate, profit factor, equity curve
- **Parameter Sweep**: tests 25 combinations of entry threshold x confidence level to find optimal settings

#### Shadow Trading (Paper Trading)
- Uses **live market data** but simulates order fills — no real money at risk
- **Add Demo Funds**: Instant preset buttons ($100, $500, $1,000, $5,000, $10,000) or enter a custom amount. Adds to your balance without resetting positions or trade history
- **Reset All**: Wipes everything and starts fresh with a chosen balance
- Train the model first, then click **Scan Once** or enable **Auto Scan** (every 60 seconds)
- Watch the shadow equity curve grow in real time
- Full performance tracking: Sharpe Ratio, win rate, MAE/MFE, P&L, open positions

### Settings
Adjust risk parameters, toggle 24/7 auto-scan and auto-trade, view scan logs, and check connection status for Kalshi, Claude AI, RF model, and Supabase.

---

## Database (Supabase)

Supabase is **optional** but recommended — it persists all data across server restarts and redeploys.

### What It Stores

| Table | Purpose |
|-------|---------|
| `trades` | Every paper and live trade with full metrics |
| `scan_logs` | Every scan with signal counts, entries, exits |
| `paper_state` | Paper trader balance, positions, scan count |
| `performance_snapshots` | Periodic Sharpe, win rate, equity snapshots |
| `model_training_runs` | CV accuracy, OOB score, feature importance per training |

### Setup

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor** and run the schema SQL to create the 5 tables
3. Add `SUPABASE_URL` and `SUPABASE_KEY` (service role key) to your `.env`
4. The bot auto-connects on startup — all methods gracefully no-op when Supabase isn't configured

### History Endpoints

When Supabase is connected, these endpoints return persistent data:

- `GET /api/history/trades` — trade history by mode (paper/live)
- `GET /api/history/scans` — scan log history
- `GET /api/history/performance` — performance snapshots over time
- `GET /api/history/training` — model training run history

---

## Always-Running Deployment

### Deploy Backend to Railway

```bash
railway login
railway init
railway up
```

Add your `.env` variables in Railway's dashboard under **Variables**. For the private key, paste the full PEM content into a `KALSHI_PRIVATE_KEY` variable (the bot reads from env var when no file is found).

Railway runs the server using the `Procfile`:

```
web: uvicorn bot.server:app --host 0.0.0.0 --port $PORT
```

**Auto-deploy**: Connect your GitHub repo in Railway's dashboard. Every push to `main` triggers a new deploy.

### Deploy Frontend to Vercel

```bash
cd frontend
npx vercel
```

Set the environment variable `VITE_API_URL` to your Railway backend URL.

**Auto-deploy**: Connect your GitHub repo in Vercel. Set root directory to `frontend`. Every push auto-deploys.

### Or Run Locally with PM2

```bash
npm install -g pm2

# Start backend
pm2 start "uvicorn bot.server:app --port 8000" --name bot-api

# Start frontend
pm2 start "npm run dev" --name bot-ui --cwd frontend

# Auto-start on reboot
pm2 save
pm2 startup
```

---

## API Endpoints (32+)

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| Status | `/api/status` | GET | Bot status, connections, config |
| Portfolio | `/api/portfolio` | GET | Kalshi portfolio summary |
| Positions | `/api/positions` | GET | Open positions with P&L |
| Scanning | `/api/scan` | POST | Run market scan (RF + optional AI) |
| Signals | `/api/signals` | GET | Cached scan results |
| Events | `/api/events` | GET | Top Kalshi events by volume |
| Market | `/api/market/{ticker}` | GET | Full 106-feature analysis |
| Trading | `/api/trade` | POST | Place a trade on Kalshi |
| Orders | `/api/orders` | GET | Open orders |
| Cancel | `/api/order/{id}` | DELETE | Cancel an order |
| Performance | `/api/performance` | GET | Metrics, equity curve, trade log |
| Features | `/api/model/features` | GET | Feature importance rankings |
| Arbitrage | `/api/arbitrage` | GET | Cross-platform scan |
| Config | `/api/config` | PATCH | Update risk parameters |
| Backtest | `/api/backtest` | POST | Run historical backtest |
| Sweep | `/api/backtest/sweep` | POST | Parameter sweep (25 combos) |
| Paper | `/api/paper` | GET | Paper trading state |
| Paper | `/api/paper/configure` | POST | Reset paper trader |
| Paper | `/api/paper/add-funds` | POST | Add demo funds (no reset) |
| Paper | `/api/paper/scan` | POST | Paper scan cycle |
| Paper | `/api/paper/train` | POST | Train model on history |
| Scheduler | `/api/autoscan` | POST | Toggle auto-scan (60s) |
| Scheduler | `/api/autotrade` | POST | Toggle auto-trade (live) |
| Scheduler | `/api/autoscan/status` | GET | Scheduler status + log |
| History | `/api/history/trades` | GET | Persistent trade history |
| History | `/api/history/scans` | GET | Persistent scan log |
| History | `/api/history/performance` | GET | Performance snapshots |
| History | `/api/history/training` | GET | Model training runs |

---

## Project Structure

```
prediction-market-bot/
├── bot/
│   ├── server.py          # FastAPI REST API (32+ endpoints)
│   ├── rf_model.py        # RF+GB ensemble, 106 features
│   ├── backtester.py      # Historical backtesting + paper trader
│   ├── kalshi_client.py   # Kalshi Trade API v2 client (RSA-PSS auth)
│   ├── draftkings_client.py # DraftKings scraper
│   ├── analyzer.py        # Claude AI market analysis
│   ├── performance.py     # Sharpe, log returns, MAE/MFE tracking
│   ├── risk_manager.py    # Position limits, daily loss, Kelly sizing
│   ├── arbitrage.py       # Cross-platform arbitrage detection
│   ├── database.py        # Supabase persistence layer (optional)
│   ├── models.py          # Pydantic data models
│   ├── config.py          # Environment configuration
│   └── main.py            # CLI entry point
├── frontend/
│   ├── src/
│   │   ├── App.jsx        # Main app with 7 tabs
│   │   ├── api.js         # API client
│   │   └── components/
│   │       ├── Dashboard.jsx
│   │       ├── Signals.jsx
│   │       ├── Portfolio.jsx
│   │       ├── Markets.jsx
│   │       ├── Performance.jsx
│   │       ├── Testing.jsx
│   │       └── Settings.jsx
│   └── package.json
├── .env.example
├── pyproject.toml
├── requirements.txt
├── Procfile
└── README.md
```

---

## Risk Disclaimer

This bot is for **educational and research purposes**. Prediction market trading involves real financial risk. Always start with **demo mode** (`KALSHI_USE_DEMO=true`) and validate with **shadow trading** before using real money. Past backtesting performance does not guarantee future results.

---

## License

MIT
