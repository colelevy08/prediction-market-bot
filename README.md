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
- **Parameter Sweep**: tests 25 combinations of entry threshold × confidence level to find optimal settings

#### Shadow Trading
- Uses **live market data** but simulates order fills
- Records every trade the bot _would have made_
- Train the model first, then click **Scan Once** or enable **Auto Scan** (every 60 seconds)
- Watch the shadow equity curve grow in real time
- Perfect for validating the strategy before committing real money

### Settings
Adjust risk parameters, view connection status, and see the full strategy reference.

---

## Always-Running Deployment

To keep the bot scanning 24/7, you need:

| Component | Recommended Service | Purpose |
|-----------|-------------------|---------|
| **Python Backend** | [Railway](https://railway.app) or [Fly.io](https://fly.io) | Runs FastAPI server + shadow trader |
| **React Frontend** | [Vercel](https://vercel.com) | Hosts the UI |
| **Database** (optional) | [Supabase](https://supabase.com) | Persists trade history across restarts |

### Deploy Backend to Railway

```bash
# Railway auto-detects Python projects
railway login
railway init
railway up
```

Add your `.env` variables in Railway's dashboard under **Variables**.

Railway will run the server using the `Procfile`:

```
web: uvicorn bot.server:app --host 0.0.0.0 --port $PORT
```

### Deploy Frontend to Vercel

```bash
cd frontend
npx vercel
```

Set the environment variable `VITE_API_URL` to your Railway backend URL.

### Or Run Locally with PM2

For a simpler always-on setup on your own machine:

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

## Project Structure

```
prediction-market-bot/
├── bot/
│   ├── server.py          # FastAPI REST API (25+ endpoints)
│   ├── rf_model.py        # RF+GB ensemble, 106 features
│   ├── backtester.py       # Historical backtesting + paper trader
│   ├── kalshi_client.py    # Kalshi Trade API v2 client
│   ├── draftkings_client.py # DraftKings scraper
│   ├── analyzer.py         # Claude AI market analysis
│   ├── performance.py      # Sharpe, log returns, MAE/MFE
│   ├── risk_manager.py     # Position limits, daily loss, Kelly sizing
│   ├── arbitrage.py        # Cross-platform arbitrage detection
│   ├── models.py           # Pydantic data models
│   ├── config.py           # Environment configuration
│   └── main.py             # CLI entry point
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main app with 7 tabs
│   │   ├── api.js          # API client
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
└── README.md
```

---

## Risk Disclaimer

This bot is for **educational and research purposes**. Prediction market trading involves real financial risk. Always start with **demo mode** (`KALSHI_USE_DEMO=true`) and validate with **shadow trading** before using real money. Past backtesting performance does not guarantee future results.

---

## License

MIT
