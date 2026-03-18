# PredictionBot

AI-powered prediction market trading bot for [Kalshi](https://kalshi.com) with a full-stack dashboard, ensemble ML model, paper trading, and live automation.

## Overview

PredictionBot is a full-stack application that uses machine learning to find mispriced contracts on the Kalshi prediction market exchange and trade them automatically. The core strategy is based on a **Random Forest + Gradient Boosting ensemble** that engineers **106 numerical features** from market data across 8 categories (price, volume, time decay, orderbook, efficiency, momentum, historical, and interaction features). When the model estimates a market is **2x undervalued** relative to its true probability, the bot generates a buy signal.

The system has two operating modes: **shadow trading** (paper trading with virtual funds using real market data) and **live trading** (placing real orders on Kalshi via their Trade API v2). Shadow trading lets you validate the strategy risk-free before committing real money. The bot can run 24/7 with auto-scanning every 60 seconds, scanning all ~5,000+ Kalshi events and ~41,000+ markets per cycle.

The frontend is a React dashboard with 7 pages covering every aspect of the trading pipeline: signal discovery, portfolio management, market browsing, performance analytics, backtesting, and configuration. The backend is a FastAPI REST API with 40+ endpoints that connects to Kalshi for market data and order execution, Supabase for persistent storage, and optionally Claude AI for hybrid analysis. Slack and Discord webhook notifications keep you informed of trades and retrains.

## Architecture

```
Frontend (React + Vite + Tailwind)  <-->  Backend (FastAPI + Python)  <-->  Kalshi API
         Vercel                              Railway                   DraftKings
                                              |
                                      Supabase (PostgreSQL)
                                              |
                                    Slack / Discord Webhooks
```

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn (Random Forest + Gradient Boosting), NumPy, StandardScaler |
| Backend | FastAPI, uvicorn, APScheduler, httpx, Pydantic |
| Frontend | React 18, Vite, Tailwind CSS, Recharts |
| AI | Anthropic Claude (optional hybrid analysis) |
| Database | Supabase (PostgreSQL) -- optional persistent storage |
| Auth | RSA-PSS signed API requests to Kalshi Trade API v2 |
| Notifications | Slack and Discord webhooks for trade alerts |
| Deploy | Railway (backend), Vercel (frontend), auto-deploy on git push |

## Quick Start

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Node.js 18+**
- **Kalshi account** with API access ([demo.kalshi.co](https://demo.kalshi.co) for testing)
- **Anthropic API key** (optional -- enables AI+RF hybrid mode)

### 1. Clone and Install

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

Edit `.env` with your credentials (see [Environment Variables](#environment-variables) for the full list).

### 3. Launch

Open two terminals:

```bash
# Terminal 1 -- Backend API
source .venv/bin/activate
uvicorn bot.server:app --reload --port 8000
```

```bash
# Terminal 2 -- Frontend UI
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Pages and Features

The frontend has 7 tabs accessible from the top navigation bar (or via keyboard shortcuts `1`-`7`). Each page is documented in full below, including every column header, stat card, and abbreviation.

#### Global Features

- **Keyboard shortcuts**: Press `1`-`7` to switch tabs instantly (disabled when typing in inputs).
- **Toast notifications**: All success/error messages appear as non-blocking toast popups in the bottom-right corner (replacing browser alert dialogs).
- **Theme toggle**: Switch between dark and light mode via the header button.
- **Tooltips**: Hover over the info icon (ⓘ) on any label for a detailed explanation.

---

### Dashboard

The main overview page showing the strategy configuration, key performance metrics, charts, and latest scan results.

#### Strategy Overview

Four info cards summarizing the core strategy parameters:

| Card | Description |
|------|-------------|
| **Ensemble** | Shows the model configuration: RF(n_trees) + GB(150) x n_features. For example, `RF(500) + GB(150) x 106` means 500 Random Forest trees, 150 Gradient Boosting trees, and 106 features. |
| **Entry Rule** | `mkt <= model x 0.5` -- the bot only buys when the market price is at most half the model's estimated probability (i.e., 2x undervalued). |
| **Exit Rule** | `mkt >= model x 0.9` -- the bot sells when the market price has converged to 90% of the model's estimate. |
| **Min Confidence** | `70%+ required` -- the model must be at least 70% confident in its prediction before generating a signal. |

#### Stat Cards (6 cards)

| Card | What It Means |
|------|--------------|
| **Balance** | Current cash balance available for trading (in dollars). This is your liquid capital not tied up in positions. |
| **Positions** | Number of currently open market positions (contracts you hold). |
| **Win Rate** | Percentage of resolved trades that were profitable. Shown as a percentage with a W/L breakdown (e.g., `12W / 5L`). Green if >= 60%, yellow if >= 50%, red if below 50%. A progress bar visualizes this. |
| **Sharpe** | The Sharpe Ratio -- a risk-adjusted return metric. Calculated as `(average return - risk-free rate) / standard deviation of returns`. Rated: < 1 = poor, 1-2 = good, > 2 = excellent. A badge shows the rating. |
| **P&L** | Total Profit and Loss across all resolved trades, in dollars. Green when positive, red when negative. |
| **Profit Factor** | Gross profit divided by gross loss. A value above 1 means you are making more than you are losing. Above 1.5 is good, above 2 is excellent. |

#### Top 3 Opportunities

Shows the three best trading opportunities from the latest scan, with a bet amount selector ($10-$500). For each opportunity, displays: ticker, model probability, market price, edge percentage, and estimated earning potential based on the selected bet amount. Auto-refreshes every 30 seconds.

#### Last Updated Timestamp

A relative time indicator (e.g., "Updated 2m ago") at the top of the dashboard showing when data was last refreshed. Automatically updates every 15 seconds.

#### Equity Curve Chart

A line chart showing cumulative equity (in dollars) over the trade sequence. Each point represents the portfolio value after a trade resolved. An upward-sloping curve indicates consistent profitability.

#### Feature Importance Chart

A horizontal bar chart showing the top 10 most important features used by the trained model, ranked by their contribution to predictions. The importance value is a percentage. If the model is untrained, the chart shows "Model untrained. Using heuristic mode."

#### Latest Scan Summary

Appears after running a scan. Shows 5 stat cards:

| Card | Description |
|------|-------------|
| **Events** | Number of Kalshi events scanned. |
| **Markets** | Number of individual markets analyzed across those events. |
| **RF Signals** | Number of entry signals found by the Random Forest model. Glows green when > 0. |
| **AI Signals** | Number of entry signals found by Claude AI analysis (only when AI scan is used). |
| **Exit Signals** | Number of exit signals for positions that should be sold. Glows red when > 0. |

#### Strategy Formulas

Four formula cards as a quick reference:

- **Entry**: `if mkt <= prob * 0.5: buy()`
- **Exit**: `if mkt >= prob * 0.9: sell()`
- **Log Return**: `r = ln(P1 / P0)`
- **Sharpe Ratio**: `SR = (Rp - Rf) / sigma`

---

### Signals

The signal discovery and trade execution page. Run scans here, view entry/exit signals, and execute trades.

#### Controls

- **RF Scan** button: Runs a market scan using only the Random Forest model.
- **RF + AI** button: Runs a scan using both the Random Forest model and Claude AI for hybrid analysis.
- **Filter pills**: Filter signals by `All`, `Ready` (passes all risk checks), `RF` (Random Forest only), or `AI` (Claude AI only).

#### Entry Signals Table

Shows markets where the bot has found buying opportunities. The entry rule is `mkt <= model x 0.5`.

| Column | Full Name | Description |
|--------|-----------|-------------|
| **Ticker** | Ticker | The unique Kalshi market identifier (e.g., `KXBTCD-25MAR14-T55500`). |
| **Market** | Market Title | Human-readable description of the market (e.g., "Bitcoin above $55,500 on March 14?"). |
| **Side** | Side | `YES` or `NO` -- which side of the contract the bot recommends buying. YES means betting the event will happen; NO means betting it will not. |
| **Src** | Source | Signal source. `RF` = Random Forest model, `AI` = Claude AI analysis. |
| **Model** | Model Probability | The probability estimated by the ML model (as a percentage). This is the model's belief about the true likelihood of the event. |
| **Market** | Market Probability | The current implied probability from the market price (as a percentage). This is what the market thinks. |
| **Edge** | Edge | Model probability minus market price, as a percentage. Higher edge means the market is more undervalued relative to the model. Green badge if >= 15%, yellow if >= 8%, muted if positive but < 8%, red if negative. |
| **Conf** | Confidence | The model's confidence in this particular prediction (as a percentage). 70% minimum is required. Green >= 80%, yellow >= 70%, red < 70%. |
| **Size** | Recommended Size | The recommended dollar amount to wager, calculated using the Kelly Criterion position sizing formula. |
| **Status** | Risk Check Status | `Ready` = all risk limits pass (position limits, daily loss, edge threshold). If not ready, shows the first few words of the rejection reason. `Done` = trade already executed. |
| **Action** | Action | A `Trade` button appears for signals with `Ready` status. Click to execute the trade on Kalshi. |

#### Exit Signals Table

Shows positions that should be sold. The exit rule is `mkt >= model x 0.9` or `expiry < 7 days`.

| Column | Description |
|--------|-------------|
| **Ticker** | Market identifier. |
| **Market** | Market title. |
| **Model** | Model's estimated probability. |
| **Market** | Current market-implied probability. |
| **Reason** | Why the exit was triggered (e.g., "Price converged to fair value" or "Approaching expiry"). |

#### Signal Reasoning

A list showing the model's reasoning for each signal (up to 5). Displays the ticker and a human-readable explanation of why the signal was generated.

---

### Portfolio

Live view of your Kalshi account: cash, positions, orders, and a category heatmap.

#### Summary Cards (4 cards)

| Card | Description |
|------|-------------|
| **Cash Balance** | Cash available for trading, not including unrealized gains or losses on open positions. |
| **Open Positions** | Number of markets where the bot currently holds a position. |
| **Open Orders** | Number of pending limit orders waiting to be filled on Kalshi. |
| **Unrealized P&L** | Sum of unrealized profit/loss across all open positions. This is how much you would gain or lose if all positions were closed at current market prices. |

#### Positions Table

| Column | Full Name | Description |
|--------|-----------|-------------|
| **Ticker** | Ticker | Kalshi market identifier. |
| **Market** | Market Title | Human-readable market description. |
| **Side** | Side | `YES` or `NO` -- which side you hold. |
| **Qty** | Quantity | Number of contracts held in this position. |
| **Avg** | Average Price | Average price paid per contract, in cents. For example, `35c` means you paid 35 cents per contract. |
| **Current** | Current Price | Current market price per contract, in cents. |
| **P&L** | Profit/Loss | Unrealized profit or loss for this position if it were closed now. Green for profit, red for loss. |

Rows are highlighted green (winning) or red (losing) based on unrealized P&L.

#### Portfolio Heatmap

Positions grouped by market category (e.g., "Politics", "Economics", "Weather"). Each tile shows:
- Category name
- Number of positions
- Total dollars invested
- Ticker badges for each position in that category

This helps you see concentration risk at a glance.

#### Open Orders Table

| Column | Description |
|--------|-------------|
| **ID** | First 12 characters of the Kalshi order ID. |
| **Ticker** | Market identifier. |
| **Side** | YES or NO. |
| **Action** | `BUY` (opening a new position) or `SELL` (closing an existing one). |
| **Price** | Limit price for the order, in cents. |
| **Count** | Number of contracts remaining to be filled. |
| *(button)* | **Cancel** button to cancel the order. |

---

### Markets

Browse all live Kalshi markets, sorted by volume. Analyze any market in detail.

#### Controls

- **Search**: Type in the search box to filter markets by ticker or title (case-insensitive).
- **Status filter pills**: Filter by market status (`All`, `open`, `closed`, `settled`).
- **Sort dropdown**: Sort by Volume, Open Interest, Spread (tightest), Bid (highest/lowest), or Status.
- **Event limit**: Fetch 10, 20, or 50 events.
- **Refresh**: Reload market data.

#### Markets Table

| Column | Full Name | Description |
|--------|-----------|-------------|
| **Ticker** | Ticker | Unique Kalshi market identifier. |
| **Market** | Market Title | Human-readable market question. |
| **Event** | Event Title | The parent event this market belongs to (e.g., an event might be "Bitcoin Price" with multiple markets for different price thresholds). |
| **Bid** | YES Bid | The highest price (in cents) that a buyer is currently willing to pay for a YES contract. This is the best price you could sell at immediately. |
| **Ask** | YES Ask | The lowest price (in cents) that a seller is currently willing to accept for a YES contract. This is the best price you could buy at immediately. |
| **Spread** | Spread | The difference between Ask and Bid, in cents. A lower spread means the market is more liquid and easier to trade. For example, a market with Bid=40c and Ask=45c has a spread of 5c. |
| **Vol** | Volume | Total number of contracts traded on this market since it opened. Higher volume generally indicates more liquid, actively traded markets. Color-coded: green > 10,000, blue > 1,000, gray otherwise. |
| **OI** | **Open Interest** | **The total number of outstanding contracts that have not yet been settled.** Every contract has a buyer and a seller; OI counts the number of these buyer-seller pairs that are still active (not yet resolved by the event outcome or closed by the participants). OI increases when a new buyer and a new seller create a fresh contract, and decreases when both sides close their positions or the market settles. High OI means many participants have money at stake in this market. OI is different from volume: volume counts every trade that happens (including day traders who buy and sell the same contract), while OI only counts contracts that are currently open. |
| *(button)* | Analyze | Opens the Market Detail view for this ticker. |

The search box, sort dropdown, and status filter pills work together to narrow down the displayed markets.

#### Market Detail View

When you click **Analyze** on any market, an expanded detail panel opens showing:

**Four stat cards:**

| Card | Description |
|------|-------------|
| **Model** | Probability estimated by the Random Forest model. |
| **Market** | Current implied probability from the market price. |
| **Edge** | Model probability minus market price. Positive means the market is undervalued according to the model. |
| **Spread** | Ask minus Bid in cents. Lower means tighter, more liquid market. |

**Entry/Exit Signal Indicators:**

- **Entry**: Shows whether `market_price <= model_probability x 0.5`. If yes, displays "BUY SIGNAL" in green with a glowing border. Otherwise shows "No entry".
- **Exit**: Shows whether `market_price >= model_probability x 0.9`. If yes, displays "SELL SIGNAL" in red. Otherwise shows "No exit".

**Order Book Stats (4 mini-cards):**

| Label | Description |
|-------|-------------|
| **YES Bid** | Highest price a buyer will pay for YES shares (in cents). |
| **YES Ask** | Lowest price a seller will accept for YES shares (in cents). |
| **Volume** | Total contracts traded. |
| **OI** | Open Interest -- total outstanding contracts not yet settled (see explanation above). |

**All Features:** An expandable section showing every one of the model's features and their computed values for this market.

---

### Performance

Comprehensive trading performance analytics for all resolved trades.

#### Metric Cards (10 cards)

**Top row (6 cards):**

| Card | Description | Formula/Details |
|------|-------------|-----------------|
| **Win Rate** | Percentage of trades that were profitable. | Shown with a W/L breakdown and progress bar. |
| **Sharpe Ratio** | Risk-adjusted return measurement. | `SR = (Rp - Rf) / sigma` where Rp = portfolio return, Rf = risk-free rate, sigma = standard deviation. < 1 is bad, 1-2 is good, > 2 is excellent. |
| **P&L** | Total profit and loss across all resolved trades, in dollars. | Green if positive, red if negative. |
| **Profit Factor** | Ratio of gross profit to gross loss. | > 1 means profitable overall. > 1.5 is good, > 2 is excellent. A profit factor of 2.0 means you make $2 for every $1 you lose. |
| **Avg MAE** | Average Max Adverse Excursion. | The average worst drawdown experienced during each trade before it was closed. Measures how much pain you typically endure per trade. Shown as a percentage. |
| **Avg MFE** | Average Max Favorable Excursion. | The average best unrealized gain during each trade. Measures how much opportunity existed per trade. If MFE is much larger than realized P&L, you may be exiting too early. Shown as a percentage. |

**Second row (4 cards):**

| Card | Description | Formula/Details |
|------|-------------|-----------------|
| **Avg Log Return** | Average logarithmic return across trades. | `ln(exit_price / entry_price)`. Log returns are used instead of simple returns because they are additive across trades and handle large price moves correctly. |
| **Avg Edge** | Average difference between model probability and market price at entry. | Higher average edge means the model is consistently finding undervalued markets. |
| **Max Drawdown** | Largest peak-to-trough decline in equity. | The worst cumulative loss from a high point. For example, if equity peaked at $150 and dropped to $120, the max drawdown is $30. |
| **Best / Worst** | Best single-trade profit and worst single-trade loss. | Shown side by side: green for best, red for worst. |

#### Equity Curve Chart

Line chart showing cumulative equity over the trade sequence. Same concept as the Dashboard chart but larger.

#### P&L Per Trade Chart

Bar chart showing the profit or loss of each individual trade. Green bars are wins, red bars are losses. Useful for spotting streaks and outliers.

#### P&L by Category

Breakdown of performance grouped by market category (e.g., "Politics", "Crypto", "Economics"). Each category tile shows:
- Total P&L in dollars (green/red)
- Number of trades
- Win rate percentage with a progress bar

This helps identify which market categories the model performs best in.

#### Trade History Table

Full log of all resolved trades with CSV export.

| Column | Full Name | Description |
|--------|-----------|-------------|
| **#** | Trade Number | Sequential trade number. |
| **Ticker** | Ticker | Kalshi market identifier. |
| **Side** | Side | YES or NO. |
| **Category** | Category | Market category (e.g., "Politics", "Economics"). |
| **Entry** | Entry Price | Price in cents when the position was opened. |
| **Exit** | Exit Price | Price in cents when the position was closed. |
| **Log R** | Log Return | Logarithmic return: `ln(exit_price / entry_price)`. Positive means profit, negative means loss. |
| **P&L** | Profit/Loss | Dollar profit or loss for this trade. |
| **MAE** | Max Adverse Excursion | Worst drawdown during this specific trade, as a percentage. |
| **MFE** | Max Favorable Excursion | Best unrealized gain during this specific trade, as a percentage. |
| **Result** | Win/Loss | `Win` (green) or `Loss` (red) badge. |
| **Notes** | Notes | Editable text field. Click to add personal notes about the trade. |

Two export buttons:
- **Export CSV**: Downloads live trade history as a CSV file.
- **Export Paper CSV**: Downloads paper/shadow trade history as a CSV file.

#### Sharpe Ratio Guide

Visual reference at the bottom of the page:
- **< 1** = Bad (red)
- **1 - 2** = Good (yellow)
- **> 2** = Excellent (green)

---

### Testing

Two modes for validating the strategy without risking real money, toggled via a segmented pill control at the top.

#### Backtester Mode

Fetches settled (resolved) markets from Kalshi's history, trains the model on a portion, and tests on the remainder.

**Configuration Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Balance ($)** | Starting balance for the simulated backtest. | $100 |
| **Entry Threshold** | Market price must be <= model probability x this value to trigger a buy. Lower = stricter. | 0.5 |
| **Exit Threshold** | Market price must be >= model probability x this value to trigger a sell. | 0.9 |
| **Min Confidence** | Minimum model confidence required to act on a signal. | 0.70 (70%) |
| **Max Markets** | Maximum number of settled markets to fetch for backtesting. | 200 |

**Buttons:**
- **Run Backtest**: Executes a single backtest with the configured parameters.
- **Parameter Sweep**: Tests 25 combinations of entry threshold x confidence level, ranked by Sharpe ratio.

**Backtest Results (6 stat cards):**

| Card | Description |
|------|-------------|
| **Win Rate** | Percentage of backtest trades that were profitable, with W/L count. |
| **Sharpe** | Sharpe Ratio of the backtest run. |
| **P&L** | Total profit/loss of the backtest. |
| **Profit Factor** | Gross profit / gross loss. |
| **CV Accuracy** | **Cross-Validation Accuracy** -- the model's prediction accuracy measured using k-fold cross-validation during training. The training data is split into k folds; the model trains on k-1 folds and tests on the held-out fold, rotating through all folds. The average accuracy across folds is reported. Also shows **OOB** (Out-of-Bag Score) -- Random Forest's built-in validation using samples that were not included in each individual tree's bootstrap sample. OOB provides a "free" accuracy estimate without needing a separate validation set. |
| **Max Drawdown** | Largest peak-to-trough equity decline during the backtest. |

**Additional info cards:**
- **Train / Test**: Number of samples used for training vs. testing (e.g., `600 / 400`).
- **Signals / Filtered**: How many signals the model generated vs. how many were filtered out by entry/confidence rules.
- **Avg Edge / Log Return**: Average edge and average log return across backtest trades.

**Equity Curve**: Line chart of backtest equity over trade sequence.

**Feature Importance (Top 15)**: Horizontal bar chart of the 15 most important features from the backtest model.

**Trade Log Table**: Every simulated trade with columns: #, Ticker, Side, Entry, Exit, P&L, Edge.

**Parameter Sweep Results Table:**

| Column | Description |
|--------|-------------|
| **Rank** | Ranking by Sharpe Ratio (best first). |
| **Entry** | Entry threshold used for this combination. |
| **Conf** | Minimum confidence used. |
| **Sharpe** | Resulting Sharpe Ratio. |
| **Win%** | Win rate percentage. |
| **P&L** | Total profit/loss. |
| **Trades** | Number of trades generated. |
| **PF** | Profit Factor. |

The top-ranked row is highlighted.

#### Shadow Trading Mode (Paper Trading)

Simulates live trading using **real market data** but with virtual funds. No real money is at risk.

**Add Demo Funds**: Preset buttons for $100, $500, $1,000, $5,000, and $10,000, or enter a custom dollar amount. Adds to your existing shadow balance without resetting positions or trade history.

**Controls:**
- **Reset Balance ($)** input + **Reset All** button: Wipes everything (balance, positions, trades, scans) and starts fresh with the specified balance.
- **Train Model**: Fetches settled markets from Kalshi and trains the ML model. Shows cumulative sample count, new samples added, and CV accuracy.
- **Scan Once**: Runs one market scan cycle using the trained model against live data.
- **Auto Scan (60s)**: Toggles automatic scanning every 60 seconds. When active, shows a pulsing "LIVE" indicator.

**Stat Cards (6 cards):**

| Card | Description |
|------|-------------|
| **Balance** | Current shadow trading balance in dollars. |
| **Positions** | Number of open shadow positions. |
| **Trades** | Total resolved shadow trades with W/L breakdown. |
| **Sharpe** | Sharpe Ratio of shadow trading performance. |
| **P&L** | Total profit/loss from resolved shadow trades. |
| **Model** | Shows `Trained` or `Heuristic` (untrained). Also shows total scan count and training sample count. |

**Open Shadow Positions**: List of current shadow positions with ticker, side badge, contract count, entry price, and model probability.

**Shadow Equity Chart**: Line chart of shadow trading equity over the trade sequence.

**Shadow Trade Log Table**: Columns: Ticker, Side, Entry, Exit, P&L. Rows highlighted green (win) or red (loss).

**Scan Log**: Reverse-chronological feed of scan results. Each entry shows: scan number, entries added (+N in green), exits triggered (-N in red), open positions count, and balance after the scan.

---

### Settings

The central control panel for configuration, monitoring, and automation. Displayed as a two-column layout.

#### Left Column

**Connections Panel**: Shows live connection status for:
- **Kalshi API** -- green dot if connected, red if not.
- **Claude AI** -- green if API key is configured, yellow if not (optional).
- **Supabase DB** -- green if connected, yellow if not (optional).
- **Env** -- badge showing `DEMO` or `LIVE` (red warning for live).
- **RF Model** -- badge showing feature count, tree count, and whether the model is trained or using heuristic mode.

**Trading Parameters**: Editable form with these fields:

| Parameter | Description | Default |
|-----------|-------------|---------|
| **Max Bet ($)** | Maximum dollar amount the bot can wager on a single trade. | $25 |
| **Min Edge (%)** | Minimum percentage difference between model probability and market price required to enter a trade. | 8% |
| **Daily Loss Limit ($)** | Bot stops trading for the day once cumulative losses hit this amount. | $100 |
| **Max Positions** | Maximum number of simultaneously open positions allowed. | 10 |
| **Markets to Scan** | Number of top events (by volume) fetched from Kalshi each on-demand scan cycle. | 20 |
| **Kelly Fraction** | Fraction of the Kelly Criterion used for position sizing. 0.5 = Half-Kelly (safer, recommended). 1.0 = Full Kelly (aggressive, higher variance). 0.25 = Quarter-Kelly (very conservative). | 0.5 |

- **Save Settings** button persists changes.
- **Reset Daily P&L** button resets the daily loss counter (useful after manual intervention).

**Notifications**: Shows Slack and Discord webhook configuration status. A **Test Notifications** button sends a test message to configured channels. Instructions explain how to set `SLACK_WEBHOOK_URL` and `DISCORD_WEBHOOK_URL` in `.env`.

**Setup**: Step-by-step setup instructions with copy buttons for terminal commands.

#### Right Column

**Paper Trader Status** (6 mini stat cards): Balance, Model status, Training Samples, Open Positions, Total Scans, Auto-Scan status.

**24/7 Automation**:

- **Auto-Scan (Paper)** toggle: Enables background scanning every N seconds. Scans markets and records shadow trades automatically using real data.
- **Scan Interval** input: Seconds between scans (minimum 30, default 60).
- **Auto-Trade (Live)** toggle: Executes real trades on Kalshi. Requires a confirmation dialog because it uses real money. Shown with a red warning.
- **Recent Scans** log: Shows timestamp, events/markets scanned, and entries/exits per scan.

**Trade Logs**: Tabbed view with `Shadow` and `Live` tabs.

- **Shadow tab**: Shows open shadow positions (ticker, side, contracts, entry price, model probability), completed trades table (Ticker, Side, Entry, Exit, P&L, Result), and a recent activity feed.
- **Live tab**: Shows completed live trades and recent live activity feed.
- Auto-refreshes every 30 seconds when auto-scan is active.

**Model Retrain Schedule**:
- Status indicator (Active/Inactive).
- Next scheduled run timestamp.
- **Days** input: Comma-separated days of the week (e.g., `mon,wed,fri`).
- **Hour** input: Hour in UTC (0-23).
- **Update Schedule** button saves the cron schedule.
- **Retrain Now** button triggers an immediate retrain using the latest settled market data.

**Strategy Reference**: Read-only numbered list summarizing the complete strategy:
1. Model configuration (RF trees + GB, features per tree)
2. Kelly Criterion sizing formula and fraction
3. Risk limits (max bet, daily loss, max positions)
4. Entry rule (edge threshold)
5. Sharpe Ratio formula and rating scale
6. Log return calculation
7. Exit rule (price convergence or time stop)
8. Cumulative training data persistence

---

## Glossary

Alphabetical list of all abbreviations and terms used throughout the application:

| Term | Full Name | Definition |
|------|-----------|------------|
| **Ask** | Ask Price | The lowest price a seller will accept. The price you pay to buy immediately. |
| **Avg** | Average | Average entry price per contract (in cents). |
| **Bid** | Bid Price | The highest price a buyer will pay. The price you receive if you sell immediately. |
| **Conf** | Confidence | The model's certainty in its prediction, as a percentage. Minimum 70% required. |
| **CV** | Cross-Validation | A model evaluation technique where training data is split into k folds; the model trains on k-1 and tests on 1, rotating through all folds. |
| **Edge** | Edge | The difference between model probability and market price. Positive edge = market is undervalued. |
| **GB** | Gradient Boosting | An ensemble ML method that builds trees sequentially, each correcting the errors of the previous. |
| **Kelly Fraction** | Kelly Criterion Fraction | The multiplier applied to the Kelly Criterion formula for position sizing. Half-Kelly (0.5) is the recommended default. |
| **Log R** | Log Return | Logarithmic return: `ln(exit_price / entry_price)`. Preferred over simple returns because log returns are additive across trades. |
| **MAE** | Max Adverse Excursion | The worst unrealized loss (drawdown) during a trade's lifetime, before it was closed. |
| **MFE** | Max Favorable Excursion | The best unrealized gain during a trade's lifetime. If MFE >> realized P&L, you may be exiting too early. |
| **Model** | Model Probability | The probability estimated by the RF+GB ensemble model. |
| **OI** | **Open Interest** | The total number of outstanding contracts that have not yet been settled. Unlike volume (which counts every trade), OI counts only contracts currently held open. OI increases when a new contract is created between a buyer and seller, and decreases when both sides close or the market settles. High OI means many participants have capital at stake. |
| **OOB** | Out-of-Bag Score | Random Forest's built-in validation metric. Each tree is trained on a bootstrap sample; the OOB score measures accuracy on the samples that were left out of that tree's training set. |
| **P&L** | Profit and Loss | Total dollar gain or loss from trading. |
| **PF** | Profit Factor | Gross profit divided by gross loss. PF > 1 = profitable. PF > 1.5 = good. PF > 2 = excellent. |
| **Qty** | Quantity | Number of contracts held. |
| **RF** | Random Forest | An ensemble ML method that builds many independent decision trees and averages their predictions. |
| **Sharpe Ratio** | Sharpe Ratio | Risk-adjusted return: `(avg return - risk-free rate) / std deviation`. < 1 bad, 1-2 good, > 2 excellent. |
| **Side** | Side | `YES` = betting the event will happen. `NO` = betting it will not happen. |
| **Spread** | Bid-Ask Spread | Ask minus Bid, in cents. Lower spread = more liquid market = easier to trade. |
| **Src** | Source | Which component generated the signal: `RF` (Random Forest) or `AI` (Claude). |
| **Vol** | Volume | Total number of contracts traded on a market since it opened. Higher = more active. |
| **Win Rate** | Win Rate | Percentage of resolved trades that were profitable: `wins / total_trades`. |

---

## Trading Strategy

### 1. Feature Engineering (106 Features)

The model extracts 106 numerical features from each market, organized into 8 categories:

| Category | Count | Examples |
|----------|-------|---------|
| Price | 18 | Bid/ask prices, log odds, midpoint, spread, price buckets |
| Volume and Liquidity | 14 | Log volume, OI ratio, dollar volume, volume intensity |
| Time Decay | 14 | Days/hours/minutes to expiry, theta proxy, time urgency |
| Orderbook Imbalance | 12 | Bid/ask pressure, microprice, weighted midpoint |
| Cross-Market Efficiency | 10 | Overround, vig, arbitrage spread, dislocation score |
| Momentum Proxies | 8 | Price momentum, mean reversion signal, conviction score |
| Historical Momentum | 14 | Trend strength, volatility, volume trend, skew |
| Interaction/Cross Features | 16 | Price x volume, edge x liquidity, risk-adjusted edge |

### 2. Ensemble Model

- **Random Forest**: 500 trees, max_depth=10, sqrt(features) per split, OOB scoring enabled.
- **Gradient Boosting**: 150 trees, learning rate 0.01.
- **Ensemble Weight**: 70% RF + 30% GB (optimized via Brier score grid search).
- **Fallback**: When untrained, uses a 12-signal heuristic probability estimate.
- **Calibration**: CalibratedClassifierCV for probability calibration.
- **Scaling**: StandardScaler for feature normalization.

### 3. Entry Rules

```
if market_price <= model_probability * 0.5:     # Market is 2x undervalued
    if model_confidence >= 0.70:                 # Model is confident
        if edge >= min_edge_threshold:           # Edge exceeds minimum
            buy()
```

### 4. Exit Rules

```
if market_price >= model_probability * 0.9:     # Price converged to ~fair value
    sell()

if days_to_expiry <= 7:                         # Time stop
    sell()
```

### 5. Kelly Criterion Position Sizing

```
f* = (b * p - q) / b

Where:
  f* = fraction of bankroll to wager
  b  = odds (payout ratio)
  p  = model probability of winning
  q  = 1 - p (probability of losing)

Actual size = f* * kelly_fraction * bankroll
```

The Kelly fraction (default 0.5 = Half-Kelly) reduces variance at the cost of slightly lower expected growth.

### 6. Risk Management

- **Max bet per trade**: Configurable (default $25).
- **Daily loss limit**: Bot stops trading when cumulative daily losses exceed the limit (default $100).
- **Max open positions**: Prevents over-concentration (default 10).
- **Pre-trade risk check**: Every signal is validated against all limits before execution.

---

## API Endpoints

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| Status | `/api/status` | GET | Bot status, connections, config, model info |
| Portfolio | `/api/portfolio` | GET | Kalshi portfolio summary (balance) |
| Positions | `/api/positions` | GET | Open positions with unrealized P&L |
| Scanning | `/api/scan` | POST | Run on-demand market scan (RF + optional AI) |
| Signals | `/api/signals` | GET | Cached scan results (entry + exit signals) |
| Events | `/api/events` | GET | Top Kalshi events by volume (configurable limit) |
| Market | `/api/market/{ticker}` | GET | Full 106-feature analysis for a single market |
| Trading | `/api/trade` | POST | Place a trade on Kalshi |
| Orders | `/api/orders` | GET | Open orders on Kalshi |
| Cancel | `/api/order/{order_id}` | DELETE | Cancel a pending order |
| Performance | `/api/performance` | GET | Metrics, equity curve, and trade log |
| Performance | `/api/performance/by-category` | GET | P&L breakdown by market category |
| Features | `/api/model/features` | GET | Feature importance rankings |
| Arbitrage | `/api/arbitrage` | GET | Cross-platform arbitrage scan (Kalshi vs DraftKings) |
| Config | `/api/config` | PATCH | Update trading parameters at runtime |
| Backtest | `/api/backtest` | POST | Run historical backtest |
| Sweep | `/api/backtest/sweep` | POST | Parameter sweep (25 combinations) |
| Paper | `/api/paper` | GET | Paper trading state (balance, positions, metrics) |
| Paper | `/api/paper/configure` | POST | Reset paper trader with a new balance |
| Paper | `/api/paper/add-funds` | POST | Add demo funds without resetting |
| Paper | `/api/paper/scan` | POST | Run one paper trading scan cycle |
| Paper | `/api/paper/train` | POST | Train model on settled market history |
| Risk | `/api/risk/reset-daily` | POST | Reset the daily P&L counter |
| Auto-Scan | `/api/autoscan` | POST | Toggle auto-scan (configurable interval) |
| Auto-Trade | `/api/autotrade` | POST | Toggle live auto-trading |
| Auto-Scan | `/api/autoscan/status` | GET | Scheduler status and scan log |
| Trade Logs | `/api/trades/shadow` | GET | Shadow/paper trade log with positions |
| Trade Logs | `/api/trades/live` | GET | Live trade log |
| History | `/api/history/trades` | GET | Persistent trade history (Supabase) |
| History | `/api/history/scans` | GET | Persistent scan log (Supabase) |
| History | `/api/history/performance` | GET | Performance snapshots over time (Supabase) |
| History | `/api/history/training` | GET | Model training run history (Supabase) |
| Webhook | `/api/webhook` | POST | External trigger for scans/retrains (authenticated) |
| Notifications | `/api/notifications/test` | POST | Send test notification to Slack/Discord |
| Notifications | `/api/notifications/config` | GET | Notification channel configuration status |
| Export | `/api/export/trades` | GET | CSV download of trade history |
| Heatmap | `/api/portfolio/heatmap` | GET | Portfolio positions grouped by category |
| Notes | `/api/trade/{trade_index}/notes` | PATCH | Update notes on a specific trade |
| Retrain | `/api/retrain/schedule` | GET | Get current retrain schedule |
| Retrain | `/api/retrain/schedule` | POST | Update retrain schedule (days + hour) |
| Retrain | `/api/retrain/now` | POST | Trigger immediate model retrain |

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KALSHI_API_KEY_ID` | Yes | -- | API key ID from Kalshi settings |
| `KALSHI_PRIVATE_KEY_PATH` | Yes* | `./kalshi_private_key.pem` | Path to PEM private key file (local dev) |
| `KALSHI_PRIVATE_KEY` | Yes* | -- | Raw PEM content as env var (cloud deploy, e.g., Railway) |
| `KALSHI_USE_DEMO` | No | `true` | `true` = demo API, `false` = real money production API |
| `ANTHROPIC_API_KEY` | No | -- | Enables Claude AI hybrid analysis alongside RF |
| `SUPABASE_URL` | No | -- | Supabase project URL for persistent storage |
| `SUPABASE_KEY` | No | -- | Supabase service role key |
| `MAX_BET_AMOUNT_CENTS` | No | `2500` | Max per-trade wager in cents ($25) |
| `MIN_EDGE_THRESHOLD` | No | `0.08` | Minimum model-vs-market edge (8%) |
| `MAX_DAILY_LOSS_CENTS` | No | `10000` | Daily stop-loss in cents ($100) |
| `MAX_OPEN_POSITIONS` | No | `10` | Max simultaneous positions |
| `MAX_EVENTS_TO_ANALYZE` | No | `20` | Events per on-demand scan |
| `KELLY_FRACTION` | No | `0.5` | Kelly Criterion multiplier (0.5 = Half-Kelly) |
| `SLACK_WEBHOOK_URL` | No | -- | Slack incoming webhook URL for trade alerts |
| `DISCORD_WEBHOOK_URL` | No | -- | Discord webhook URL for trade alerts |
| `WEBHOOK_SECRET` | No | -- | Shared secret for authenticating inbound `/api/webhook` calls |
| `RETRAIN_DAYS` | No | `mon,wed,fri` | Days to auto-retrain (APScheduler cron format) |
| `RETRAIN_HOUR` | No | `3` | Hour in UTC to run auto-retrain (0-23) |

*One of `KALSHI_PRIVATE_KEY_PATH` or `KALSHI_PRIVATE_KEY` is required. The env var takes priority for cloud deployments.

For the frontend (Vercel), set `VITE_API_URL` in `frontend/.env` to point to your backend URL.

---

## Database (Supabase)

Supabase is **optional** but recommended -- it persists all data across server restarts and redeploys.

| Table | Purpose |
|-------|---------|
| `trades` | Every paper and live trade with full metrics |
| `scan_logs` | Every scan with signal counts, entries, exits |
| `paper_state` | Paper trader balance, positions, scan count |
| `performance_snapshots` | Periodic Sharpe, win rate, equity snapshots |
| `model_training_runs` | CV accuracy, OOB score, feature importance per training |

### Setup

1. Create a project at [supabase.com](https://supabase.com).
2. Go to **SQL Editor** and run the schema from `supabase/schema.sql` to create the required tables.
3. Add `SUPABASE_URL` and `SUPABASE_KEY` (service role key) to your `.env`.
4. The bot auto-connects on startup. All database methods gracefully no-op when Supabase is not configured.

---

## Deployment

### Backend on Railway

```bash
railway login
railway init
railway up
```

Add your `.env` variables in Railway's dashboard under **Variables**. For the private key, paste the full PEM content into a `KALSHI_PRIVATE_KEY` variable (the bot reads from the env var when no file is found).

Railway runs the server using the `Procfile`:

```
web: uvicorn bot.server:app --host 0.0.0.0 --port $PORT
```

**Auto-deploy**: Connect your GitHub repo in Railway's dashboard. Every push to `main` triggers a new deploy.

### Frontend on Vercel

```bash
cd frontend
npx vercel
```

Set the environment variable `VITE_API_URL` to your Railway backend URL (e.g., `https://your-app.up.railway.app`).

**Auto-deploy**: Connect your GitHub repo in Vercel. Set root directory to `frontend`. Every push auto-deploys.

### Local with PM2

```bash
npm install -g pm2

pm2 start "uvicorn bot.server:app --port 8000" --name bot-api
pm2 start "npm run dev" --name bot-ui --cwd frontend

pm2 save
pm2 startup
```

---

## Project Structure

```
prediction-market-bot/
├── bot/
│   ├── server.py           # FastAPI REST API (40+ endpoints)
│   ├── rf_model.py         # RF+GB ensemble, 106 features, signal generation
│   ├── backtester.py       # Historical backtesting, parameter sweep, paper trader
│   ├── kalshi_client.py    # Kalshi Trade API v2 client (RSA-PSS auth)
│   ├── draftkings_client.py # DraftKings scraper for cross-platform arbitrage
│   ├── analyzer.py         # Claude AI market analysis (optional)
│   ├── performance.py      # Sharpe, log returns, MAE/MFE tracking
│   ├── risk_manager.py     # Position limits, daily loss, Kelly sizing
│   ├── arbitrage.py        # Cross-platform arbitrage detection
│   ├── database.py         # Supabase persistence layer (optional)
│   ├── notifier.py         # Slack/Discord webhook notifications
│   ├── models.py           # Pydantic data models
│   ├── config.py           # Environment configuration
│   └── main.py             # CLI entry point
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # Main app with 7 tabs
│   │   ├── api.js          # API client (all endpoint methods)
│   │   └── components/
│   │       ├── Dashboard.jsx
│   │       ├── Signals.jsx
│   │       ├── Portfolio.jsx
│   │       ├── Markets.jsx
│   │       ├── Performance.jsx
│   │       ├── Testing.jsx
│   │       ├── Settings.jsx
│   │       ├── Toast.jsx
│   │       └── Tooltip.jsx
│   └── package.json
├── supabase/               # Database schema SQL
├── data/                   # Local data directory (JSON fallback)
├── .env.example            # Environment variable template
├── pyproject.toml          # Python project config
├── requirements.txt        # Python dependencies
├── Procfile                # Railway deployment command
├── railway.toml            # Railway config
└── README.md
```

---

## Risk Disclaimer

This bot is for **educational and research purposes**. Prediction market trading involves real financial risk. Always start with **demo mode** (`KALSHI_USE_DEMO=true`) and validate with **shadow trading** before using real money. Past backtesting performance does not guarantee future results.

---

## License

MIT
