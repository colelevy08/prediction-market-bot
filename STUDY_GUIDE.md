# Prediction Market Bot — Complete Study Guide

> **Your goal:** By the end of this guide, you should be able to rebuild this entire bot from scratch without AI help.
> 
> **How to use this guide:** Follow the chapters in order. Each chapter tells you exactly which file to open and what to look for. Do not skip ahead — each chapter builds on the last.
> 
> **Assumed knowledge:** None. Zero. If you know what a `.py` file is, you're ready.

---

## How This Bot Makes Money (The 30-Second Version)

Kalshi is a prediction market exchange where people bet on things like "Will Bitcoin be higher in 15 minutes?" with YES/NO contracts that pay $1.00 if correct, $0 if wrong.

The market prices these contracts. If the market says there's a **45% chance** Bitcoin goes up but our math says **52%**, we can buy the YES contract at 45¢ and expect to collect more than we paid on average. That 7-cent difference is our **edge**.

Do this hundreds of times with smart bet sizing and you have a profitable strategy. This bot automates every part of that process.

---

## Table of Contents

| Chapter | Topic | File(s) |
|---------|-------|---------|
| 0 | **Running the Bot Without Claude** | `botctl`, `/tmp/shadow_snap.py` |
| 1 | What Are Prediction Markets? | Conceptual |
| 2 | How Python Programs Work | `bot/main.py` |
| 3 | Configuration & Environment Variables | `bot/config.py` |
| 4 | Data Structures: Modeling the World | `bot/models.py` |
| 5 | Talking to APIs: The Kalshi Client | `bot/kalshi_client.py` |
| 6 | Getting Live Prices | `bot/crypto_feed.py` |
| 7 | Machine Learning Signals | `bot/rf_model.py` |
| 8 | Risk Management & Kelly Criterion | `bot/risk_manager.py` |
| 9 | Tracking Performance | `bot/performance.py` |
| 10 | Testing Strategies Safely | `bot/backtester.py` |
| 11 | The Web Server & Trading Brain | `bot/server.py` (Part 1) |
| 12 | The SMART Signal: Computing Edge | `bot/server.py` (Part 2) |
| 13 | Shadow Bots: A/B Testing Strategies | `bot/server.py` (Part 3) |
| 14 | Auto-Deploy: Letting the Best Bot Win | `/tmp/autodeploy.py` |
| 15 | Supporting Systems | `bot/database.py`, `bot/notifier.py`, `bot/arbitrage.py` |
| 16 | The Full Picture | Everything together |

---

## CHAPTER 0 — Running the Bot Without Claude

This chapter is a pure operations manual. No theory. Just commands.

---

### Starting the bot from scratch

```bash
cd ~/Desktop/Development/prediction-market-bot

./botctl start            # starts server, auto-enables autoscan
./botctl autotrade on     # enables real money trading (does NOT auto-enable on restart)
```

That's it. The bot is now:
- Scanning every 0.15 seconds (portfolio/balance cached at 1s)
- Running 90 shadow bots in parallel (Wave 6: 49 bots, Wave 8: 41 bots)
- Placing real orders when signals fire

---

### Daily check-in (30 seconds)

```bash
./botctl status           # is everything alive? balance? flags?
python3 /tmp/shadow_snap.py   # how are the 90 shadow bots doing?
```

---

### Reading the shadow snapshot

```
BOT          PARAMS                      BAL       PnL  OPEN    W    L      WR    SHRP
H3-g10       W90_E10_K30 [CTL]     $32825237  $32824903     0  218  143    60%    0.16
```

| Column | Meaning |
|--------|---------|
| BOT | Bot label (e.g. H3-g10 = Hypothesis group 3, variant g10) |
| PARAMS | What this bot is testing (signal weight, edge floor, Kelly fraction, optional gates) |
| BAL | Virtual balance (all bots start at $333.23) |
| PnL | Profit or loss since start |
| OPEN | Positions not yet settled |
| W/L | Settled wins / losses |
| WR | Win rate % |
| SHRP | Sharpe ratio (risk-adjusted return; higher = better consistency) |

Wave 6 (49 bots) tests WP gates, Kelly fractions, edge floors, spread filters, timing gates, and consensus filters. Wave 8 (41 bots) extends the sweep with finer granularity on edge and Kelly at specific WP values.

**What to look for:**
- H3-g10 (W90_E10_K30) = current live bot config — compare all others against it
- WR > 60% with 20+ trades = strategy is working
- Bot clearly ahead in both PnL and Sharpe = likely superior configuration

---

---

### Emergency stop

```bash
./botctl stop-trading     # disables autotrade + autoscan (server stays up)
./botctl stop             # kills server entirely
```

**Hard floor:** If total portfolio value (cash + open positions) drops below **$200**, disable both loops immediately:
```bash
curl -X POST http://localhost:8000/api/autotrade -H "Content-Type: application/json" -d '{"enabled": false}'
curl -X POST http://localhost:8000/api/autoscan  -H "Content-Type: application/json" -d '{"enabled": false}'
```
The floor is based on total portfolio — not cash alone — to avoid false triggers while positions are open.

---

### Key .env settings — what to touch and what not to

**Safe to adjust:**
```bash
MAX_BET_AMOUNT_CENTS=1500         # floor bet size ($15); dynamic cap = max(this, 10% of balance)
MAX_OPEN_POSITIONS=5              # max simultaneous open positions
MAX_DAILY_LOSS_CENTS=3440         # daily loss limit (scales dynamically via MAX_DRAWDOWN_PCT)
KELLY_FRACTION=0.15               # 15% Kelly base (further scaled by vol, spread, correlation)
```

**Do not touch without understanding the impact:**
```bash
KALSHI_USE_DEMO=false             # this is real money — set true to test
AUTO_SCAN_INTERVAL=0.15           # 0.15s scan; portfolio cached at 1s — do not raise without testing
MAX_DRAWDOWN_PCT=0.20             # circuit breaker — raising this increases max loss
```

To apply .env changes: `./botctl stop && ./botctl start && ./botctl autotrade on`

---

### Runtime config changes (no restart needed)

```bash
./botctl config '{"max_bet_amount_cents": 1000}'
./botctl config '{"live_min_win_prob": 0.65, "live_min_edge_cents": 4}'
./botctl config '{"max_open_positions": 5}'
```

Resets to .env values on next server restart.

---

### Checking open positions and recent trades

```bash
./botctl positions        # open positions right now
./botctl trades           # last 20 settled trades
./botctl performance      # P&L, win rate, Sharpe ratio
./botctl portfolio        # balance + all positions
```

---

### Log files

| File | What it contains |
|------|-----------------|
| `/tmp/predbot.log` | Main server log (errors, trade entries, WS events, settlements) |
| `/tmp/shadow_monitor.log` | Persistent history of all shadow snapshots (also written to `data/shadow_monitor.log`) |
| `data/bot.db` | SQLite database — all live trades, shadow snapshots, order fill records |

```bash
./botctl logs                    # last 50 lines of server log
tail -f /tmp/predbot.log         # live stream
tail -f /tmp/shadow_monitor.log  # watch shadow bot snapshots accumulate
```

---

### Scheduled monitoring (set-and-forget)

In a Claude Code session, run:
```
/loop 5m Check live bot status and persist to DB
/loop 15m python3 /tmp/shadow_snap.py
```
The 5-min loop monitors balance and flags issues. The 15-min shadow snap tracks bot tournament progress. Deployment decisions are made manually after reviewing wave results.

---

### Querying your trade database directly

```bash
sqlite3 data/bot.db

# How many live trades total?
SELECT COUNT(*) FROM trades WHERE mode='live';

# Win rate on live trades
SELECT ROUND(AVG(won)*100, 1) || '%' FROM trades WHERE mode='live';

# P&L by coin
SELECT SUBSTR(ticker, 1, 5) AS coin, COUNT(*) AS trades,
       SUM(pnl_cents)/100.0 AS pnl_usd
FROM trades WHERE mode='live'
GROUP BY coin ORDER BY pnl_usd DESC;

.quit
```

---

### Key API endpoints (for debugging without botctl)

```bash
# Everything at once
curl -s http://localhost:8000/api/status | python3 -m json.tool

# Shadow bot details (use the bot's api_key, e.g. "g10", "xco", "r")
curl http://localhost:8000/api/shadow/g10    # H3-g10 (live bot config control)
curl http://localhost:8000/api/shadow/xco    # H11-xco (high-edge control)
curl http://localhost:8000/api/shadow/all    # all 90 shadow bots at once

# Force enable autotrade
curl -X POST http://localhost:8000/api/autotrade \
  -H "Content-Type: application/json" -d '{"enabled": true}'

# Force enable autoscan
curl -X POST http://localhost:8000/api/autoscan \
  -H "Content-Type: application/json" -d '{"enabled": true}'
```

---

### Troubleshooting quick reference

| Problem | Command |
|---------|---------|
| Server not responding | `./botctl start` |
| Port 8000 already in use | `pkill -f "uvicorn bot.server:app"` then `./botctl start` |
| No trades firing | Check `./botctl status` → `auto_trade_enabled` must be `true` |
| Shadow bots all zero | `curl -X POST localhost:8000/api/shadow -d '{"enabled":true}' -H 'Content-Type: application/json'` |
| Thursday 3–5 AM ET | Kalshi maintenance window — no markets available |
| Daily loss limit hit | Check `./botctl status` → `daily_loss_limit_hit` |

---

## CHAPTER 1 — What Are Prediction Markets?

### What you'll learn
- What prediction markets are and why they exist
- How binary contracts work
- What "edge" means in gambling and trading
- Why bet sizing matters more than win rate

### The Concept

A **prediction market** is an exchange where you bet on real-world outcomes. Unlike sports betting where the house sets odds, prediction markets let participants trade against each other — like a stock market for beliefs.

**Kalshi** is the largest regulated prediction market in the US. Their most popular products are crypto 15-minute markets:

> "Will Bitcoin's price be higher at 2:30 PM than it is right now at 2:15 PM?"

- If YES: You win $1.00 per contract
- If NO: You get $0

The market prices this at roughly 45–55 cents, because it's nearly a coin flip. But "nearly" is where we make money.

### Edge: The Core Concept

**Edge** = what we think the true probability is, minus what the market is charging.

```
Example:
  Market price:      42¢  (implies 42% chance of YES)
  Our fair value:    51¢  (we calculate 51% chance of YES)
  Edge:               9¢  per contract

If we buy 100 contracts at 42¢:
  Cost:             $42.00
  If we're right about 51%:
    Expected winnings: 51 × $1.00 = $51.00
  Expected profit:    $9.00
```

This is **positive expected value (EV)**. We don't win every time, but over hundreds of bets, we profit.

### Kelly Criterion: How Much to Bet

Knowing you have an edge isn't enough — you can go broke betting too much on coin flips. The **Kelly Criterion** solves this:

```
f = (p × b - (1 - p)) / b

Where:
  f = fraction of your bankroll to bet
  p = your estimated probability of winning
  b = net odds (how much you win per dollar risked)

Example:
  p = 0.51 (we think 51% chance of winning)
  b = (100 - 42) / 42 = 1.38 (win $1.38 for every $1 risked at 42¢)

  f = (0.51 × 1.38 - 0.49) / 1.38
  f = (0.703 - 0.49) / 1.38
  f = 0.154 = 15.4% of bankroll

We'd bet 15.4% of our money on this trade.
```

In practice we use a fraction of Kelly — the live bot uses **30% Kelly base** (K30), then further scales by volatility, spread, and position correlation. A hard cap of **10% of balance** prevents any single bet from exceeding one-tenth of the portfolio regardless of what Kelly suggests.

### Checkpoint ✓

Before moving on, you should be able to answer:
1. What does a YES contract pay if it resolves YES?
2. If a market is priced at 40¢ and you calculate 55% true probability, what is your edge?
3. Why would you NOT bet your entire bankroll even with positive edge?

---

## CHAPTER 2 — How Python Programs Work

### File to open: `bot/main.py`

### What you'll learn
- What a Python module is
- What functions and classes are
- What "entry points" are
- How our server starts up

### Read the file

Open `bot/main.py`. This is the simplest file in the project. It imports the other modules and provides a command-line interface.

Key concepts to notice:
- `import` statements: pulling in code from other files
- `if __name__ == "__main__"`: this code only runs when you execute the file directly
- Functions that call other functions: how code is organized into logical units

### What to understand

A Python project is a collection of `.py` files. Each file is a **module**. Modules import from each other to share code. `main.py` is the entry point — the first thing that runs.

Our server actually starts differently (via `uvicorn` — explained in Chapter 11), but `main.py` shows you the module structure.

### Checkpoint ✓

1. What does `import` do?
2. What is a function?
3. If `main.py` imports from `bot.server`, what does that mean?

---

## CHAPTER 3 — Configuration & Environment Variables

### File to open: `bot/config.py`

### What you'll learn
- What environment variables are and why we use them
- The singleton pattern in Python
- How to safely store secrets (API keys)
- Why configuration should be separate from code

### The Problem With Hardcoding

Imagine you hardcoded your Kalshi password directly in the code:
```python
password = "mySecretPassword123"  # BAD! Never do this!
```

If you ever share your code or push it to GitHub, your password is public. Anyone can drain your account.

**Environment variables** solve this. They live in a `.env` file that never gets shared:
```
# .env file (never committed to git)
KALSHI_API_KEY=abc123secret
KALSHI_USE_DEMO=false
MAX_BET_AMOUNT_CENTS=1500
```

The code reads these at startup:
```python
api_key = os.environ.get("KALSHI_API_KEY")
```

### Read the file

Open `bot/config.py`. Notice:
- It reads dozens of settings from environment variables
- It has default values for everything (so the bot works even if you forget to set something)
- There's a single `config` object that all other files import — this is the **singleton pattern**

The singleton pattern ensures you only have one config object shared across the entire application. If every file created its own config, they might disagree on settings.

### Checkpoint ✓

1. Why do we use environment variables instead of hardcoding values?
2. What is the `.env` file and why is it not committed to git?
3. What is a singleton pattern and why is it useful?

---

## CHAPTER 4 — Data Structures: Modeling the World

### File to open: `bot/models.py`

### What you'll learn
- What a data model is
- Pydantic: Python's data validation library
- How we represent markets, events, and orders in code
- The difference between a dictionary and a typed object

### Why We Need Models

The Kalshi API sends us raw JSON like this:
```json
{
  "ticker": "KXBTC15M-26MAR261430-30",
  "yes_bid": 42,
  "yes_ask": 44,
  "status": "open",
  "close_time": "2026-03-26T18:30:00Z"
}
```

We could use plain Python dictionaries to pass this data around, but that's error-prone:
```python
# Dictionary — no type safety
market["yes_bId"]  # Typo! But Python won't warn you until it crashes
```

Instead, we use **Pydantic models** — typed data containers that validate their contents:
```python
# Pydantic model — safe and clear
market.yes_bid  # If you typo this, your editor warns you immediately
```

### Read the file

Open `bot/models.py`. Look for:
- `class Market(BaseModel)`: represents one market (e.g., BTC 15M at 2:30)
- `class Event(BaseModel)`: represents a group of related markets (e.g., all BTC 15M markets for one day)
- `class OrderRequest`: what we send to Kalshi when placing a trade
- `class Position`: a contract we own right now

These classes are the "language" the rest of the code speaks. Every time data moves between files, it's in these shapes.

### Checkpoint ✓

1. What is a Pydantic model and why is it better than a plain dictionary?
2. What is the difference between a `Market` and an `Event`?
3. What fields would an `OrderRequest` need? (side, price, count, ticker)

---

## CHAPTER 5 — Talking to APIs: The Kalshi Client

### File to open: `bot/kalshi_client.py`

### What you'll learn
- What an API is (in depth)
- How HTTP requests work (GET, POST, DELETE)
- Cryptographic authentication (signing requests)
- Error handling and retries
- Rate limiting and circuit breakers

### What Is an API?

API = Application Programming Interface. It's a way for two programs to talk to each other.

Kalshi is a company with servers running their exchange. We communicate with those servers by sending **HTTP requests** — the same protocol your browser uses to load webpages.

```
Our bot                    Kalshi's servers
    |                           |
    |  GET /events               |   "Give me all active markets"
    |-------------------------->|
    |                           |
    |  { events: [...] }        |   Here are 500 markets
    |<--------------------------|
    |                           |
    |  POST /portfolio/orders   |   "Place this order"
    |-------------------------->|
    |  { order_id: "abc123" }   |   Order placed, here's the ID
    |<--------------------------|
```

### Authentication: Proving Who You Are

Kalshi requires every request to be **cryptographically signed**. Here's why:

Without signing, anyone who intercepts your request could:
1. Replay it (place the same order again)
2. Modify it (change the price or quantity)

With RSA/ECDSA signing:
1. You generate a **private key** (a long random number — keep it secret)
2. You register a **public key** with Kalshi (mathematically derived from your private key)
3. For every request, you create a "signature" by running the request data through a math function using your private key
4. Kalshi verifies the signature using your public key — only someone with your private key could have created that signature

It's like a wax seal: the signature ring (private key) is yours alone. Anyone can verify the seal (public key) but can't forge it.

### Read the file

Find the `_sign_request` method. It:
1. Takes the timestamp, HTTP method, and URL path
2. Encodes them as bytes
3. Runs them through RSA-PSS or ECDSA signing (depending on your key type)
4. Returns a base64-encoded signature

Find `get_events`. It:
1. Calls the `/trade-api/v2/events` endpoint
2. Pages through results (Kalshi returns 200 at a time)
3. Converts raw JSON into `Event` objects (using our models)

Find the circuit breaker logic. If the API fails 5 times in a row, the circuit "opens" and we stop hammering a broken connection.

### Checkpoint ✓

1. What is the difference between GET and POST?
2. Why do we need to sign requests?
3. What does a circuit breaker do?
4. What is rate limiting and how does exponential backoff help?

---

## CHAPTER 6 — Getting Live Prices

### File to open: `bot/crypto_feed.py`

### What you'll learn
- Why we need live prices separate from Kalshi
- CoinGecko API: free real-time crypto prices
- Binance API: funding rates as a market signal
- Caching: why we don't fetch every second
- What a funding rate is and why it matters

### Why Live Prices?

Kalshi markets ask "will BTC go UP in the next 15 minutes?" The market price (e.g., 48¢) represents what traders collectively think. But we want our OWN estimate.

Our estimate uses:
1. **Current BTC price**: Is it trending up or down right now?
2. **Recent price change**: How much has it moved in the last few minutes?
3. **Funding rate**: Are leveraged traders bullish or bearish?

### CoinGecko

CoinGecko is a free API that provides live crypto prices. We call it every few seconds to get the latest prices for BTC, ETH, XRP, SOL, BNB, DOGE, and HYPE.

```python
# Simplified example of what we do
import requests
response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd")
btc_price = response.json()["bitcoin"]["usd"]  # e.g., 87432.50
```

### Funding Rates

On crypto futures exchanges like Binance, **long** traders (betting price goes up) pay a fee to **short** traders (betting price goes down) when everyone is long — and vice versa. This fee, paid every 8 hours, is the **funding rate**.

Why it matters: If funding is very positive (longs paying shorts heavily), it means the market is overcrowded with bulls. **Contrarian signal**: overcrowded longs often precede a price drop.

```
Funding rate interpretation:
  Very positive (+0.1%)  → Too many bulls → price may drop → we lean NO
  Very negative (-0.1%)  → Too many bears → price may rise → we lean YES
  Near zero              → Balanced → no signal
```

### Checkpoint ✓

1. Why do we need live crypto prices if Kalshi already has market prices?
2. What is a funding rate and why is it a useful signal?
3. Why do we cache prices instead of fetching every tick?

---

## CHAPTER 7 — Machine Learning Signals

### File to open: `bot/rf_model.py`

### What you'll learn
- What machine learning is (simply explained)
- Random Forest: how it works
- Gradient Boosting: a more powerful alternative
- Why we use an ensemble of both
- Feature engineering: turning raw data into model inputs
- Training vs prediction
- Why more data = better model

### What Is Machine Learning?

Traditional programming: **you write the rules**.
```python
if price_went_up and funding_is_negative:
    bet_yes()
```

Machine learning: **the computer learns the rules from data**.
```
You give it 10,000 examples:
  "When features looked like THIS, the market went UP"
  "When features looked like THIS, the market went DOWN"

The model learns which patterns predict which outcome.
```

### Random Forest

A **Random Forest** is a collection of decision trees. Each tree is like a flowchart:
```
Is the spread < 3 cents?
├── YES: Is the volume > 500?
│   ├── YES: Predict UP (72%)
│   └── NO:  Predict DOWN (61%)
└── NO: Is momentum > 0.02?
    ├── YES: Predict UP (58%)
    └── NO:  Predict DOWN (67%)
```

A single tree is fragile (overfits to the training data). A "forest" of 50 trees each trained on slightly different data averages their predictions — much more robust.

### Gradient Boosting

Gradient Boosting builds trees **sequentially**: each new tree focuses on the errors the previous trees made. It's often more accurate than Random Forest but slower to train.

We use **both** and average their outputs — this is an **ensemble**. Two models that disagree help us understand our uncertainty.

### Features

The model doesn't see raw ticker symbols. It sees **features** — numerical summaries of the market:

```python
features = [
    yes_bid,           # 42 cents
    yes_ask,           # 44 cents
    spread,            # 2 cents
    implied_prob,      # 0.43 (market says 43% chance)
    volume,            # 1234 contracts traded today
    minutes_to_expiry, # 8.3 minutes left
    order_imbalance,   # 0.12 (more buy orders than sell)
    momentum_1,        # 0.003 (price trended up 0.3% recently)
    # ... 100+ more features
]
```

The model uses these 109 features to predict: **"What's the probability this market resolves YES?"**

### Checkpoint ✓

1. What is the difference between traditional programming and machine learning?
2. Why do we use 50 trees instead of 1?
3. What is a "feature" in machine learning?
4. Why do we need to train the model before we can use it?

---

## CHAPTER 8 — Risk Management & Kelly Criterion

### File to open: `bot/risk_manager.py`

### What you'll learn
- Pre-trade safety checks
- Kelly criterion (the full formula)
- Position limits and drawdown limits
- Why risk management matters more than entry signals
- Expected value vs realized value

### The Kelly Criterion (Deep Dive)

The Kelly formula tells you the **mathematically optimal fraction of your bankroll** to bet to maximize long-run wealth growth.

```
f* = (p × b - (1 - p)) / b

Variables:
  f* = optimal fraction of bankroll
  p  = probability of winning (your estimate)
  b  = net odds = (payout - cost) / cost
  (1 - p) = probability of losing

Example trade:
  Market price:  42¢  → we pay 42¢ per contract
  Payout if win: 100¢
  Net odds b:    (100 - 42) / 42 = 1.381

  Our estimate:  p = 0.52 (52% chance YES)

  f* = (0.52 × 1.381 - 0.48) / 1.381
     = (0.718 - 0.48) / 1.381
     = 0.238 / 1.381
     = 0.172  → bet 17.2% of bankroll

But we use "fractional Kelly" (K30 = 30% of full Kelly):
  kelly_base = f* × 0.30 = 17.2% × 0.30 = 5.2% of bankroll

Then further scaled:
  × vol_scale:    reduce in high-volatility markets
  × spread_adj:   reduce when bid/ask spread is wide
  × corr_deflator: reduce when holding correlated positions

Hard cap: single bet never exceeds 10% of balance.

Why fractional? Because our probability estimate (p=0.52) might be wrong.
Full Kelly on a wrong estimate can blow up your account.
K30 with additional scaling is conservative but captures most of the optimal growth rate.
```

### Safety Gates in risk_manager.py

The risk manager is checked BEFORE every real trade. It blocks trades when:
- Daily loss limit exceeded (e.g., down $500 today → stop trading)
- Too many open positions (e.g., 8 positions max)
- This specific position already open (no doubling up)
- Market spread too wide (bad fill risk)
- Price too high (>70¢ → likely already resolved direction, no edge)

### What Is Drawdown?

**Drawdown** = how far below your peak balance you currently are.

```
Balance history: $333 → $341 → $355 → $330 → $318 → $340 → $360

Peak: $355
Current: $318
Drawdown: ($355 - $318) / $355 = 10.4%
```

A large drawdown means your strategy is losing. Risk management limits how far you fall before stopping.

### Checkpoint ✓

1. What does the Kelly fraction represent?
2. Why do we use "quarter Kelly" instead of full Kelly?
3. What is drawdown and why does it matter?
4. Name 3 conditions the risk manager checks before allowing a trade.

---

## CHAPTER 9 — Tracking Performance

### File to open: `bot/performance.py`

### What you'll learn
- P&L (Profit and Loss) calculation
- Win rate and its limitations
- Sharpe ratio: risk-adjusted returns
- Why win rate alone is misleading
- How to evaluate a trading strategy

### P&L: Profit and Loss

Simple: `P&L = (exit_price - entry_price) × contracts`

```
Example:
  Bought 5 YES contracts at 42¢ each:   spent $2.10
  Market resolved YES, got 100¢ each:   received $5.00
  P&L = $5.00 - $2.10 = +$2.90

Another example:
  Bought 3 NO contracts at 55¢ each:    spent $1.65
  Market resolved YES (we were wrong):  received $0
  P&L = $0 - $1.65 = -$1.65
```

### Win Rate Is Not Enough

A strategy with 30% win rate can be MORE profitable than one with 70% win rate.

```
Strategy A: Win rate 70%, avg win $1, avg loss $3
  Expected per trade: 0.70 × $1 - 0.30 × $3 = $0.70 - $0.90 = -$0.20 (LOSING)

Strategy B: Win rate 30%, avg win $5, avg loss $1
  Expected per trade: 0.30 × $5 - 0.70 × $1 = $1.50 - $0.70 = +$0.80 (WINNING)
```

What matters is **Profit Per Trade (PPT)** = total P&L / number of trades.

### Sharpe Ratio

The **Sharpe ratio** measures risk-adjusted returns. It answers: "How much profit did we make per unit of risk taken?"

```
Sharpe = (Average Return - Risk Free Rate) / Standard Deviation of Returns

Interpretation:
  Sharpe < 1.0:  Acceptable but not great
  Sharpe > 1.5:  Good
  Sharpe > 2.0:  Excellent
  Sharpe > 3.0:  Exceptional (usually means something's wrong or it's temporary)
```

A high Sharpe ratio means consistent profits without wild swings. A low Sharpe means high volatility — even if profitable, you might blow up before you get there.

### Checkpoint ✓

1. Calculate P&L: bought 10 YES at 35¢, resolved NO. What's your P&L?
2. Why can a 30% win rate strategy be more profitable than 70%?
3. What does Sharpe ratio measure?
4. What is "Profit Per Trade" and why is it better than just win rate?

---

## CHAPTER 10 — Testing Strategies Safely

### File to open: `bot/backtester.py`

### What you'll learn
- What backtesting is and why it's essential
- Paper trading vs live trading
- Dangers of backtesting (overfitting, look-ahead bias)
- How to validate a strategy before risking real money

### What Is Backtesting?

Backtesting = running your strategy on **historical data** to see how it would have performed.

```
Step 1: Collect historical market data (prices, volumes, outcomes)
Step 2: Run your strategy as if you were trading live
Step 3: Measure the results (win rate, P&L, Sharpe ratio)
Step 4: Decide if the strategy is worth deploying
```

It's like a flight simulator — you practice before risking real planes.

### Paper Trading

**Paper trading** = running your strategy on LIVE data but with fake money.

The difference from backtesting:
- Backtesting uses historical data (might have biases)
- Paper trading uses live data in real-time (no hindsight bias)

Paper trading is the final test before going live. If a strategy makes fake money for a week, it's ready for real money.

### The Dangers of Backtesting

**Overfitting**: Your strategy learned to trade the specific historical data you tested on, not the underlying patterns. It performs great in backtests but fails in live trading.

Example: You find that whenever the BTC spread is 3 cents at exactly 2:17 PM on Tuesdays, the price goes up. This "pattern" might be pure coincidence in your 100-day dataset.

**Look-ahead bias**: Accidentally using future information in your backtest.

Example: Using "the high of the day" as a signal at 10 AM — but you can't know the day's high until the day ends.

**Survivorship bias**: Only testing on assets that survived. Companies that went bankrupt aren't in your test set, making results look better than reality.

### Checkpoint ✓

1. What is backtesting and why is it done before live trading?
2. What is overfitting?
3. Why is paper trading different from backtesting?
4. What is look-ahead bias?

---

## CHAPTER 11 — The Web Server & Trading Brain

### File to open: `bot/server.py` (lines 1–400)

### What you'll learn
- What FastAPI is (building web servers in Python)
- Async programming (why the bot can do many things at once)
- Global state (how the bot remembers things between ticks)
- Background jobs (the auto-scan loop)
- What REST API endpoints are

### FastAPI: Our Web Server

The bot is built as a **web server**. That means:
1. It runs as a program that listens for HTTP requests
2. Other programs (including the frontend dashboard) can ask it questions
3. You can control it via curl commands from the terminal

```bash
# Check if the bot is running:
curl http://localhost:8000/api/status

# Enable live trading:
curl -X POST http://localhost:8000/api/autotrade -d '{"enabled": true}'

# Check current positions:
curl http://localhost:8000/api/portfolio
```

**FastAPI** is the Python library we use to build this server. It's fast, modern, and automatically generates documentation.

### Async Programming

Normal (synchronous) code does one thing at a time:
```python
fetch_prices()    # takes 100ms
fetch_markets()   # takes 200ms
# Total: 300ms
```

Async code does multiple things simultaneously:
```python
await asyncio.gather(
    fetch_prices(),    # starts immediately
    fetch_markets(),   # also starts immediately
)
# Total: 200ms (limited by the slower task)
```

The bot uses `async/await` throughout so it can fetch prices, scan markets, and process signals simultaneously — reducing latency from 300ms to ~75ms per cycle.

The scan loop runs every **0.15 seconds**, but portfolio data (positions + balance) is fetched from REST at most once per second and cached. A persistent **WebSocket connection** streams real-time bid/ask prices from Kalshi and overlays them onto each market before the edge calculation — so the bot always uses the freshest available price data without hammering the REST API.

### Global State

Python global variables hold the bot's "memory" between scan cycles:
```python
# At the top of server.py
_kalshi_ws_prices: dict   # WS real-time bid/ask per ticker (updated sub-second)
_ws_positions: dict       # WS-confirmed open positions (updated on every fill)
_ws_fills_log: list       # recent fill events from WS (last 50)
_cached_positions: list   # REST positions cache (refreshed at most 1/s)
_cached_balance: int      # REST balance cache (refreshed at most 1/s)
```

WS state is updated as messages arrive (sub-second). REST cache is refreshed at most once per second — the 0.15s scan loop uses the cached values between REST refreshes.

### Checkpoint ✓

1. What is a web server and why is the bot built as one?
2. What is async programming and why is it faster?
3. Why do we use global variables for the bot's state?
4. What does `GET /api/status` return?

---

## CHAPTER 12 — The SMART Signal: Computing Edge

### File to open: `bot/server.py` (search for `_kx_yes_ce` and `_kx_no_ce`)

### What you'll learn
- The full SMART signal computation
- How we estimate fair probability
- Edge calculation (YES side and NO side)
- Why we might bet NO on a market asking "will price go up?"
- The t-distribution and why we use it instead of normal distribution

### How We Estimate Fair Probability

For a "Will BTC go UP?" market, the live bot uses the **W90 signal**:

```
Step 1: Get current BTC price from CoinGecko
Step 2: Calculate recent price change (% move in last few minutes)
Step 3: Convert to p_price — a probability using the t-distribution
        (positive momentum → p_price > 0.5 → lean YES)
Step 4: Get market_prob — the current market mid-price as a probability
Step 5: Blend in logit space:
        fair_prob = sigmoid(0.90 × logit(p_price) + 0.10 × logit(market_prob))

Example:
  p_price:      0.55   (price signal says 55% chance YES)
  market_prob:  0.48   (market mid-price implies 48%)

  logit(0.55) = 0.201
  logit(0.48) = -0.080
  blend = 0.90 × 0.201 + 0.10 × (-0.080) = 0.173
  fair_prob = sigmoid(0.173) ≈ 0.543
```

The 90/10 weight (W90) strongly favors the price signal over the market consensus. The blend is done in **logit space** (log-odds) — not simple averaging — because probabilities close to 0 or 1 need to be stretched before averaging to avoid compressing the tails.

### Direction Gate and Edge Calculation

The live bot uses a **directional gate** — side is determined by fair_prob alone:
- `fair_prob > 0.5` → bet YES
- `fair_prob ≤ 0.5` → bet NO

Then edge is calculated as the profit margin at the current ask:
```python
# YES bet: we pay yes_ask, expect to collect 100¢ with probability fair_prob
yes_edge_cents = round(fair_prob * 100) - yes_ask

# NO bet: we pay no_ask, expect to collect 100¢ with probability (1 - fair_prob)
no_edge_cents = round((1 - fair_prob) * 100) - no_ask

# Minimum edge floor: 10¢ (trades with less edge are skipped)
```

Example:
```
fair_prob = 0.58 → bet YES
yes_ask = 44¢
yes_edge = round(0.58 × 100) - 44 = 58 - 44 = +14¢  ✓ (above 10¢ floor)
```

The edge floor prevents trading in near-certain markets (e.g., 96¢ ask) where the spread eats all profit.

### Why t-Distribution?

Crypto prices have **fat tails** — extreme moves happen more often than a normal distribution predicts. The t-distribution (used in `scipy.stats.t.cdf`) captures this by having heavier tails than a bell curve.

For a "will price go above $X?" market with volatility σ and time T:
```
z = log(current_price / strike) / (σ × √T)
probability = t.cdf(z, degrees_of_freedom=5)
```

Lower degrees of freedom = fatter tails = more probability of extreme moves.

### Order Execution: IOC Orders

When a trade signal fires, the bot places a **limit order with `time_in_force: immediate_or_cancel` (IOC)**:

```
IOC order = "fill what you can at this price right now, cancel the rest instantly"
```

The price ceiling is set to **fair value** — for a YES bet, the ceiling is `round(fair_prob × 100)` cents. This means the bot will pay anything from the current ask up to fair value. Any fill at or below fair value is profitable; the IOC cancels anything unfilled at the ceiling instantly.

This replicates shadow bot behavior (which assumes instant fills) as closely as possible on real markets.

---

## CHAPTER 13 — Shadow Bots: A/B Testing Strategies

### File to open: `bot/server.py` (search for `_shadow_bots` and `_write_shadow_trade_db`)

### What you'll learn
- What shadow bots are and why they exist
- How A/B testing works
- The confirmed optimal strategy (no DCA, no hedge, no stop-loss)
- The current wave structure (Wave 6 + Wave 8)
- How we pick a winner

### What Are Shadow Bots?

A **shadow bot** is a virtual trading bot that:
- Tracks positions in memory (no real orders placed)
- Uses real market data and prices
- Keeps accurate simulated P&L
- Competes against 89 other shadow bots simultaneously

We currently run **90 shadow bots in parallel** across two waves. The best bot by PnL and Sharpe ratio — once it has ≥100 settled trades, ≥60% WR, and beats ≥5 bots — becomes the candidate for manual deployment as the live strategy.

### The Current Wave Structure

**Wave 6 (49 bots) — the tournament baseline:**
All Wave 6 bots use W90 (logit blend) and directional logic (fair_prob > 0.5 → side).

- **R (control):** W90_E4_K30 — baseline with no extra gates
- **H1 (9 bots):** Win-probability gate sweep (WP51–WP59)
- **H2 (5 bots):** Kelly sweep at WP55+E4 (K15–K50)
- **H3 (6 bots):** Kelly sweep at WP57+E4, plus **g10** (W90_E10_K30 = live bot config)
- **H4 (5 bots):** Edge floor sweep at WP55+K30 (E2–E8)
- **H5 (4 bots):** Edge floor sweep at WP57+K30 (E3–E8)
- **H6 (4 bots):** Best WP×Edge combos with varying Kelly
- **H7 (4 bots):** +Consensus gate (require market_prob ≥ threshold)
- **H8 (3 bots):** +Timing gate (require seconds remaining > threshold)
- **H9 (4 bots):** High edge + WP combos (E8–E10, K25)
- **H10 (3 bots):** +Spread filter (require bid/ask spread ≤ 1 or 2 cents)
- **H11 (1 bot):** xco — W90_E12_K30 (pure high-edge, no WP gate)

**Wave 8 (41 bots) — finer sweeps:**
Extends Wave 6 findings with narrower parameter grids: edge sweep E7–E15, Kelly sweep K10–K50, dead zone sweep, WP gate sweep, signal weight sweep SW70–SW100.

**Key Wave 6 finding:** g10 (W90_E10_K30) leads in total PnL (~$32M vs xco's ~$12M). Lower edge floor = more trades = more compounding. All top bots use directional logic.

### Confirmed Optimal Strategy: Single Entry, No Adjustments

Backtesting and shadow data confirm:
- **DCA (doubling down):** reduces PnL 30–55%
- **Hedging (buying the other side):** reduces PnL 25–40%
- **Stop-loss / take-profit:** <0.1% effect — not worth implementing

The live bot places **one entry per market per candle** and holds to settlement. No exit until the market resolves.

### Checkpoint ✓

1. What is a shadow bot and why don't we just trade with real money to test?
2. What is A/B testing?
3. What does "W90_E10_K30" mean? (W=signal weight, E=edge floor cents, K=Kelly base %)
4. Why does the confirmed strategy use single entry with no stop-loss or DCA?

---

## CHAPTER 14 — Deploying the Best Bot

### What you'll learn
- How we pick a winner from the shadow tournament
- Deployment criteria (what qualifies)
- Why deployment is manual, not automatic

### Deployment Is Manual

Shadow tournament data is **collected only** — no script automatically deploys winners. Each new wave is a controlled experiment. Deploying a winner requires human review because:
- A bot that wins in one market regime may fail in another
- Shadow bots assume instant fills; live execution has slippage
- We need to verify the win reflects genuine edge, not lucky variance

### Deployment Criteria (Hard Rules)

A bot must meet ALL of these before being manually deployed:

```
1. At least 100 settled trades  → enough data to trust the result
2. Win rate ≥ 60%               → consistently winning more than losing
3. Beats ≥ 5 other bots by PPT → not just winning by luck on easy markets
```

**Profit Per Trade (PPT)** is the ranking metric:
```python
PPT = total_pnl_cents / number_of_settled_trades

Example:
  H3-g10: $32M PnL over ~360 trades = $88,900 PPT
  H11-xco: $12M PnL over ~280 trades = $42,500 PPT  ← g10 wins
```

### What Happens at Manual Deployment

```bash
# 1. Patch the live bot config with winner's parameters
curl -X PATCH http://localhost:8000/api/config \
  -H "Content-Type: application/json" \
  -d '{"live_min_edge_cents": 10, "live_kelly_base": 0.15}'

# 2. Restart to apply cleanly
./botctl stop && ./botctl start

# 3. Enable autotrade (never auto-enables on restart)
./botctl autotrade on
```

### Checkpoint ✓

1. Why is deployment manual rather than automatic?
2. What 3 criteria must a bot meet before deployment?
3. Why do we need 100+ trades minimum instead of just 10?
4. What is PPT and why is it better than raw PnL for ranking bots?

---

## CHAPTER 15 — Supporting Systems

### Files: `bot/database.py`, `bot/notifier.py`, `bot/arbitrage.py`

### What you'll learn
- Database persistence (why we save data)
- Notification systems (Slack/Discord alerts)
- Arbitrage detection

### Database (`bot/database.py`)

**SQLite** is a local file-based database built into Python. No cloud service, no API keys.
Data lives at `data/bot.db` — a single file on your machine.

We use it to save:
- Every trade (entry, exit, P&L)
- Model training data (features + outcomes)
- Performance history, scan logs, paper trading state

Why save to a database instead of memory?
- If the server crashes, memory is lost — database survives
- You can query historical data later: `sqlite3 data/bot.db "SELECT * FROM trades"`
- All history persists across restarts

To inspect the database directly:
```bash
sqlite3 data/bot.db
> SELECT COUNT(*), AVG(pnl_cents), SUM(won) FROM trades WHERE mode='live';
> .quit
```

### Notifications (`bot/notifier.py`)

The bot sends messages to Slack or Discord when:
- A trade is placed
- A position is closed (with P&L)
- A risk limit is hit
- The model is retrained

This lets you monitor the bot without staring at logs.

### Arbitrage (`bot/arbitrage.py`)

Sometimes Kalshi and DraftKings (another prediction market) price the same event differently. If Kalshi says 40% and DraftKings says 60%, you can buy YES on Kalshi and NO on DraftKings for a risk-free profit.

This is **arbitrage** — exploiting price differences between markets. Our bot detects these opportunities and logs them.

### Checkpoint ✓

1. Why do we save data to a database instead of just keeping it in memory?
2. What events trigger a notification?
3. What is arbitrage and why is it risk-free?

---

## CHAPTER 16 — The Full Picture

### Putting It All Together

Here's the complete flow of one trading cycle (runs every 0.15 seconds):

```
┌──────────────────────────────────────────────────────────────┐
│  CONTINUOUSLY: _kalshi_ws_loop() (background task)           │
│     └─ Maintains persistent WebSocket to Kalshi             │
│     └─ Subscribed to: ticker (7 markets) + fill +           │
│        market_positions                                      │
│     └─ On ticker msg: updates _kalshi_ws_prices[ticker]      │
│     └─ On fill msg: logs to _ws_fills_log                    │
│     └─ On market_positions msg: updates _ws_positions        │
│     └─ Reconnects after 14 min (new candles + fresh auth)   │
│                                                              │
│  EVERY 0.15s: _auto_scan_job() runs                         │
│                                                              │
│  1. Portfolio data (cached at 1s — not fetched every tick)  │
│     └─ If >1s since last fetch: REST call for positions      │
│        + balance; store in _cached_positions/_cached_balance │
│     └─ Else: use cache (typical path at 0.15s scan rate)    │
│                                                              │
│  2. Get live prices from CoinGecko (cached)                 │
│     └─ Latest BTC, ETH, XRP, SOL, DOGE, BNB, HYPE prices   │
│     └─ Recent price change % (momentum → p_price signal)    │
│                                                              │
│  3. Fetch 15M crypto markets from Kalshi REST               │
│     └─ Market list, bid/ask, volume, open interest          │
│     └─ Overlay WS prices: for each ticker, if               │
│        _kalshi_ws_prices has fresh data, update bid/ask     │
│                                                              │
│  4. For each market, compute fair_prob (W90 formula)        │
│     └─ p_price:  t-distribution on price momentum           │
│     └─ market_prob: Kalshi mid-price                        │
│     └─ fair_prob = sigmoid(0.90×logit(p_price)             │
│                          + 0.10×logit(market_prob))         │
│                                                              │
│  5. Direction gate + edge check                             │
│     └─ side = YES if fair_prob > 0.5, NO otherwise          │
│     └─ edge = fair_cents - ask_price                        │
│     └─ Skip if edge < 10¢ (minimum floor)                   │
│                                                              │
│  6. For live bot: place IOC order if signal passes          │
│     └─ price ceiling = round(fair_prob × 100) for YES       │
│     └─ time_in_force = "immediate_or_cancel"                │
│     └─ Fills anything at ask up to fair value instantly     │
│     └─ Logs fill result to order_fills table                │
│                                                              │
│  7. For each of 90 shadow bots                              │
│     └─ Check bot-specific gates (WP, edge floor, spread…)  │
│     └─ If pass: compute Kelly size and record virtual entry │
│                                                              │
│  8. Check shadow positions for settlement                    │
│     └─ Market closed + resolved? Credit or debit balance    │
│     └─ Write settled trade to shadow_snapshots DB table     │
└──────────────────────────────────────────────────────────────┘
```

### The Three-Layer Architecture

```
Layer 1: Data Collection
  bot/kalshi_client.py  → market prices, order book
  bot/crypto_feed.py    → live prices, funding rates

Layer 2: Signal Processing  
  bot/rf_model.py       → ML probability estimate
  bot/server.py         → SMART signal blending
  bot/risk_manager.py   → pre-trade safety gates

Layer 3: Execution & Tracking
  bot/server.py         → shadow bots, live trading
  bot/performance.py    → P&L, Sharpe, win rate
  bot/database.py       → persistence
  bot/notifier.py       → alerts
```

---

## Final Exam: Can You Rebuild This?

If you've read all the code and followed this guide, you should be able to answer:

1. **Data flow**: Trace the path from "Kalshi publishes a new BTC 15M market" to "live bot places an IOC order". Name every function/file involved and when WS vs REST data is used.

2. **Signal formula**: p_price=0.60, market_prob=0.52. Compute fair_prob using W90. (Hint: logit(0.60)=0.405, logit(0.52)=0.080; sigmoid(x)=1/(1+e^-x))

3. **Direction and edge**: fair_prob=0.61, yes_ask=48¢, no_ask=55¢. (a) Which side do we bet? (b) What is the edge? (c) What is the IOC price ceiling?

4. **Kelly sizing**: fair_prob=0.61, yes_ask=48¢, balance=$300, kelly_base=0.30, hard cap=10% of balance. What is the full Kelly fraction? What is the K30 fraction? What is the capped bet size in dollars?

5. **IOC mechanics**: You place a YES IOC order with ceiling=61¢ when yes_ask=48¢. The order book has 20 contracts at 48¢ and 5 more at 55¢. The market then jumps to 65¢. Describe what fills and what gets cancelled.

6. **Shadow tournament**: H3-g10 has 360 trades, 60% WR, $32M PnL, beats all 48 other Wave 6 bots. Is it ready for manual deployment? Walk through the 3 criteria.

7. **Architecture**: You want to add a new signal source (e.g., on-chain flow data). Which files would you need to modify and why?

---

## Glossary

| Term | Definition |
|------|-----------|
| **API** | Application Programming Interface — rules for how programs talk to each other |
| **Arbitrage** | Buying and selling the same thing in different markets to profit from price differences |
| **Ask price** | Lowest price a seller will accept |
| **Backtesting** | Testing a strategy on historical data |
| **Bid price** | Highest price a buyer will pay |
| **Binary market** | A market with exactly two outcomes (YES/NO) |
| **Circuit breaker** | Stops API calls when the API is failing repeatedly |
| **Edge** | Our estimate of true probability minus what the market charges |
| **Expected value (EV)** | Average outcome over many trials: p × win - (1-p) × loss |
| **Feature** | A numerical input to a machine learning model |
| **Funding rate** | Fee paid between leveraged traders on futures exchanges |
| **Kelly Criterion** | Formula for optimal bet sizing given probability and odds |
| **Overfitting** | When a model memorizes training data instead of learning patterns |
| **Paper trading** | Simulated trading with fake money on live data |
| **P&L** | Profit and Loss |
| **PPT** | Profit Per Trade — total P&L divided by number of trades |
| **REST** | Standard web API style using GET/POST/DELETE verbs |
| **BRTI** | Bitcoin Real-Time Index — CF Benchmarks price oracle; what Kalshi settles against |
| **Directional gate** | fair_prob > 0.5 → YES, ≤ 0.5 → NO; no dead zone in current live config |
| **Edge floor** | Minimum edge in cents required before placing a trade (currently 10¢) |
| **IOC order** | Immediate-Or-Cancel: fills what it can at the given price, cancels the rest instantly |
| **Fair-value ceiling** | IOC price limit = round(fair_prob × 100); bot pays up to this but no more |
| **g10** | Shadow bot H3-g10 (W90_E10_K30) — current live bot config used as shadow control |
| **K30** | Kelly base fraction of 30% (further scaled by vol, spread, correlation) |
| **Logit** | log(p / (1-p)) — transforms a probability to log-odds space for blending |
| **Portfolio floor** | Emergency stop triggers when cash + open position value drops below $200 |
| **Shadow bot** | Virtual bot tracking simulated positions without real orders |
| **SQLite** | Local file-based database (data/bot.db) — no cloud service needed |
| **Student-t** | Probability distribution with fat tails — better than normal for crypto price moves |
| **W90** | Signal weight: 90% price signal + 10% market consensus, blended in logit space |
| **Wave 6** | Current 49-bot shadow tournament (H1–H11 groups; control = g10) |
| **Wave 8** | Current 41-bot shadow tournament (finer parameter sweeps on top of Wave 6 findings) |
| **WebSocket (WS)** | Persistent real-time connection to Kalshi; streams ticker, fill, and market_positions |
| **WS price overlay** | Before each edge calculation, WS bid/ask replaces stale REST bid/ask if available |
| **Sharpe ratio** | Risk-adjusted return metric: average return divided by return volatility |
| **Singleton** | A pattern ensuring only one instance of an object exists |
| **Win rate** | Percentage of trades that profit |

---

*Built with FastAPI, Kalshi API v2, BRTI constituent exchanges (Bitstamp/Coinbase/Gemini/Kraken), Binance Futures, scikit-learn Random Forest + Gradient Boosting, Kelly Criterion sizing, local SQLite persistence, and 42-bot Wave 4 shadow A/B experiment (V1/V2/V3 formulas × EV/PM/BL strategy variants).*
