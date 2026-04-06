"""
Comprehensive strategy optimization for Kalshi crypto prediction markets.

═══════════════════════════════════════════════════════════════════════════════
  CHAPTER 2: STRATEGY-SPECIFIC BACKTESTING
═══════════════════════════════════════════════════════════════════════════════

While backtester.py tests the ML model's raw prediction accuracy, THIS file
tests specific trading strategy mechanics: HOW to enter, HOW to manage the
trade while it's open, and HOW to exit.

The two files answer different questions:
  backtester.py:        "Can the model accurately predict market outcomes?"
  strategy_backtest.py: "Given accurate predictions, what's the best way to trade?"

──────────────────────────────────────────────────────────────────────────────
CANDLE-BASED BACKTESTING

Instead of Kalshi settlement data, this file uses raw price "candles" from
Binance (a cryptocurrency exchange). A candle summarizes price action within
a time window:
  - open:  price at the START of the window
  - close: price at the END of the window
  - high:  highest price during the window
  - low:   lowest price during the window

Kalshi's 15-minute binary markets ask: "Will BTC be above $X at 2:15pm?"
We can simulate this with 5-minute candles:
  Candle 1 (2:00pm close): entry signal — is BTC above/below the strike?
  Candle 2 (2:05pm close): intermediate — has the situation changed?
  Candle 3 (2:10pm close): intermediate — stop/take-profit check
  Candle 4 (2:15pm close): SETTLEMENT — did it end above or below?

Using 5-minute sub-candles is crucial for realistic stop-loss and DCA
simulation. With only settlement data, you'd never know if the price briefly
dipped below your stop before recovering — the candle data reveals that.

──────────────────────────────────────────────────────────────────────────────
WHAT IS DCA (DOLLAR-COST AVERAGING)?

Normally you enter a trade all at once. DCA means splitting your investment
into multiple smaller entries over time.

Example without DCA:
  2:00pm: Buy 10 contracts at 55¢ ($5.50 total cost)

Example with 2-tranche DCA:
  2:00pm: Buy 6 contracts at 55¢ ($3.30 first tranche)
  2:05pm: Buy 4 contracts at 52¢ ($2.08 second tranche — cheaper!)
  Average entry: (3.30 + 2.08) / 10 = 53.8¢ per contract (better than 55¢)

DCA can improve your average entry price if the market dips after your first
entry. The 'dca_only_better' flag enforces that you only add the second tranche
if the new price is actually better (cheaper) than your first entry.

──────────────────────────────────────────────────────────────────────────────
WHAT IS HEDGING?

A hedge is a second position that profits if your main position loses. It's
like buying insurance on your trade.

Example with 15% hedge on a YES trade:
  Main position: Buy 10 YES contracts at 55¢
  Hedge:         Buy 1.5 NO contracts at 45¢ (approximately)

  If YES wins: Main profit = 45¢ × 10 = $4.50, Hedge loss = 45¢ × 1.5 = -$0.68
               Net = $3.82 (slightly less, but you had insurance)

  If NO wins:  Main loss = 55¢ × 10 = -$5.50, Hedge profit = 55¢ × 1.5 = $0.83
               Net = -$4.67 (less catastrophic than without hedge)

Hedging reduces both your upside AND your downside. It's useful when:
  - You're confident in the direction but worried about tail risks.
  - The market is illiquid and a big move could compound losses quickly.

The hedge_ratio sweep tests: is the reduced variance worth the reduced profit?

──────────────────────────────────────────────────────────────────────────────
WHAT IS A STOP-LOSS?

A stop-loss is an automatic exit rule: "If this position loses more than X%,
get out — don't wait to see if it recovers."

Example with stop_loss_pct=0.75:
  Entry: 10 contracts at 55¢ (cost: $5.50)
  Stop trigger: price falls to 55¢ × (1 - 0.75) = 55¢ × 0.25 = 13.75¢
  If the intermediate candle shows a market price at or below 13.75¢ → EXIT.
  Loss = (13.75¢ - 55¢) × 10 = -$4.13 instead of the full -$5.50 at 0¢.

Without stop-losses, every losing trade costs the full entry price.
With stop-losses, you cap the damage but may get stopped out of trades that
would have recovered.

The sweep tests 0%, 25%, 50%, 75%, 90% stop thresholds to find the optimal level.

──────────────────────────────────────────────────────────────────────────────
WHAT IS TAKE-PROFIT?

The mirror image of a stop-loss: "If this position gains more than X%, exit
now and lock in the profit instead of waiting for settlement."

Example with take_profit_pct=0.50:
  Entry at 55¢. Maximum possible gain = 100¢ - 55¢ = 45¢ per contract.
  Take-profit trigger: gain reaches 50% of 45¢ = 22.5¢ → exit at 77.5¢.
  We lock in 22.5¢ profit instead of risking holding to settlement at 100¢ or 0¢.

Take-profits are especially useful in prediction markets because prices can
spike toward 100¢ and then reverse. Locking in early can outperform waiting.

──────────────────────────────────────────────────────────────────────────────
THE PROBABILITY MODEL IN THIS FILE

This file uses a mathematical model (not ML) to estimate fair probability:
  1. Student-t distribution for directional markets ("will BTC be above $X?")
     The t-distribution has "fat tails" — it assigns more probability to large
     price moves than a normal bell curve. Cryptocurrencies are notorious for
     sudden large moves, so this is more realistic than a simple bell curve.

  2. Combined fair probability merges multiple signals (price, funding rate,
     order book imbalance, time-of-day bias) using a weighted logit formula.
     Each signal is converted to log-odds, weighted, then converted back to
     a probability using the sigmoid function.

This is separate from the RF/GB machine learning model in backtester.py.
Here, the probability comes from finance math rather than pattern recognition.

──────────────────────────────────────────────────────────────────────────────
WHAT IS THE KELLY CRITERION?

Kelly Criterion answers: "How much of my bankroll should I bet?"

Full Kelly formula: f = (edge × bankroll) / net_profit_per_unit
  where: edge = win_probability × net_profit - loss_probability × cost

Betting full Kelly maximizes long-term bankroll growth mathematically, but
causes large swings (drawdowns). Quarter-Kelly (0.25 × full Kelly) is a
common conservative choice that grows slower but survives bad runs better.

The tiered approach here:
  win_prob ≥ 80%: bet 50% of full Kelly (high confidence → larger fraction)
  win_prob ≥ 70%: bet 35% of full Kelly
  win_prob < 70%: bet 20% of full Kelly (uncertain → smaller fraction)

──────────────────────────────────────────────────────────────────────────────
THE PARAMETER SWEEP FRAMEWORK

run_sweep() tests every combination of settings defined in its sweep grids.
For each combination, it calls run_strategy_backtest() with a fresh balance
and records: trades/day, win rate, total P&L, max drawdown, profit factor,
Sharpe ratio, stop exits, take-profit exits, and DCA additions.

The results are sorted by total P&L (not Sharpe) to find the highest absolute
profit configuration, with the top-5 shown at the end.

This is computationally intensive — the "full" sweep runs 216 combinations,
each simulating 90 days of trading across multiple coins and market types.
Expect it to take several minutes per run.

──────────────────────────────────────────────────────────────────────────────
TESTS ALL COMBINATIONS OF:
  EXIT STRATEGIES:
    stop_loss_pct  — exit early if position loses X% of entry (0=disabled)
    take_profit_pct — exit early if position gains X% of max profit (0=disabled)

  DCA (Dollar-Cost Averaging):
    dca_tranches   — 1/2/3 entry tranches within the 15m window (5m sub-candles)
    dca_sizes      — size schedule [0.6,0.4] for 2T, [0.5,0.3,0.2] for 3T
    dca_only_better — only DCA if new entry price is better than first entry

  HEDGING:
    hedge_ratio    — fraction of main position bought as opposite side (0=none)

  SIZING:
    All bets Kelly-sized proportional to current live balance.
    Per-coin min_prob thresholds (stricter for low-WR coins).
    Per-coin Kalshi liquidity caps (realistic fill limits).

  MARKET TYPES: 15m directional, 1h directional, 1h_range (combined)

Data: 5m candles for 15m markets (real intermediate prices for stop/TP/DCA).
      15m candles for 1h markets.

Usage:
    python -m bot.strategy_backtest                  # best single run
    python -m bot.strategy_backtest --sweep exit     # exit strategy sweep
    python -m bot.strategy_backtest --sweep dca      # DCA sweep
    python -m bot.strategy_backtest --sweep hedge    # hedge sweep
    python -m bot.strategy_backtest --sweep full     # all 216 combinations
    python -m bot.strategy_backtest --sweep combined # exit+dca+hedge combined
"""

import math
import time
import statistics
from datetime import datetime, timezone
from typing import Any

import requests
from scipy.stats import t as student_t

# ─── Re-use core signal logic from crypto_backtest ───────────────────────────

def _logit(p: float) -> float:
    """Convert a probability (0–1) into log-odds (−∞ to +∞).

    The "logit" function is a key mathematical transformation in probability.
    It converts a probability into a scale where 0.5 maps to 0, probabilities
    above 0.5 give positive values, and probabilities below 0.5 give negative
    values.

    Formula: logit(p) = log(p / (1 - p))
    Examples:
      logit(0.5)  = log(1/1)   = 0       (completely uncertain)
      logit(0.9)  = log(9/1)   = 2.20    (strongly YES)
      logit(0.1)  = log(1/9)   = -2.20   (strongly NO)
      logit(0.99) = log(99/1)  = 4.60    (near-certain YES)

    WHY USE LOG-ODDS?
    Probabilities can't be simply averaged or added — adding 50% + 50% = 100%
    (which means certainty) doesn't make sense. Log-odds CAN be averaged and
    added, which is why they're used here to combine multiple signals.

    The clamp to [0.001, 0.999] prevents log(0) = -infinity or log(inf).
    """
    p = max(0.001, min(0.999, p))
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    """Convert log-odds back to probability (the inverse of logit).

    While logit() converts probability → log-odds,
    sigmoid() converts log-odds → probability.

    Formula: sigmoid(x) = 1 / (1 + e^(-x))
    Examples:
      sigmoid(0)     = 0.50  (neutral)
      sigmoid(2.20)  = 0.90  (strongly YES)
      sigmoid(-2.20) = 0.10  (strongly NO)
      sigmoid(4.60)  = 0.99  (near-certain YES)

    Together, logit and sigmoid let you:
      1. Convert each signal probability into log-odds.
      2. Take a weighted average of the log-odds (signals combined in "logit space").
      3. Convert the result back to a probability.

    This avoids the nonsensical arithmetic that would result from averaging
    probabilities directly.
    """
    return 1.0 / (1.0 + math.exp(-x))

def _combined_fair(p_price, p_funding, p_imbalance, p_consensus, p_tod) -> float:
    """Combine five independent probability signals into one fair probability.

    Each input (p_price, p_funding, etc.) is a probability between 0 and 1
    representing one "opinion" about whether the market will resolve YES.

    The combination works in three steps:
      1. Convert each probability to log-odds using _logit().
      2. Take a weighted sum of those log-odds (weights below must sum to 1).
      3. Convert the result back to a probability using _sigmoid().

    The weights reflect how reliable each signal is:
      0.62 × p_price:     Price vs strike (dominant — crypto price is the main factor)
      0.20 × p_funding:   Funding rate (contrarian signal: high funding → overbought)
      0.08 × p_imbalance: Order book imbalance (who's buying vs selling right now)
      0.05 × p_consensus: Market consensus (what other traders think)
      0.05 × p_tod:       Time-of-day bias (markets behave differently at different hours)

    The result is clamped to [0.01, 0.99] to avoid probabilities of exactly 0 or 1,
    which would cause mathematical problems downstream (e.g., log(0)).
    """
    raw = _sigmoid(
        0.62 * _logit(p_price) + 0.20 * _logit(p_funding) +
        0.08 * _logit(p_imbalance) + 0.05 * _logit(p_consensus) +
        0.05 * _logit(p_tod)
    )
    return max(0.01, min(0.99, raw))

# _DF: Student-t degrees of freedom per coin.
# The t-distribution is like a "fattened" bell curve — it assigns more
# probability to extreme outcomes (very large price moves) than a normal
# distribution does. This is realistic for crypto: BTC can move 5% in a day,
# something that would be astronomically unlikely under a normal distribution.
#
# Higher degrees of freedom → closer to a normal bell curve (thinner tails)
# Lower degrees of freedom → fatter tails → more probability assigned to extremes
#
# Bitcoin has df=4 (relatively stable for crypto), Dogecoin has df=2.5 (very
# volatile, much fatter tails). This calibration was tuned to historical data.
_DF = {
    "bitcoin": 4, "ethereum": 3.5, "ripple": 3,
    "solana": 3.0, "dogecoin": 2.5, "bnb": 4.0,
    "hyperliquid": 2.5, "litecoin": 4.0, "cardano": 3.0, "avalanche": 3.0,
}

# _ANN_VOL: Annualized volatility per coin (as a fraction, not percent).
# "Annualized volatility" is a standard finance metric measuring how much a
# price typically swings over a year.
# - 0.75 for Bitcoin means BTC typically moves ~75% of its value in a year.
# - 2.00 for Hyperliquid means HYPE could move ~200% in a year (very volatile).
#
# This is used to compute how likely the price is to cross the strike by
# settlement time. More volatile coins need a larger price advantage to achieve
# the same probability — they could swing back easily.
#
# Formula context: the t-distribution CDF is evaluated at:
#   z = log(current_price / strike) / (annualized_vol × sqrt(time_remaining_in_years))
# A coin far above the strike + low vol = high probability of staying above.
_ANN_VOL = {
    "bitcoin": 0.75, "ethereum": 0.95, "ripple": 1.15,
    "solana": 1.25, "dogecoin": 1.50, "bnb": 0.85,
    "hyperliquid": 2.00, "litecoin": 0.90, "cardano": 1.20, "avalanche": 1.30,
}

def _fair_prob_dt(spot, strike, secs_remaining, coin, vol_scale=1.0):
    """Estimate the probability that 'spot' price will be ABOVE 'strike' at settlement.

    This uses the Student-t distribution (a "fatter-tailed" version of the normal
    distribution) to model cryptocurrency price movements.

    The key formula:
      vol_h = annualized_volatility × sqrt(days_remaining / 365)

    This is the "expected volatility over the remaining time." Volatility scales
    with the SQUARE ROOT of time — a result from financial theory (Brownian motion).
    For example: if annual vol is 75% (0.75), then over 15 minutes (= 1/96th of a day):
      vol_h = 0.75 × sqrt((1/96) / 365) ≈ 0.0040 = 0.4%

    The z-score: z = log(spot/strike) / vol_h
      How many "standard deviations" is the current price above the strike?
      - If spot = 50,000 and strike = 49,000: z > 0 → price is above strike → YES is more likely
      - If spot = 48,000 and strike = 49,000: z < 0 → price is below strike → NO is more likely

    student_t.cdf(z, df) converts z-score to probability using the t-distribution.
    """
    ann_vol = _ANN_VOL.get(coin, 1.0) * vol_scale
    days_left = max(secs_remaining / 86400, 1 / 1440)  # minimum of 1 minute to avoid div/0
    vol_h = ann_vol * (days_left / 365.0) ** 0.5        # sqrt(time) scaling — see explanation above
    if vol_h <= 0: return 0.5                            # no volatility → coin can't move → 50/50
    z = math.log(spot / strike) / vol_h
    return float(student_t.cdf(z, df=_DF.get(coin, 3)))

def _fair_prob_range(spot, strike, secs_remaining, coin, range_pct=0.01, vol_scale=1.0):
    """Estimate the probability that the price will stay WITHIN +/-range_pct of the strike.

    "Range" markets ask: "Will BTC stay within 1% of $50,000?"
    This is a different question from "Will BTC be above $50,000?"

    The probability is computed as:
      P(|price_at_settlement / strike - 1| <= range_pct)

    Using symmetry of the t-distribution:
      = 2 × CDF(z_bound) - 1
      where z_bound = log(1 + range_pct) / vol_h

    Intuitively: if the price needs to stay within a tiny range (0.5%) but
    volatility is high, it's unlikely. If the range is wide (2%) and volatility
    is low, it's much more likely.

    The 2× CDF - 1 formula comes from the fact that the distribution is symmetric:
    P(within range) = P(below upper bound) - P(below lower bound)
                    = CDF(+z) - CDF(-z)
                    = 2 × CDF(z) - 1
    """
    ann_vol = _ANN_VOL.get(coin, 1.0) * vol_scale
    days_left = max(secs_remaining / 86400, 1 / 1440)
    vol_h = ann_vol * (days_left / 365.0) ** 0.5
    if vol_h <= 0: return 0.5
    z_bound = math.log(1.0 + range_pct) / vol_h
    return max(0.01, min(0.99, 2.0 * float(student_t.cdf(z_bound, df=_DF.get(coin, 3))) - 1.0))

def _tod_bias(hour_utc):
    """Return a small probability adjustment based on the hour of day (UTC).

    Crypto markets aren't equally active all day. Historical patterns show:
      - 12:00–20:00 UTC (US market hours): Higher activity, bullish lean → +2% bias
      - 00:00–06:00 UTC (Asian overnight):  Lower activity, slight bearish lean → -2% bias
      - All other hours: No bias → 0%

    This bias is a small nudge (+/- 2%) applied to the combined fair probability.
    It's the weakest of the five signals (weight = 0.05 in _combined_fair) and is
    applied as a small shift to the base probability rather than a multiplier.

    Note: This is a simple heuristic, not a hard rule. Markets can and do move
    against the expected time-of-day pattern. The small weight reflects this
    uncertainty — we're saying "there's a slight tendency" not "this is certain."
    """
    if 12 <= hour_utc < 20: return 0.02
    elif 0 <= hour_utc < 6: return -0.02
    return 0.0

def _kelly_size(win_prob, price_cents, min_bet, max_bet, kelly_fraction, bankroll_cents):
    """Compute the Kelly-optimal bet size in cents.

    The Kelly Criterion gives the mathematically optimal fraction of your bankroll
    to bet to maximize long-term wealth growth.

    For a binary market (YES pays $1, NO pays $0):
      - You pay price_cents to enter.
      - If you win: receive $1 (profit = 100 - price_cents, net of fees).
      - If you lose: lose price_cents.

      Kelly formula: f* = (b × p - q) / b
        where: b = net_profit / price_cents   (the odds: how much you win vs risk)
               p = win_prob
               q = 1 - win_prob

      Actual bet = f* × kelly_fraction × bankroll_cents

    The kelly_fraction (typically 0.20–0.50) is a "fractional Kelly" that reduces
    the bet size below full Kelly. This is standard practice because:
      1. Full Kelly bets are mathematically correct but psychologically brutal
         (you might bet 80% of your bankroll at high confidence — huge swings).
      2. Our probability estimates are imperfect, so we're more conservative.
      3. Real-world constraints (fees, slippage, position limits) further erode edge.

    The result is clamped between min_bet and max_bet. If the calculated size is
    below min_bet, we skip the trade (not worth the transaction overhead). If it's
    above max_bet (or the liquidity cap for this coin), we cap it.
    """
    if price_cents <= 0 or price_cents >= 100: return 0  # invalid price → skip
    gross_profit = 100 - price_cents              # how much we WIN per contract (before fees)
    fee_cents = 0.01 * price_cents * (100 - price_cents) / 100  # maker limit orders ≈ 0% fee; 1% buffer
    net_profit = gross_profit - fee_cents         # how much we actually keep after fees
    if net_profit <= 0: return 0                  # fees eat the profit → never trade this
    loss = price_cents                             # how much we LOSE per contract if wrong
    # edge = expected value per contract
    # If win_prob=0.65, net_profit=43¢, loss=55¢:
    #   edge = 0.65 × 43 - 0.35 × 55 = 27.95 - 19.25 = 8.7¢ per contract
    # This is positive edge — we expect to profit 8.7¢ for every contract traded.
    edge = win_prob * net_profit - (1 - win_prob) * loss
    if edge <= 0: return 0   # no edge or negative edge → don't trade
    full_kelly = edge / net_profit   # optimal fraction of bankroll to bet
    fraction = full_kelly * kelly_fraction   # scale down to fractional Kelly
    bet = int(fraction * bankroll_cents)     # convert to actual dollar amount
    # No flat % cap — Kelly fraction + budget cap + directional cap bound exposure
    return max(0, min(max_bet, max(min_bet if bet >= min_bet else 0, bet)))


# ─── Constants ────────────────────────────────────────────────────────────────

START_BAL   = 24_764   # Starting balance in cents ($247.64 — the actual funded balance)
MKTEFF      = 0.80     # "Market efficiency factor": assumes markets are 80% efficient.
                       # The bot cannot bet at pure theoretical fair value; markets
                       # are somewhat efficient and the price partially reflects the
                       # same signals we're using. So we model the actual market price
                       # as: mkt_mid = 0.5 + 0.80 × (fair_prob - 0.5)
                       # This is conservative — it means we only capture 80% of the edge.
SLIPPAGE    = 1        # 1 cent of slippage per trade (see file-level docstring for explanation)
MIN_BET       = 200    # Minimum bet size in cents ($2.00). Below this, transaction overhead
                       # eats too much of the potential profit.
MAX_POSITIONS = 50     # Global upper ceiling — the per-coin cap below is the real binding limit
MAX_POS_PER_COIN = 2   # Max simultaneous positions per coin (prevents over-concentration in one coin)

# Kalshi liquidity reality: once balance exceeds this, bet sizing is capped at this
# level to prevent projecting fills that Kalshi can't actually supply.
# ~$5,000 is where $500 bets represent 10% of capital — roughly the realistic scale.
# If the backtest shows the balance compounding to $50,000, we don't pretend we'd
# get $5,000 fills — Kalshi may only have $500 of depth at the price we want.
MAX_COMPOUND_BALANCE = 500_000  # $5,000 in cents

# Per-coin minimum entry probability (stricter for low-WR coins).
# Not all coins trade equally well. BNB and Hyperliquid have historically been
# harder to profit on at lower confidence thresholds, so they require a higher
# minimum probability before we enter a trade.
# These thresholds were tuned by running the sweep and observing which coins
# were net losers at the default 60% threshold.
COIN_MIN_PROB = {
    "bitcoin": 0.60, "ethereum": 0.60, "ripple": 0.60,
    "solana": 0.60, "dogecoin": 0.60,
    "bnb": 0.75,          # was net loser at 60% threshold
    "hyperliquid": 0.72,  # borderline
    "litecoin": 0.82,
    "cardano": 0.60, "avalanche": 0.60,
}

# Per-coin Kalshi liquidity caps (realistic max fill per market in cents)
LIQUIDITY_CAPS = {
    "bitcoin":     50_000,  # $500 — raised from $300; Kalshi has millions daily volume, fills within 1c spread
    "ethereum":    40_000,  # $400
    "ripple":      30_000,  # $300
    "solana":      30_000,  # $300
    "dogecoin":    20_000,  # $200
    "bnb":         20_000,  # $200
    "hyperliquid": 15_000,  # $150
    "litecoin":    20_000,  # $200
    "cardano":     20_000,  # $200
    "avalanche":   15_000,  # $150
}

COINS_TARGET = {
    "bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "ripple": "XRPUSDT",
    "solana": "SOLUSDT", "dogecoin": "DOGEUSDT",
    "bnb": "BNBUSDT", "hyperliquid": "HYPEUSDT",
}

# Extended coin set — adds Kalshi-listed coins with decent liquidity
COINS_EXTENDED = {
    **COINS_TARGET,
    # "litecoin":  "LTCUSDT",  # REMOVED: net loser in every backtest config (81% WR, -$600/90d)
    "cardano":   "ADAUSDT",   # Kalshi: KXADA15M
    "avalanche": "AVAXUSDT",  # Kalshi: KXAVAX15M
}


# ─── Candle fetching ──────────────────────────────────────────────────────────

def _fetch_klines(symbol, interval="15m", limit=1000, end_time_ms=None):
    """Fetch raw OHLCV candle data from Binance.US for a given symbol and interval.

    "klines" is Binance's name for candlestick (OHLCV) data. OHLCV stands for:
      Open, High, Low, Close, Volume

    This is the raw HTTP request. The response is a list of arrays, where each
    array encodes one candle in a packed format:
      [open_time, open, high, low, close, volume, close_time, ...]

    We use Binance.US (the US-regulated version of Binance) as the price data
    source because it provides free historical data with high reliability and
    a 1,000-candle-per-request limit.

    Note: This is ONLY used for backtesting, not live trading. Live prices come
    from CoinGecko (see crypto_feed.py). We use Binance here for historical
    depth (90 days of 5-minute candles = 25,920 data points).
    """
    url = "https://api.binance.us/api/v3/klines"
    params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms:
        params["endTime"] = end_time_ms
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def fetch_candles(symbol, n_days=90, interval="15m"):
    """Fetch and return the last n_days of candles for a symbol at the given interval.

    This function handles pagination: Binance only returns 1,000 candles per request.
    For 90 days of 5-minute data, we need 90 × 288 = 25,920 candles, requiring
    26 separate API requests. The function works backwards in time:
      Request 1: most recent 1,000 candles
      Request 2: the 1,000 candles before that (using end_time_ms as a bookmark)
      ...continue until we have enough

    After fetching, duplicates are removed (can happen at page boundaries) and
    the list is sorted chronologically. Only the last `target` candles are returned
    to ensure consistent length across runs.

    The 0.05-second sleep between requests is out of courtesy to the API — making
    requests too fast can trigger rate-limiting (the API temporarily blocks you).
    """
    # cpd = "candles per day" — depends on interval
    # 5m candles → 288 per day (60min/5min × 24h), 15m → 96 per day, etc.
    cpd = {"1m":1440,"3m":480,"5m":288,"15m":96,"30m":48,"1h":24,"4h":6,"1d":1}.get(interval,96)
    target = n_days * cpd
    print(f"  Fetching {n_days}d {interval} {symbol}...", flush=True)
    candles, end_ms = [], None
    while len(candles) < target:
        batch = _fetch_klines(symbol, interval, 1000, end_ms)
        if not batch: break
        for row in batch:
            candles.append({
                "open_time_ms": int(row[0]), "open": float(row[1]),
                "high": float(row[2]),       "low":  float(row[3]),
                "close": float(row[4]),      "close_time_ms": int(row[6]),
            })
        end_ms = batch[0][0] - 1   # Move the bookmark to 1ms before the oldest candle in this batch
        if len(batch) < 1000: break  # Fewer than 1,000 returned → we've hit the beginning of available data
        time.sleep(0.05)             # Brief pause to avoid rate-limiting
    candles.sort(key=lambda x: x["open_time_ms"])
    # Remove duplicate candles (can occur at pagination boundaries)
    seen, unique = set(), []
    for c in candles:
        if c["open_time_ms"] not in seen:
            seen.add(c["open_time_ms"]); unique.append(c)
    print(f"    → {len(unique)} candles", flush=True)
    return unique[-target:]   # Return only the most recent `target` candles


# ─── Core strategy simulation ─────────────────────────────────────────────────

def run_strategy_backtest(
    n_days: int = 90,
    candles_5m: dict | None = None,   # {coin: [candle,...]} — 5-minute candles for 15m markets
    candles_15m: dict | None = None,  # {coin: [candle,...]} — 15-minute candles for 1h markets
    coins_override: dict | None = None,
    # Entry parameters
    min_fair_prob: float = 0.60,       # Minimum model probability to enter (0.60 = must be 60%+ confident)
    min_edge: float = 0.0,             # Minimum edge (fair_prob - entry_price) to enter; 0 = no minimum
    kelly_mult: float = 1.0,           # Multiplier for Kelly bet sizes; 0.5 = half Kelly
    max_positions: int = MAX_POSITIONS,
    max_pos_per_coin: int = MAX_POS_PER_COIN,
    _pos_cap: int | None = None,                    # sweep override for max_pos_per_coin
    _coin_min_prob_override: dict | None = None,    # sweep override for COIN_MIN_PROB
    # Risk controls
    budget_pct: float = 0.25,         # max fraction of balance deployable at once
                                       # e.g., 0.25 means never risk more than 25% of balance
    max_directional: int = 4,         # max same-direction (YES or NO) open positions
                                       # prevents betting everything one way (a "directional bet")
    daily_loss_pct: float = 0.20,     # pause trading if daily loss > X% of balance
                                       # circuit breaker: if today's losses exceed 20%, stop for the day
    # Exit strategy
    stop_loss_pct: float = 0.0,       # 0=off; 0.75=exit if 75% of entry lost
    take_profit_pct: float = 0.0,     # 0=off; 0.50=exit if 50% of max gain reached
    # DCA
    dca_tranches: int = 1,            # 1/2/3
    dca_sizes: list | None = None,    # fraction per tranche, must sum to ≤1
    dca_only_better: bool = True,     # only add tranche if price improved
    # Hedging
    hedge_ratio: float = 0.0,         # 0.0–0.30 opposite-side hedge fraction
    # Market types
    include_15m: bool = True,
    include_1h: bool = True,
    include_range: bool = True,
    range_pcts: list | None = None,
) -> dict:

    active_coins = coins_override or COINS_TARGET
    _range_pcts = range_pcts or [0.005, 0.010, 0.020]
    _eff_pos_cap = _pos_cap if _pos_cap is not None else max_pos_per_coin
    _eff_coin_min = _coin_min_prob_override if _coin_min_prob_override is not None else COIN_MIN_PROB

    # Default DCA size schedules
    if dca_sizes is None:
        if dca_tranches == 1:   dca_sizes = [1.0]
        elif dca_tranches == 2: dca_sizes = [0.60, 0.40]
        else:                   dca_sizes = [0.50, 0.30, 0.20]

    # ── Initialize simulation state ──────────────────────────────────────────
    # This is the "fake account" the simulation runs against.
    balance     = START_BAL        # Starts at the real funded balance and compounds from there
    open_pos: list[dict] = []      # List of currently open (unsettled) positions
    all_trades: list[dict] = []    # Complete history of all closed trades (for metrics)
    daily_loss  = 0   # tracks losses within current simulated day for circuit breaker
    per_coin   = {c: {"trades":0,"wins":0,"pnl":0,"stop_exits":0,"tp_exits":0,"dca_adds":0}
                  for c in active_coins}
    # Per-coin open position counter (prevents any coin from dominating slots)
    per_coin_open = {c: 0 for c in active_coins}
    _last_day = -1   # track day boundary for daily_loss reset

    # ── Build unified timeline ───────────────────────────────────────────────
    # The "timeline" is the backbone of the simulation: a sorted list of all the
    # moments in time when we might enter a trade, across all coins and market types.
    #
    # Think of it like a schedule:
    #   "At 9:00am, consider a BTC 15m trade."
    #   "At 9:00am, consider an ETH 15m trade."
    #   "At 9:00am, consider a BTC 1h trade."
    #   "At 9:15am, consider a BTC 15m trade."
    #   ...
    #
    # For 15m markets: we step through 5m candles with a stride of 3.
    #   Each "step" represents opening at one 5m candle and settling 15m later.
    # For 1h markets: we step through 15m candles with a stride of 4.
    #   Each "step" represents opening at one 15m candle and settling 60m later.
    timeline: list[tuple] = []  # (ts_ms, coin, idx, mtype, candle_interval)

    if include_15m and candles_5m:
        for coin, clist in candles_5m.items():
            if coin not in active_coins: continue
            for i in range(0, len(clist) - 3, 3):
                timeline.append((clist[i]["open_time_ms"], coin, i, "15m", "5m"))

    if (include_1h or include_range) and candles_15m:
        for coin, clist in candles_15m.items():
            if coin not in active_coins: continue
            for i in range(0, len(clist) - 4, 4):
                if include_1h:
                    timeline.append((clist[i]["open_time_ms"], coin, i, "1h", "15m"))
                if include_range:
                    for rp in _range_pcts:
                        timeline.append((clist[i]["open_time_ms"], coin, i, f"range_{rp}", "15m"))

    # Sort by (ts_ms, mtype_priority, coin) so at each timestamp:
    # all coins' 15m markets are processed before 1h, then range markets.
    # This prevents BTC (the highest-volume coin) from claiming all available
    # position slots before other coins get evaluated.
    # Priority: 15m first (shortest duration, most frequent), then 1h, then range.
    _mtype_rank = {"15m": 0, "1h": 1}
    def _tkey(x):
        rank = _mtype_rank.get(x[3], 2) if not x[3].startswith("range_") else 2
        return (x[0], rank, x[1])
    timeline.sort(key=_tkey)

    for ts_ms, coin, i, mtype, cinterval in timeline:
        clist = candles_5m[coin] if cinterval == "5m" else candles_15m[coin]
        c0 = clist[i]    # entry signal candle

        # Determine intermediate and settlement candles
        if cinterval == "5m":
            # 15m market: c0=signal, c1=intermediate(T+10m), c2=settlement(T+15m)
            if i + 2 >= len(clist): continue
            c1 = clist[i + 1]   # T+10m intermediate
            c2 = clist[i + 2]   # T+15m settlement
            gap_ms = c1["open_time_ms"] - c0["close_time_ms"]
            if gap_ms > 5_000: continue
            secs_entry = 600.0      # 10 min remaining at entry
            secs_interm = 300.0     # 5 min remaining at intermediate
        else:
            # 1h market: c0=signal, c1/c2/c3 = subsequent 15m candles, c3=settlement
            if i + 3 >= len(clist): continue
            c1 = clist[i + 1]   # T+15m intermediate
            c2 = clist[i + 2]   # T+30m (second intermediate)
            c3 = clist[i + 3]   # T+45m / settlement
            gap_ms = c1["open_time_ms"] - c0["close_time_ms"]
            if gap_ms > 5_000: continue
            c2_settle = c3
            secs_entry = 2700.0     # 45 min remaining
            secs_interm = 1800.0    # 30 min remaining

        # ── Settle expired positions ─────────────────────────────────────
        # Before evaluating a new potential trade, we first check if any
        # existing positions have reached their settlement time. If so, we
        # resolve them: compare the settlement price to the strike price,
        # determine win/loss, and update the balance.
        #
        # Why settle BEFORE evaluating new entries?
        # Because settling a position frees up balance and position slots.
        # Without settling first, the simulation might incorrectly skip trades
        # due to a "full" position count, even though some would have just closed.
        still_open = []
        for pos in open_pos:
            if pos["settle_ts_ms"] <= ts_ms:
                _settle_c = pos["settle_c"]
                sp = _settle_c["close"]
                if pos["mtype"].startswith("range_"):
                    rp = float(pos["mtype"].split("_")[1])
                    settled_yes = abs(sp - pos["strike"]) / pos["strike"] <= rp
                else:
                    settled_yes = sp > pos["strike"]
                won = (pos["side"]=="yes" and settled_yes) or (pos["side"]=="no" and not settled_yes)
                ec, n = pos["entry"], pos["n"]
                fee = 0.01 * ec * (100 - ec) / 100 * n  # maker limit orders ≈ 0% fee; 1% buffer
                pnl = int(((100 - ec) * n - fee) if won else (0 - ec * n - fee))
                if won: balance += int(100 * n - fee)
                else:   balance -= int(fee)
                all_trades.append({**pos, "won": won, "pnl": pnl, "exit": "settle"})
                _pc = pos["coin"]  # use position's own coin, not outer-loop coin
                per_coin[_pc]["trades"] += 1
                per_coin[_pc]["wins"]   += int(won)
                per_coin[_pc]["pnl"]    += pnl
                per_coin_open[_pc] = max(0, per_coin_open[_pc] - 1)
                if pnl < 0: daily_loss += pnl
            else:
                still_open.append(pos)
        open_pos = still_open

        if len(open_pos) >= max_positions: continue
        if per_coin_open.get(coin, 0) >= _eff_pos_cap: continue

        # Daily loss circuit breaker
        _day_num = int(ts_ms / 86_400_000)
        if _day_num != _last_day:
            daily_loss = 0; _last_day = _day_num
        if daily_loss <= -(daily_loss_pct * balance):
            continue  # Halt for rest of this simulated day

        # Directional concentration cap
        _yes_open = sum(1 for p in open_pos if p["side"] == "yes")
        _no_open  = sum(1 for p in open_pos if p["side"] == "no")

        # Cap effective sizing balance at realistic Kalshi fill capacity.
        # Actual balance still compounds (shows real cumulative profits),
        # but bet sizing plateaus once we'd exceed market liquidity.
        _eff_balance = min(balance, MAX_COMPOUND_BALANCE)
        budget    = int(_eff_balance * budget_pct)
        exposure  = sum(p["entry"] * p["n"] for p in open_pos)
        available = max(0, budget - exposure)
        if available < MIN_BET: continue

        # ── Compute entry signal ─────────────────────────────────────────
        # "strike" = the target price level the market asks about.
        #   e.g., "Will BTC be above $50,000 at 2:15pm?" — $50,000 is the strike.
        #   In this simulation, we use the candle's OPEN price as the strike
        #   (what the price was at the start of the window).
        #
        # "spot" = the current price of the cryptocurrency.
        #   We use the candle's CLOSE price as the spot (what it was at the end).
        #   The question becomes: given the spot vs the strike at entry, how likely
        #   is the spot to be on the correct side at settlement?
        strike = c0["open"]
        spot   = c0["close"]
        if strike <= 0 or spot <= 0: continue  # invalid candle data → skip

        coin_min_p = max(min_fair_prob, _eff_coin_min.get(coin, min_fair_prob))
        coin_min_no = round(1.0 - coin_min_p, 4)
        liq_cap = LIQUIDITY_CAPS.get(coin, 10_000)

        hour_utc = datetime.fromtimestamp(c0["open_time_ms"]/1000, tz=timezone.utc).hour

        if mtype.startswith("range_"):
            rp = float(mtype.split("_")[1])
            fair_prob = _fair_prob_range(spot, strike, secs_entry, coin, range_pct=rp)
            if fair_prob < coin_min_p: continue
            side = "yes"
            mkt_mid = 0.5 + MKTEFF * (fair_prob - 0.5)
            entry_cents = min(98, int(mkt_mid * 100) + 2 + SLIPPAGE)
            win_prob = fair_prob
        else:
            p_price  = _fair_prob_dt(spot, strike, secs_entry, coin)
            p_tod    = max(0.01, min(0.99, 0.5 + _tod_bias(hour_utc)))
            fair_prob = _combined_fair(p_price, 0.5, 0.5, max(0.01,min(0.99,p_price)), p_tod)

            if fair_prob >= coin_min_p:
                side = "yes"
                mkt_mid = 0.5 + MKTEFF * (fair_prob - 0.5)
                entry_cents = min(98, int(mkt_mid * 100) + 2 + SLIPPAGE)
                win_prob = fair_prob
            elif fair_prob <= coin_min_no:
                side = "no"
                no_fair = 1.0 - fair_prob
                mkt_mid_no = 0.5 + MKTEFF * (no_fair - 0.5)
                entry_cents = min(98, int(mkt_mid_no * 100) + 2 + SLIPPAGE)
                win_prob = no_fair
            else:
                continue

        if entry_cents < 15: continue
        if win_prob - entry_cents / 100.0 < min_edge: continue

        # Directional cap: max `max_directional` same-direction open positions
        if side == "yes" and _yes_open >= max_directional: continue
        if side == "no"  and _no_open  >= max_directional: continue

        # ── Kelly sizing (tranche 1) ──────────────────────────────────────
        # The Kelly fraction scales with confidence:
        #   High confidence (win_prob ≥ 80%): bet 50% of full Kelly
        #   Medium confidence (70–80%):       bet 35% of full Kelly
        #   Lower confidence (below 70%):     bet 20% of full Kelly
        #
        # This is a "tiered Kelly" — more aggressive when the signal is strong,
        # more conservative when the signal is borderline. The tiers match
        # exactly what the live bot uses in server.py, ensuring the backtest
        # reflects real live trading behavior.
        #
        # kelly_mult is an external multiplier that can be swept (e.g., 0.5 = half
        # of the already-fractional Kelly, for more conservative sizing).
        if win_prob >= 0.80:   kf = 0.50 * kelly_mult
        elif win_prob >= 0.70: kf = 0.35 * kelly_mult
        else:                  kf = 0.20 * kelly_mult

        full_kelly_size = _kelly_size(win_prob, entry_cents, MIN_BET,
                                      min(liq_cap, available), kf, balance)
        if full_kelly_size <= 0: continue

        t1_size = int(full_kelly_size * dca_sizes[0])
        if t1_size < MIN_BET: continue
        n1 = max(1, t1_size // max(entry_cents, 1))
        cost1 = entry_cents * n1
        if cost1 > available: continue

        # ── Hedge: buy opposite side at market ───────────────────────────
        # If hedge_ratio > 0, we simultaneously buy a small position on the
        # OPPOSITE side. For a YES trade at entry_cents, the NO side costs
        # approximately (100 - entry_cents) cents.
        #
        # The hedge protects us if we're wrong: if YES loses, the NO position
        # partially recovers the loss. The tradeoff is reduced upside.
        #
        # n_hedge = n1 × hedge_ratio. So with hedge_ratio=0.15, if we bought
        # 10 YES contracts (n1=10), we also buy 1 NO contract (0.15 × 10 = 1.5 → 1).
        hedge_cost = 0
        n_hedge = 0
        if hedge_ratio > 0:
            hedge_entry = min(98, (100 - entry_cents) + SLIPPAGE)  # NO side entry price
            n_hedge = max(0, int(n1 * hedge_ratio))
            hedge_cost = hedge_entry * n_hedge

        total_cost1 = cost1 + hedge_cost
        if total_cost1 > available: continue
        balance -= total_cost1

        # ── DCA Tranche 2 at intermediate candle ─────────────────────────
        # If DCA is enabled (dca_tranches >= 2), we check the price at the
        # intermediate candle (T+5 minutes for 15m markets) and add a second
        # tranche of the position at that new price.
        #
        # The key check: 'dca_only_better' enforces that we only add the second
        # tranche if the price IMPROVED (is cheaper or same) since the first entry.
        # We don't want to average into a position that's moving against us.
        #
        # This only applies to 15m markets (5m candles) because we have an
        # intermediate data point at T+5 minutes. For 1h markets (15m candles),
        # the intermediate candles represent larger time steps.
        t2_size = 0
        t2_entry = 0
        n2 = 0
        if dca_tranches >= 2 and cinterval == "5m":
            fp2 = _fair_prob_dt(c1["close"], strike, secs_interm, coin)
            p_tod2 = max(0.01, min(0.99, 0.5 + _tod_bias(hour_utc)))
            fp2_comb = _combined_fair(fp2, 0.5, 0.5, max(0.01,min(0.99,fp2)), p_tod2)
            mid2 = 0.5 + MKTEFF * ((fp2_comb if side=="yes" else 1-fp2_comb) - 0.5)
            t2_entry = min(98, int(mid2 * 100) + 2 + SLIPPAGE)
            still_valid = (fp2_comb >= coin_min_p) if side=="yes" else (fp2_comb <= coin_min_no)
            price_better = (t2_entry <= entry_cents) if side=="yes" else (t2_entry <= entry_cents)
            if still_valid and (not dca_only_better or price_better):
                avail2 = max(0, int(balance * 0.80) - sum(p["entry"]*p["n"] for p in open_pos) - cost1)
                t2_size = int(full_kelly_size * dca_sizes[1])
                t2_size = max(0, min(t2_size, avail2))
                if t2_size >= MIN_BET:
                    n2 = max(1, t2_size // max(t2_entry, 1))
                    balance -= t2_entry * n2
                    per_coin[coin]["dca_adds"] += 1
                else:
                    n2 = 0

        # ── DCA Tranche 3 ─────────────────────────────────────────────────
        n3 = 0
        t3_entry = entry_cents
        if dca_tranches >= 3 and cinterval == "5m" and n2 > 0:
            # Use c2's close as a 3rd entry — but c2 IS settlement candle, skip
            # Instead, for 3 tranches: split T1+T2 and add a partial at T+5m also
            # Implement as: T1 at c0.close (already done), T2 at c1.close, no T3
            pass  # 3-tranche only meaningful for 1h markets (4 × 15m windows)

        # ── Determine settlement candle ───────────────────────────────────
        if cinterval == "5m":
            settle_c = c2
            settle_ts = c2["close_time_ms"]
        else:
            settle_c = c3  # type: ignore
            settle_ts = c3["close_time_ms"]  # type: ignore

        total_n = n1 + n2
        if total_n <= 0: continue

        # Compute blended entry for P&L tracking
        blended_entry = (entry_cents * n1 + t2_entry * n2) // total_n if total_n > 0 else entry_cents

        pos = {
            "coin": coin, "side": side, "strike": strike,
            "entry": blended_entry, "entry1": entry_cents, "entry2": t2_entry,
            "n": total_n, "n1": n1, "n2": n2, "n_hedge": n_hedge,
            "hedge_entry": (100 - entry_cents + SLIPPAGE) if n_hedge > 0 else 0,
            "fair_prob": round(fair_prob, 3), "win_prob": round(win_prob, 3),
            "mtype": mtype, "settle_ts_ms": settle_ts, "settle_c": settle_c,
            "hour": hour_utc, "spot": spot,
            "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct,
        }

        # ── Intermediate stop-loss / take-profit check ────────────────────
        # This is what makes this simulation more realistic than a simple
        # "enter at open, settle at close" model. We check the intermediate
        # candle price to see if stop-loss or take-profit would have been
        # triggered BEFORE settlement.
        #
        # Why this matters: a YES trade at 55¢ that settles correctly at 100¢
        # might have temporarily dipped to 10¢ at the intermediate check.
        # With stop_loss_pct=0.75, the stop trigger is 55¢ × 0.25 = 13.75¢.
        # At 10¢, the stop would fire — we'd exit for a large loss even though
        # the trade eventually would have won. That's realistic!
        #
        # Without this check, stop-losses look artificially better in backtests
        # because you'd only ever exit at settlement. This is look-ahead bias.
        #
        # Only possible for 15m markets (which have a c1 intermediate candle at T+5m).
        exited_early = False
        if cinterval == "5m" and (stop_loss_pct > 0 or take_profit_pct > 0):
            fp_interm = _fair_prob_dt(c1["close"], strike, secs_interm, coin)
            pt2 = max(0.01, min(0.99, 0.5 + _tod_bias(hour_utc)))
            fp_interm_comb = _combined_fair(fp_interm, 0.5, 0.5, max(0.01,min(0.99,fp_interm)), pt2)

            if side == "yes":
                mid_interm = 0.5 + MKTEFF * (fp_interm_comb - 0.5)
            else:
                mid_interm = 0.5 + MKTEFF * ((1 - fp_interm_comb) - 0.5)
            interm_exit_cents = max(1, min(98, int(mid_interm * 100) - SLIPPAGE))

            # Stop-loss: exit if current value dropped to (1-stop_loss_pct) of entry
            if stop_loss_pct > 0:
                stop_trigger = entry_cents * (1.0 - stop_loss_pct)
                if interm_exit_cents <= stop_trigger:
                    pnl_n1 = int((interm_exit_cents - entry_cents) * n1)
                    pnl_n2 = int((interm_exit_cents - t2_entry) * n2) if n2 > 0 else 0
                    pnl = pnl_n1 + pnl_n2
                    fee  = 0.01 * interm_exit_cents * (100 - interm_exit_cents) / 100 * total_n
                    pnl  -= int(fee)
                    balance += int(interm_exit_cents * total_n - fee)
                    # Hedge recovery
                    if n_hedge > 0:
                        h_entry = pos["hedge_entry"]
                        h_fee   = 0.01 * h_entry * (100 - h_entry) / 100 * n_hedge
                        # Hedge wins when main loses → interm_exit < entry means main is losing
                        h_won   = not (side == "yes")  # if main is down, hedge (NO) is up
                        if h_won:
                            balance += int(100 * n_hedge - h_fee)
                        else:
                            balance -= int(h_fee)
                    all_trades.append({**pos, "won": False, "pnl": pnl, "exit": "stop_loss"})
                    per_coin[coin]["trades"] += 1
                    per_coin[coin]["pnl"]    += pnl
                    per_coin[coin]["stop_exits"] += 1
                    exited_early = True

            # Take-profit: exit if gain > take_profit_pct × (100 - entry)
            if not exited_early and take_profit_pct > 0:
                max_gain = 100 - entry_cents
                current_gain = interm_exit_cents - entry_cents
                if current_gain >= take_profit_pct * max_gain:
                    pnl_n1 = int((interm_exit_cents - entry_cents) * n1)
                    pnl_n2 = int((interm_exit_cents - t2_entry) * n2) if n2 > 0 else 0
                    pnl = pnl_n1 + pnl_n2
                    fee  = 0.01 * interm_exit_cents * (100 - interm_exit_cents) / 100 * total_n
                    pnl  -= int(fee)
                    balance += int(interm_exit_cents * total_n - fee)
                    all_trades.append({**pos, "won": True, "pnl": pnl, "exit": "take_profit"})
                    per_coin[coin]["trades"] += 1
                    per_coin[coin]["wins"]   += 1
                    per_coin[coin]["pnl"]    += pnl
                    per_coin[coin]["tp_exits"] += 1
                    exited_early = True

        if not exited_early:
            open_pos.append(pos)
            per_coin_open[coin] = per_coin_open.get(coin, 0) + 1
        # early exit already deducted entry cost; per_coin_open stays 0 (never incremented)

    # ── Settle remaining positions ────────────────────────────────────────────
    for pos in open_pos:
        sp = pos["settle_c"]["close"]
        if pos["mtype"].startswith("range_"):
            rp = float(pos["mtype"].split("_")[1])
            settled_yes = abs(sp - pos["strike"]) / pos["strike"] <= rp
        else:
            settled_yes = sp > pos["strike"]
        won = (pos["side"]=="yes" and settled_yes) or (pos["side"]=="no" and not settled_yes)
        ec, n = pos["entry"], pos["n"]
        fee = 0.01 * ec * (100 - ec) / 100 * n  # maker limit orders ≈ 0% fee; 1% buffer
        pnl = int(((100 - ec) * n - fee) if won else (0 - ec * n - fee))
        if won: balance += int(100 * n - fee)
        else:   balance -= int(fee)

        # Hedge settlement
        if pos.get("n_hedge", 0) > 0:
            h_entry = pos["hedge_entry"]
            h_fee   = 0.01 * h_entry * (100 - h_entry) / 100 * pos["n_hedge"]
            h_won   = not (pos["side"] == "yes" and settled_yes) and not (pos["side"] == "no" and not settled_yes)
            if h_won:
                balance += int(100 * pos["n_hedge"] - h_fee)
                pnl += int((100 - h_entry) * pos["n_hedge"] - h_fee)
            else:
                balance -= int(h_fee)
                pnl -= int(h_entry * pos["n_hedge"] + h_fee)

        all_trades.append({**pos, "won": won, "pnl": pnl, "exit": "settle"})
        c = pos["coin"]
        per_coin[c]["trades"] += 1
        per_coin[c]["wins"]   += int(won)
        per_coin[c]["pnl"]    += pnl

    if not all_trades:
        return {}

    # ── Aggregate results ─────────────────────────────────────────────────────
    # The simulation has finished replaying all historical moments. Now we
    # compute the summary statistics from the full trade history.
    total    = len(all_trades)
    wins     = sum(1 for t in all_trades if t["won"])
    wr       = wins / total if total else 0
    total_pnl = balance - START_BAL   # Net profit = final balance − starting balance

    gross_w  = sum(t["pnl"] for t in all_trades if t["won"])
    gross_l  = abs(sum(t["pnl"] for t in all_trades if not t["won"]))
    pf       = gross_w / gross_l if gross_l > 0 else float("inf")

    equity, peak, max_dd = START_BAL, START_BAL, 0
    pnl_series, equity_curve = [], []
    for t in all_trades:
        equity += t["pnl"]; peak = max(peak, equity)
        max_dd  = max(max_dd, peak - equity)
        pnl_series.append(t["pnl"]); equity_curve.append(equity)

    # ── Sharpe Ratio calculation ──────────────────────────────────────────────
    # We compute Sharpe using DAILY P&L (not per-trade), then annualize.
    #
    # WHY daily, not per-trade?
    # If we annualize per-trade Sharpe directly, we get an inflated number.
    # A strategy making 10 trades/day looks 10× "more consistent" on a per-trade
    # basis than one making 1 trade/day, even if their daily outcomes are identical.
    # Aggregating to daily P&L normalizes for trade frequency.
    #
    # Annualization: multiply by sqrt(365) because volatility scales with
    # sqrt(time). A strategy with a daily Sharpe of 0.1 has an annualized
    # Sharpe of 0.1 × sqrt(365) ≈ 1.91.
    _daily_pnl: dict[int, int] = {}
    for _t in all_trades:
        # Convert millisecond timestamp to day number (integer division by 86,400,000ms/day)
        _dk = int(_t["settle_ts_ms"] / 86_400_000)
        _daily_pnl[_dk] = _daily_pnl.get(_dk, 0) + _t["pnl"]
    _daily_series = list(_daily_pnl.values())
    if len(_daily_series) > 1:
        _avg_d = statistics.mean(_daily_series)             # average daily P&L
        _std_d = statistics.stdev(_daily_series)            # standard deviation of daily P&L
        sharpe = (_avg_d / _std_d) * math.sqrt(365) if _std_d > 0 else 0.0
    else:
        sharpe = 0.0   # can't compute Sharpe without at least 2 data points

    stop_exits = sum(1 for t in all_trades if t.get("exit")=="stop_loss")
    tp_exits   = sum(1 for t in all_trades if t.get("exit")=="take_profit")
    dca_adds   = sum(v["dca_adds"] for v in per_coin.values())

    # ── Weekly breakdown (7-day buckets from first trade) ─────────────────────
    import datetime as _dt
    _sorted_trades = sorted(all_trades, key=lambda t: t["settle_ts_ms"])
    _weekly_stats: list[dict] = []
    if _sorted_trades:
        _t0      = _sorted_trades[0]["settle_ts_ms"]
        _week_ms = 7 * 24 * 3600 * 1000
        _running_bal = START_BAL
        for _w in range(14):
            _ws = _t0 + _w * _week_ms
            _we = _ws + _week_ms
            _wt = [t for t in _sorted_trades if _ws <= t["settle_ts_ms"] < _we]
            if not _wt:
                if _ws > _sorted_trades[-1]["settle_ts_ms"]:
                    break
                continue
            _w_pnl   = sum(t["pnl"] for t in _wt)
            _w_wins  = sum(1 for t in _wt if t["won"])
            _w_avg_b = int(statistics.mean(t["entry"] * t["n"] for t in _wt))
            _ds = _dt.datetime.utcfromtimestamp(_ws / 1000).strftime("%m/%d")
            _de = _dt.datetime.utcfromtimestamp((_we - 1) / 1000).strftime("%m/%d")
            _weekly_stats.append({
                "week":                _w + 1,
                "date_range":          f"{_ds}–{_de}",
                "trades":              len(_wt),
                "wins":                _w_wins,
                "win_rate":            _w_wins / len(_wt),
                "pnl_cents":           _w_pnl,
                "start_balance_cents": _running_bal,
                "end_balance_cents":   _running_bal + _w_pnl,
                "avg_bet_cents":       _w_avg_b,
            })
            _running_bal += _w_pnl

    return {
        "total_trades": total, "wins": wins, "losses": total-wins,
        "win_rate": wr, "total_pnl_cents": total_pnl,
        "final_balance_cents": balance, "profit_factor": pf,
        "max_drawdown_cents": max_dd, "ann_sharpe": sharpe,
        "per_coin": per_coin,
        "stop_exits": stop_exits, "tp_exits": tp_exits, "dca_adds": dca_adds,
        "n_days": n_days, "equity_curve": equity_curve[-200:],
        "weekly_stats": _weekly_stats,
    }


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_report(r: dict, label: str = ""):
    """Print a human-readable summary of one backtest run's results.

    This formats the raw numbers from run_strategy_backtest() into a
    readable table. It's only used when running the script from the command
    line — the REST API in server.py returns the raw dict instead.

    The metrics shown are:
      Trades/day:    How often the strategy would trade (frequency check)
      Win rate:      What % of trades made money
      Stop/TP exits: How many trades were closed early by stop-loss or take-profit
      DCA adds:      How many second-tranche entries were made (DCA usage)
      Total PnL:     Net profit over the test period
      Ann. return:   What yearly return this implies (90d × 365/90)
      Profit factor: Gross wins / Gross losses (>1 = profitable)
      Max drawdown:  Worst peak-to-trough loss ($)
      Ann. Sharpe:   Annualized Sharpe ratio (target: >1.0)
    """
    if not r: return
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*68}")
    print(f"  STRATEGY RESULTS — {r['n_days']}-day window{tag}")
    print(f"{'='*68}")
    total = r["total_trades"]
    print(f"  Trades:       {total}  ({total/r['n_days']:.1f}/day)")
    print(f"  Win rate:     {r['win_rate']:.1%}  ({r['wins']}W / {r['losses']}L)")
    print(f"  Stop exits:   {r['stop_exits']}  |  TP exits: {r['tp_exits']}  |  DCA adds: {r['dca_adds']}")
    print(f"  Total PnL:    ${r['total_pnl_cents']/100:+.2f}")
    print(f"  Balance:      ${r['final_balance_cents']/100:.2f}  (started ${START_BAL/100:.2f})")
    print(f"  Return:       {r['total_pnl_cents']/START_BAL*100:+.1f}%  over {r['n_days']}d")
    print(f"  Ann. return:  {r['total_pnl_cents']/START_BAL*100*365/r['n_days']:+.1f}%")
    print(f"  Profit factor:{r['profit_factor']:.2f}x")
    print(f"  Max drawdown: ${r['max_drawdown_cents']/100:.2f}")
    print(f"  Ann. Sharpe:  {r['ann_sharpe']:.2f}")
    print(f"\n  ── Per-coin ──")
    for coin, s in r["per_coin"].items():
        n = s["trades"]
        if n == 0: continue
        wr_c = s["wins"]/n
        print(f"  {coin:12s} {n:4d}T  WR={wr_c:.1%}  PnL=${s['pnl']/100:+.2f}"
              f"  stops={s['stop_exits']}  tp={s['tp_exits']}  dca+={s['dca_adds']}")
    print(f"{'='*68}\n")


def print_weekly_report(r: dict, label: str = ""):
    """Print a week-by-week breakdown of trades, P&L, and balance growth.

    Why look at weekly performance?

    A 90-day backtest might look great overall but could mask that all the
    profits came from 2 lucky weeks. Printing week-by-week reveals whether
    the strategy is CONSISTENT across time.

    What you want to see:
      - Positive P&L most weeks (not just a few big winners)
      - Win rate relatively stable across weeks (no "lucky streaks")
      - Balance growing roughly linearly (not spiking then crashing)

    A strategy with consistently positive weekly P&L is far more reliable
    than one that makes all its money in 2 weeks and breaks even the rest.
    """
    ws = r.get("weekly_stats")
    if not ws:
        return
    tag = f" [{label}]" if label else ""
    print(f"\n{'─'*82}")
    print(f"  WEEKLY BREAKDOWN{tag}")
    print(f"{'─'*82}")
    hdr = (f"  {'Wk':>2}  {'Dates':>11}  {'Trades':>6}  {'WR':>6}  "
           f"{'PnL':>9}  {'Start Bal':>10}  {'End Bal':>10}  {'Avg Bet':>8}")
    print(hdr)
    print(f"  {'─'*76}")
    for w in ws:
        print(
            f"  {w['week']:>2}  {w['date_range']:>11}  {w['trades']:>6}  "
            f"{w['win_rate']:>5.1%}  ${w['pnl_cents']/100:>+8.2f}  "
            f"${w['start_balance_cents']/100:>9.2f}  "
            f"${w['end_balance_cents']/100:>9.2f}  "
            f"${w['avg_bet_cents']/100:>7.2f}"
        )
    print(f"{'─'*82}\n")


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep(sweep_mode: str, n_days: int = 90):
    """Run a systematic parameter sweep across many strategy configurations.

    This function coordinates the full sweep process:
      1. Pre-fetch all needed candle data ONCE (shared across all sweep runs).
         This is important — re-fetching for each combination would take hours.
      2. Define the set of parameter combinations to test (see 'combos' list).
      3. Run run_strategy_backtest() for each combination.
      4. Print a comparison table and highlight the best result.

    Pre-fetching is done for BOTH 5m and 15m candles for each coin because:
      - 15m markets use 5m candles (3 sub-candles per market window)
      - 1h markets use 15m candles (4 sub-candles per market window)

    sweep_mode choices:
      'exit':      Test different stop-loss and take-profit percentages
      'dca':       Test different DCA configurations (1/2/3 tranches, split ratios)
      'hedge':     Test different hedge ratios combined with stop-loss
      'full':      All combinations of exit + DCA + hedge (216 runs)
      'combined':  Curated "best of breed" combinations (12 runs, faster)
      'threshold': Sweep entry probability thresholds and coin sets
      'coins':     Test 7-coin vs 10-coin universe
      'positions': Test different per-coin position caps
      'risk':      Sweep budget, directional cap, and daily loss limits
      'kelly':     Test different Kelly multipliers
      'markets':   Compare 15m, 1h, and range market type combinations
      'coinprob':  Fine-tune per-coin probability floor thresholds
      'optimize':  Round 1: kelly multiplier vs budget fraction
      'optimize2': Round 2: directional cap vs per-coin limit
      'optimize3': Round 3: min fair probability vs range bracket widths
    """

    # ── Pre-fetch candles — extended set for threshold/coins sweeps ──────────
    _fetch_set = COINS_EXTENDED if sweep_mode in ("threshold", "coins", "positions", "risk", "kelly", "markets", "coinprob", "optimize", "optimize2", "optimize3") else COINS_TARGET
    print(f"\nPre-fetching candles for {n_days}-day sweep ({len(_fetch_set)} coins)...")
    candles_5m: dict[str, list] = {}
    candles_15m: dict[str, list] = {}
    for coin, symbol in _fetch_set.items():
        try:
            candles_5m[coin]  = fetch_candles(symbol, n_days, "5m")
            candles_15m[coin] = fetch_candles(symbol, n_days, "15m")
        except Exception as e:
            print(f"  [skip] {coin}: {e}")

    # ── Define sweep grids ────────────────────────────────────────────────────
    if sweep_mode == "exit":
        combos = [
            {"label": f"SL={sl:.0%} TP={tp:.0%}",
             "stop_loss_pct": sl, "take_profit_pct": tp,
             "dca_tranches": 1, "hedge_ratio": 0.0}
            for sl in [0.0, 0.25, 0.50, 0.75, 0.90]
            for tp in [0.0, 0.25, 0.50, 0.75]
        ]

    elif sweep_mode == "dca":
        combos = [
            {"label": f"DCA={t}T sizes={s} only_better={ob}",
             "stop_loss_pct": 0.75, "take_profit_pct": 0.0,
             "dca_tranches": t, "dca_sizes": s,
             "dca_only_better": ob, "hedge_ratio": 0.0}
            for t, s in [
                (1, [1.0]),
                (2, [0.60, 0.40]),
                (2, [0.70, 0.30]),
                (2, [0.50, 0.50]),
                (3, [0.50, 0.30, 0.20]),
                (3, [0.40, 0.35, 0.25]),
            ]
            for ob in [True, False]
        ]

    elif sweep_mode == "hedge":
        combos = [
            {"label": f"Hedge={hr:.0%} SL={sl:.0%}",
             "stop_loss_pct": sl, "take_profit_pct": 0.0,
             "dca_tranches": 1, "hedge_ratio": hr}
            for hr in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
            for sl in [0.0, 0.50, 0.75]
        ]

    elif sweep_mode == "full":
        combos = [
            {"label": f"SL={sl:.0%} TP={tp:.0%} DCA={t}T H={hr:.0%}",
             "stop_loss_pct": sl, "take_profit_pct": tp,
             "dca_tranches": t, "dca_sizes": s, "hedge_ratio": hr}
            for sl in [0.0, 0.50, 0.75]
            for tp in [0.0, 0.25, 0.50]
            for t, s  in [(1,[1.0]),(2,[0.60,0.40]),(3,[0.50,0.30,0.20])]
            for hr in [0.0, 0.15, 0.25]
        ]

    elif sweep_mode == "combined":
        # Best-of-breed combinations: well-chosen sweep points
        combos = [
            # Baselines
            {"label": "Baseline (no exit/dca/hedge)",
             "stop_loss_pct":0.0,"take_profit_pct":0.0,"dca_tranches":1,"hedge_ratio":0.0},
            {"label": "SL=75% only",
             "stop_loss_pct":0.75,"take_profit_pct":0.0,"dca_tranches":1,"hedge_ratio":0.0},
            {"label": "TP=50% only",
             "stop_loss_pct":0.0,"take_profit_pct":0.50,"dca_tranches":1,"hedge_ratio":0.0},
            {"label": "SL=75%+TP=50%",
             "stop_loss_pct":0.75,"take_profit_pct":0.50,"dca_tranches":1,"hedge_ratio":0.0},
            # DCA variants
            {"label": "DCA-2T(60/40)+SL75",
             "stop_loss_pct":0.75,"take_profit_pct":0.0,"dca_tranches":2,
             "dca_sizes":[0.60,0.40],"hedge_ratio":0.0},
            {"label": "DCA-2T(70/30)+SL75",
             "stop_loss_pct":0.75,"take_profit_pct":0.0,"dca_tranches":2,
             "dca_sizes":[0.70,0.30],"hedge_ratio":0.0},
            {"label": "DCA-3T(50/30/20)+SL75",
             "stop_loss_pct":0.75,"take_profit_pct":0.0,"dca_tranches":3,
             "dca_sizes":[0.50,0.30,0.20],"hedge_ratio":0.0},
            # Hedge variants
            {"label": "Hedge=15%+SL75",
             "stop_loss_pct":0.75,"take_profit_pct":0.0,"dca_tranches":1,"hedge_ratio":0.15},
            {"label": "Hedge=25%+SL75",
             "stop_loss_pct":0.75,"take_profit_pct":0.0,"dca_tranches":1,"hedge_ratio":0.25},
            # Full combos
            {"label": "DCA2T+SL75+TP50+H15%",
             "stop_loss_pct":0.75,"take_profit_pct":0.50,"dca_tranches":2,
             "dca_sizes":[0.60,0.40],"hedge_ratio":0.15},
            {"label": "DCA2T+SL50+TP25+H10%",
             "stop_loss_pct":0.50,"take_profit_pct":0.25,"dca_tranches":2,
             "dca_sizes":[0.60,0.40],"hedge_ratio":0.10},
            {"label": "DCA3T+SL75+TP50+H20%",
             "stop_loss_pct":0.75,"take_profit_pct":0.50,"dca_tranches":3,
             "dca_sizes":[0.50,0.30,0.20],"hedge_ratio":0.20},
        ]
    elif sweep_mode == "threshold":
        # Sweep entry threshold, coin set, and per-coin position cap.
        # Goal: find the configuration that maximises total PnL through more trades.
        combos = []
        for thr in [0.52, 0.54, 0.56, 0.58, 0.60, 0.62]:
            for coin_set, coin_label in [
                (COINS_TARGET,   "7-coins"),
                (COINS_EXTENDED, "10-coins"),
            ]:
                for pos_cap in [3, 4, 5]:
                    # Per-coin floors: keep BNB/HYPE elevated, lower others to thr
                    overridden_min_prob = {k: max(thr, v) for k, v in COIN_MIN_PROB.items()}
                    combos.append({
                        "label": f"thr={thr:.2f} {coin_label} pos={pos_cap}",
                        "min_fair_prob": thr,
                        "coins_override": coin_set,
                        "_pos_cap": pos_cap,
                        "_coin_min_prob_override": overridden_min_prob,
                    })

    elif sweep_mode == "coins":
        # How much do extra coins add?  Test base 7 vs extended 10 at best threshold.
        combos = []
        for coin_set, label in [
            (COINS_TARGET,   "7-coins (BTC ETH XRP SOL DOGE BNB HYPE)"),
            (COINS_EXTENDED, "10-coins (+LTC ADA AVAX)"),
        ]:
            for thr in [0.56, 0.58, 0.60]:
                combos.append({
                    "label": f"{label} thr={thr:.2f}",
                    "min_fair_prob": thr,
                    "coins_override": coin_set,
                })

    elif sweep_mode == "risk":
        # Sweep risk controls: budget_pct, max_directional, per-coin limit, daily_loss_pct.
        # Goal: find the combination that maximises profit while bounding worst-case loss.
        combos = []
        for bp in [0.30, 0.40, 0.50, 0.60]:
            for md in [3, 4, 5, 6]:
                for pc in [2, 3, 4]:
                    for dl in [0.15, 0.20, 0.25]:
                        combos.append({
                            "label": f"bgt={bp:.0%} dir={md} coin={pc} dloss={dl:.0%}",
                            "budget_pct": bp, "max_directional": md,
                            "_pos_cap": pc, "daily_loss_pct": dl,
                            "coins_override": COINS_EXTENDED,
                        })

    elif sweep_mode == "positions":
        # How does MAX_POS_PER_COIN affect throughput & profit?
        combos = []
        for pos_cap in [2, 3, 4, 5, 6]:
            for thr in [0.56, 0.58, 0.60]:
                combos.append({
                    "label": f"pos_cap={pos_cap} thr={thr:.2f}",
                    "min_fair_prob": thr,
                    "coins_override": COINS_EXTENDED,
                    "_pos_cap": pos_cap,
                })

    elif sweep_mode == "kelly":
        # How does Kelly multiplier affect profitability?
        # The base fractions are probability-tiered (0.50/0.35/0.20 × 0.25).
        # kelly_mult scales all of them. 1.0 = current default.
        # Higher = bigger bets = more profit, but also more variance.
        # At 94%+ WR we expect higher mult to dominate.
        combos = []
        for km in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
            combos.append({
                "label": f"kelly_mult={km:.2f}",
                "kelly_mult": km,
                "coins_override": COINS_EXTENDED,
            })

    elif sweep_mode == "markets":
        # Which market types (15m, 1h, range) contribute to profit?
        # Tests all combinations. 15m is always on (it's the core signal).
        # Range markets (3 strike distances) may have different WR profiles.
        combos = [
            {"label": "15m only",           "include_1h": False, "include_range": False, "coins_override": COINS_EXTENDED},
            {"label": "15m + 1h",           "include_1h": True,  "include_range": False, "coins_override": COINS_EXTENDED},
            {"label": "15m + range",        "include_1h": False, "include_range": True,  "coins_override": COINS_EXTENDED},
            {"label": "15m + 1h + range",   "include_1h": True,  "include_range": True,  "coins_override": COINS_EXTENDED},
            # Range only at tighter strikes — less vol needed to win
            {"label": "15m + range(0.5%)",  "include_1h": False, "include_range": True,  "range_pcts": [0.005], "coins_override": COINS_EXTENDED},
            {"label": "15m + range(1%)",    "include_1h": False, "include_range": True,  "range_pcts": [0.010], "coins_override": COINS_EXTENDED},
            {"label": "15m + range(2%)",    "include_1h": False, "include_range": True,  "range_pcts": [0.020], "coins_override": COINS_EXTENDED},
        ]

    elif sweep_mode == "coinprob":
        # Fine-tune per-coin min probability floors.
        # Current: BNB=0.75, HYPE=0.72, LTC=0.82, others=0.60.
        # Test whether BNB/HYPE/LTC floors are set optimally.
        _base = {c: 0.60 for c in COINS_EXTENDED}
        combos = []
        for bnb_p in [0.60, 0.65, 0.70, 0.75, 0.80]:
            for hype_p in [0.60, 0.65, 0.70, 0.72, 0.75]:
                for ltc_p in [0.60, 0.65, 0.70, 0.75, 0.80, 0.82]:
                    override = {**_base, "bnb": bnb_p, "hyperliquid": hype_p, "litecoin": ltc_p}
                    combos.append({
                        "label": f"bnb={bnb_p:.2f} hype={hype_p:.2f} ltc={ltc_p:.2f}",
                        "_coin_min_prob_override": override,
                        "coins_override": COINS_EXTENDED,
                    })

    elif sweep_mode == "optimize":
        # ── Round 1: kelly_mult × budget_pct ────────────────────────────────
        # Primary driver of trades/day: lower kelly → smaller bets → more
        # concurrent positions fit in budget before cap → more total entries.
        # All runs: 9 coins, all market types, no DCA/SL/TP, dir=6 (generous cap).
        combos = []
        for km in [0.25, 0.50, 0.75, 1.0, 1.5]:
            for bp in [0.40, 0.60, 0.80]:
                combos.append({
                    "label": f"km={km:.2f} bgt={bp:.0%}",
                    "kelly_mult": km, "budget_pct": bp,
                    "max_directional": 6,
                    "coins_override": COINS_EXTENDED,
                    "include_15m": True, "include_1h": True, "include_range": True,
                })

    elif sweep_mode == "optimize2":
        # ── Round 2: directional cap × pos-per-coin at best kelly/budget ────
        # Fix km=0.25 / budget=0.80 (expected winner from optimize round 1).
        # Test how directional cap and per-coin limit affect throughput & PF.
        combos = []
        for md in [4, 5, 6, 8]:
            for pc in [2, 3, 4]:
                combos.append({
                    "label": f"dir={md} coin={pc}",
                    "kelly_mult": 0.25, "budget_pct": 0.80,
                    "max_directional": md, "_pos_cap": pc,
                    "coins_override": COINS_EXTENDED,
                    "include_15m": True, "include_1h": True, "include_range": True,
                })

    elif sweep_mode == "optimize3":
        # ── Round 3: min_fair_prob × range_pcts at best settings ────────────
        # Fix best settings from round 1+2. Test signal quality thresholds
        # and range market bracket widths.
        combos = []
        for mfp in [0.55, 0.58, 0.60, 0.62, 0.65]:
            for rpcts in [[0.005,0.010,0.020], [0.005,0.010,0.020,0.030], [0.010,0.020]]:
                combos.append({
                    "label": f"mfp={mfp:.2f} rng={'|'.join(f'{r:.1%}' for r in rpcts)}",
                    "min_fair_prob": mfp,
                    "kelly_mult": 0.25, "budget_pct": 0.80,
                    "max_directional": 6, "_pos_cap": 3,
                    "range_pcts": rpcts,
                    "coins_override": COINS_EXTENDED,
                    "include_15m": True, "include_1h": True, "include_range": True,
                })

    else:
        raise ValueError(f"Unknown sweep: {sweep_mode}. Use: exit | dca | hedge | full | combined | threshold | coins | positions | risk | kelly | markets | coinprob | optimize | optimize2 | optimize3")

    print(f"\nRunning {len(combos)} strategy combinations...\n")
    hdr = f"{'Strategy':<42} {'T/d':>5} {'WR':>7} {'PnL':>9} {'DD':>7} {'PF':>5} {'Sharpe':>7} {'Stops':>6} {'TPs':>5} {'DCA+':>5}"
    print(hdr)
    print("─" * len(hdr))

    results = []
    for combo in combos:
        lbl   = combo.pop("label")
        r = run_strategy_backtest(
            n_days=n_days,
            candles_5m=candles_5m,
            candles_15m=candles_15m,
            **combo,
        )
        combo["label"] = lbl  # restore
        if not r:
            print(f"  {lbl:<42}  NO TRADES"); continue

        tpd    = r["total_trades"] / n_days
        wr     = r["win_rate"]
        pnl    = r["total_pnl_cents"] / 100
        dd     = r["max_drawdown_cents"] / 100
        pf     = r["profit_factor"]
        sharpe = r["ann_sharpe"]
        results.append((pnl, lbl, r, combo))
        print(f"  {lbl:<42} {tpd:>5.1f}  {wr:>6.1%}  ${pnl:>+8.2f}  ${dd:>6.2f}  {pf:>4.2f}x  {sharpe:>6.1f}  {r['stop_exits']:>5}  {r['tp_exits']:>4}  {r['dca_adds']:>4}")

    if results:
        results.sort(key=lambda x: x[0], reverse=True)
        best_pnl, best_lbl, best_r, best_combo = results[0]
        print(f"\n  ► Best: {best_lbl}  (${best_pnl:+.2f})")
        print_report(best_r, label=best_lbl)

        print("\n  ► Top 5 by PnL:")
        for rank, (pnl, lbl, r, _) in enumerate(results[:5], 1):
            print(f"  #{rank} {lbl:<44} ${pnl:+.2f}  WR={r['win_rate']:.1%}  DD=${r['max_drawdown_cents']/100:.2f}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sweep = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    n_days = 90

    if "--sweep" in sys.argv and sweep:
        run_sweep(sweep, n_days=n_days)
    else:
        # Compare three budget sizes: 10%, 25%, 33%
        # All other params are optimal: 9 coins, all market types, single-entry.
        print(f"\n{'='*68}")
        print(f"  STRATEGY BACKTEST — budget comparison: 10% / 25% / 33%")
        print(f"  9 coins | 15m binary + 1h binary + hourly range")
        print(f"  Single entry | no DCA | no hedge | no SL/TP")
        print(f"  Kelly: 0.50/0.35/0.20 tiers | fee=1%")
        print(f"  Starting balance: ${START_BAL/100:.2f}")
        print(f"{'='*68}\n")
        print("Pre-fetching candles (shared across all 3 runs)...")
        c5m, c15m = {}, {}
        for coin, sym in COINS_EXTENDED.items():
            try:
                c5m[coin]  = fetch_candles(sym, n_days, "5m")
                c15m[coin] = fetch_candles(sym, n_days, "15m")
            except Exception as e:
                print(f"  [skip] {coin}: {e}")

        for budget_pct, pct_label in [(0.10, "10%"), (0.20, "20%"), (0.25, "25%"), (0.33, "33%")]:
            lbl = f"budget={pct_label} | single-entry | 15m+1h+range | 9 coins | Kelly tiers"
            print(f"\n{'~'*68}")
            print(f"  Running budget={pct_label}...")
            print(f"{'~'*68}")
            r = run_strategy_backtest(
                n_days=n_days, candles_5m=c5m, candles_15m=c15m,
                coins_override=COINS_EXTENDED,
                stop_loss_pct=0.0, take_profit_pct=0.0,
                dca_tranches=1,
                hedge_ratio=0.0,
                include_15m=True, include_1h=True, include_range=True,
                budget_pct=budget_pct,
            )
            print_report(r, label=lbl)
            print_weekly_report(r, label=f"budget={pct_label}")
