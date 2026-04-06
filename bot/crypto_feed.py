"""
Real-time crypto price feed and funding rate signals for the prediction market bot.

================================================================================
WHAT IS THIS FILE?
================================================================================

This file gathers live market data from the internet and turns it into trading
signals — mathematical numbers that tell the bot whether to bet YES, bet NO, or
skip a particular Kalshi market.

Think of it as the bot's "eyes and ears" for what's happening in crypto markets
RIGHT NOW. Without this data, the bot would be blind — it would have no idea
whether Bitcoin is currently above or below the market's target price.

================================================================================
WHAT IS A TRADING SIGNAL?
================================================================================

A "signal" is any piece of data that gives you useful information about which
way a price is likely to move. This module produces four signals:

  Signal 1 — Live price vs. target (compute_live_price_signal):
    "Bitcoin is currently $500 ABOVE the market's target price" → lean YES
    This is the strongest signal. If the price is already above the target with
    5 minutes until the market closes, it's very likely to stay above.

  Signal 2 — Time-of-day bias (compute_time_of_day_bias):
    "It's 9-11 PM UTC, which historically sees slight crypto price rises"
    This is a very weak signal (+/-0.05) based on known market patterns.

  Signal 3 — Funding rate bias (compute_funding_rate_bias):
    "Too many leveraged longs on Binance — a correction might be coming"
    A contrarian signal based on how overleveraged traders are.

  Signal 4 — Kelly criterion sizing (kelly_criterion_size):
    Not a direction signal — this calculates HOW MUCH to bet given the edge size.

These signals are combined in server.py to produce a "combined_prob":
  combined_prob >= 0.65 → bet YES
  combined_prob <= 0.35 → bet NO
  between 0.35-0.65    → skip (not enough confidence)

================================================================================
WHAT IS AN API CALL? WHAT IS CACHING?
================================================================================

An API call is a request to a remote server to get data. This file makes calls to:
  - BRTI constituent exchanges (Bitstamp, Coinbase, Gemini, Kraken) for spot prices
  - Binance Futures API for funding rates and mark prices
  - CoinGecko as a fallback if the BRTI exchanges are unavailable

Each API call takes 100-500 milliseconds and the provider may limit how many
you can make per minute ("rate limiting"). To stay fast without being blocked,
this file caches (stores) results for a short time:
  - Prices cached for 5 seconds (fresh enough for 15-minute markets)
  - Funding rates cached for 30 seconds (updates only every 8 hours anyway)
  - Volatility cached for 60 minutes (changes slowly)

Caching works like a refrigerator: instead of going to the store (API) every
time you need milk (price data), you check the fridge (cache) first, and only
go to the store if it's been empty for a while.

================================================================================
WHAT IS A FUNDING RATE?
================================================================================

In perpetual futures trading (trading crypto with leverage that doesn't expire),
"funding" is a periodic payment between buyers and sellers:
  - If more people are betting UP (long) than DOWN (short):
      → Longs pay shorts every 8 hours (positive funding rate)
      → This discourages excessive speculation and keeps futures price near spot
  - If more people are betting DOWN (short) than UP (long):
      → Shorts pay longs every 8 hours (negative funding rate)

A HIGH positive funding rate means the market is overcrowded with leveraged bets
going UP. This is a CONTRARIAN signal — overcrowded trades tend to reverse when
forced liquidations happen. So high positive funding → slight bearish bias.
A HIGH negative funding rate → slight bullish bias (short squeeze potential).

================================================================================
WHAT IS VOLATILITY? WHY DOES IT MATTER?
================================================================================

Volatility measures how much a price typically moves around. High volatility means
big swings; low volatility means stable prices.

For prediction markets, volatility determines UNCERTAINTY. If Bitcoin has very
high volatility and 10 minutes remain before a market closes, the price could
still swing either way — you're less certain of the outcome. If volatility is
low, the current price is likely to stay close to where it is.

This module computes "realized volatility" using recent actual price data (not
guesses). The HAR-RV method (Heterogeneous Autoregressive Realized Volatility)
blends 1-day, 3-day, and 5-day volatility estimates to adapt to the current
market regime.

================================================================================
WHAT IS THE HURST EXPONENT?
================================================================================

The Hurst exponent (H) measures whether price movements are "persistent" or
"mean-reverting":
  - H > 0.55: Trending — if price went up recently, it tends to keep going up
  - H < 0.45: Mean-reverting — if price went up, it tends to come back down
  - H ≈ 0.50: Random walk — past moves don't predict future moves

This tells the bot whether to use momentum strategies or contrarian strategies.

================================================================================
WHAT IS THE KELLY CRITERION?
================================================================================

Kelly criterion is a formula that tells you the OPTIMAL fraction of your bankroll
to bet. It was invented by John Kelly at Bell Labs in 1956, originally for
betting on horse races. The core insight: bet more when your edge is large,
bet less (or nothing) when your edge is small or negative.

The formula: f* = (p × b - q) / b
  where p = probability of winning, q = 1-p, b = payout odds

This module implements "quarter-Kelly" — betting only 25% of what the formula
suggests. This reduces the bet size but also reduces variance (risk), which is
important because our probability estimates are never perfectly accurate.

================================================================================
ARCHITECTURE: WHERE THIS FILE FITS
================================================================================

This module is one of the core "signal engines" powering the bot's crypto trading
strategy. The bot trades Kalshi's 15-minute crypto markets (BTC, ETH, XRP), which
are binary contracts like "Will Bitcoin be above $87,500.00 at 9:15 AM ET?"

These markets resolve using the CF Benchmarks Bitcoin Real-Time Index (BRTI), which
calculates a 60-second time-weighted average price (TWAP) during the final minute
before each 15-minute window closes. This means the settlement price is NOT a single
snapshot — it is an average over that last minute, which smooths out manipulation
and flash wicks. We compare our live CoinGecko price against the market's target
price to estimate whether the BRTI settlement will land above or below the strike.

This module provides 4 quantitative signals that get combined in server.py:
  1. Live price vs. target price  (compute_live_price_signal)  — strongest signal
  2. Time-of-day bias            (compute_time_of_day_bias)   — small historical tilt
  3. Funding rate bias           (compute_funding_rate_bias)   — leverage sentiment
  4. Kelly criterion sizing      (kelly_criterion_size)        — how much to bet

The signals are combined in server.py like this:
  combined_prob = market_consensus + price_signal + tod_bias + funding_bias + imbalance
  (clamped to 0.01-0.99)

If combined_prob >= 0.65 -> buy YES; if <= 0.35 -> buy NO; otherwise skip.
Position size is then determined by quarter-Kelly criterion.

DATA SOURCES:
  - CoinGecko (free, no API key): Live spot prices for BTC, ETH, XRP
  - Binance Futures API (free, no API key): BTC/USDT perpetual funding rate
  Both are cached for 30 seconds to respect rate limits and reduce latency.

WHY COINGECKO AND NOT BRTI DIRECTLY?
  CF Benchmarks BRTI is not freely accessible via API. CoinGecko aggregates prices
  from many exchanges and closely tracks the BRTI settlement price. The small
  discrepancy between CoinGecko spot and BRTI TWAP is acceptable because we only
  need directional confidence (above/below target), not exact price matching.

================================================================================
"""

from __future__ import annotations  # Allows type hints to reference types defined later in the file

import asyncio   # Async I/O for WebSocket background task
import json as _json  # JSON parsing for WebSocket messages
import logging   # Python's built-in logging system (better than print() for production code)
import re        # "Regular expressions" — a powerful way to search for patterns in text
import time      # For getting the current timestamp (used by the cache)
from typing import Optional  # Type hint: Optional[float] means "either a float or None"

import httpx  # Modern HTTP library for making web requests to external APIs

logger = logging.getLogger("predictionbot")  # Named logger — log messages show "predictionbot" as source

# ── Cache state ──────────────────────────────────────────────────────────────
# We use module-level globals for caching because this module is imported once
# and shared across the bot's async loop. The cache prevents hammering free APIs
# on every scan cycle. 10s TTL gives fresh prices while respecting rate limits.
#
# HOW MODULE-LEVEL CACHING WORKS:
# Python loads each module (file) once and reuses it everywhere. These global
# variables persist between function calls — they're like shared memory for this
# entire module. When get_current_prices() is called 100 times in a second, only
# the first call actually hits the network; the rest read from _price_cache.
#
# WHAT IS A TTL?
# TTL = "Time To Live" — how long cached data is considered fresh.
# After the TTL expires, the next call fetches new data and refreshes the cache.
#
# UNIX TIMESTAMP:
# time.time() returns seconds since January 1, 1970 ("Unix epoch").
# Comparing (now - last_fetch_time) < TTL tells us if the cache is still fresh.

_price_cache: dict[str, float] = {}       # Maps coin key -> USD price, e.g. {"bitcoin": 87500.0}
_price_cache_time: float = 0.0            # Unix timestamp of last successful price fetch
_PRICE_CACHE_TTL = 5   # seconds — 5s for faster signal updates (i9 can handle parallel fetches)

_funding_cache: dict[str, float] = {}     # Maps Binance symbol -> funding rate, e.g. {"BTCUSDT": 0.0003}
_funding_cache_time: float = 0.0          # Unix timestamp of last funding rate fetch
_FUNDING_CACHE_TTL = 60  # seconds — funding rates only update every 8 hours; 60s refresh is plenty

_futures_cache: dict[str, dict] = {}      # coin -> {mark_price, index_price, funding_rate} from Binance
_futures_cache_time: float = 0.0
_FUTURES_CACHE_TTL = 30  # seconds — REST fallback TTL; WS updates cache in real-time
_futures_ws_connected: bool = False       # True when Binance futures mark-price WS is live

_rv_cache: dict[str, float] = {}          # coin -> HAR-RV annualized daily volatility estimate (e.g., 0.65 = 65%)
_rv_cache_time: float = 0.0
_RV_CACHE_TTL = 300   # seconds — recompute every 5 minutes (vol regimes can shift on short timescales)

_hurst_cache: dict[str, float] = {}      # coin -> Hurst exponent (0.1 to 0.9; 0.5 = random walk)
_hurst_cache_time: float = 0.0
_HURST_CACHE_TTL = 900  # seconds — recompute every 15 minutes (market regimes are slow-moving)

# ── Binance WebSocket price cache ─────────────────────────────────────────────
# Updated in real-time by _binance_ws_loop(). get_current_prices() prefers
# these over REST polling when fresh (< 3s since last tick). Latency: ~10ms
# vs ~5s for REST polling — critical for exploiting Kalshi market-lag edge.
_ws_prices: dict[str, float] = {}       # coin -> latest trade price from Binance WS
_ws_prices_ts: dict[str, float] = {}    # coin -> Unix timestamp of last WS tick
_ws_connected: bool = False             # True when Binance WS is live

# ── Coin key mapping ─────────────────────────────────────────────────────────
# HOW KALSHI TICKER NAMES WORK:
# Kalshi uses structured ticker names that encode the market type and coin.
# Examples:
#   "KXBTC15M-26MAR230915-B87500"  ← 15-min Bitcoin market, 9:15 AM March 26
#   "KXETH15M-26MAR230930-B2100"   ← 15-min Ethereum market, 9:30 AM March 26
#   "KXBTCD-26MAR2023"             ← Daily Bitcoin directional market
#
# The PREFIX (first 7-9 characters) tells us which coin is being traded.
# This dictionary maps those prefixes to CoinGecko's internal coin ID format.
# CoinGecko uses full names like "bitcoin" and "ethereum" (not "BTC"/"ETH").
#
# Internal coin keys used throughout this module. Ticker prefix → coin key.
_PREFIX_TO_COIN = {
    # 15-minute binary markets
    "KXBTC15M":  "bitcoin",
    "KXETH15M":  "ethereum",
    "KXXRP15M":  "ripple",
    "KXSOL15M":  "solana",
    "KXDOGE15M": "dogecoin",
    "KXBNB15M":  "binancecoin",
    "KXHYPE15M": "hyperliquid",
    "KXADA15M":  "cardano",
    "KXAVAX15M": "avalanche-2",
    # Directional hourly/daily above-below markets (longer prefixes first to avoid short-circuit)
    "KXBTCD":    "bitcoin",
    "KXETHD":    "ethereum",
    "KXXRPD":    "ripple",
    "KXSOLD":    "solana",
    "KXBNBD":    "binancecoin",
    "KXHYPED":   "hyperliquid",
    "KXDOGED":   "dogecoin",
    "KXADAD":    "cardano",
    "KXAVAXD":   "avalanche-2",
    # Range/bracket markets (KXBTC-26MAR2417 etc.)
    "KXBTC-":    "bitcoin",
    "KXETH-":    "ethereum",
    "KXXRP-":    "ripple",
    "KXDOGE-":   "dogecoin",
    "KXSOLE-":   "solana",
    "KXBNB-":    "binancecoin",
    "KXHYPE-":   "hyperliquid",
}
# Keep backward compat alias
_PREFIX_TO_COINGECKO = _PREFIX_TO_COIN

_COIN_TO_SYMBOL = {
    "bitcoin":      "BTC",
    "ethereum":     "ETH",
    "ripple":       "XRP",
    "solana":       "SOL",
    "dogecoin":     "DOGE",
    "binancecoin":  "BNB",
    "hyperliquid":  "HYPE",
    "cardano":      "ADA",
    "avalanche-2":  "AVAX",
}
# Keep backward compat alias
_COINGECKO_TO_SYMBOL = _COIN_TO_SYMBOL

# Binance spot stream symbols (lowercase USDT pairs) for each coin.
# HYPE has no Binance listing — get_current_prices() fills it from REST cache.
_COIN_TO_BINANCE_STREAM: dict[str, str] = {
    "bitcoin":     "btcusdt",
    "ethereum":    "ethusdt",
    "ripple":      "xrpusdt",
    "solana":      "solusdt",
    "dogecoin":    "dogeusdt",
    "binancecoin": "bnbusdt",
    "cardano":     "adausdt",
    "avalanche-2": "avaxusdt",
}
_BINANCE_STREAM_TO_COIN: dict[str, str] = {v: k for k, v in _COIN_TO_BINANCE_STREAM.items()}

# ── Per-coin intraday volatility (% per minute, 1-sigma) ────────────────────
# Calibrated from 15-min OHLCV data. Used in compute_fair_probability() to
# model how far price can drift over the remaining window time.
# Higher vol → wider uncertainty band → need bigger price distance for confidence.
#
# WHAT IS "1-SIGMA VOLATILITY"?
# In statistics, "sigma" (σ) means "standard deviation" — a measure of how spread
# out values are around their average. For price movements, 1-sigma means "the price
# will be within this range about 68% of the time over one time period."
#
# For a 15-minute BTC market: 1-sigma per minute ≈ 0.076%
# After 15 minutes: uncertainty = 0.076% × sqrt(15) ≈ 0.29%
# So if BTC is at $87,500, it could typically move ±$254 in 15 minutes (68% probability).
#
# HOW ANNUAL VOL CONVERTS TO PER-MINUTE VOL:
# Volatility scales with the SQUARE ROOT of time (a key property of random walks).
# If annual vol is 55%, and there are 525,600 minutes per year:
#   per-minute vol = 55% / sqrt(525,600) = 0.55 / 725 ≈ 0.076% = 0.00076
#
# WHY THIS MATTERS FOR TRADING:
# If the live price is close to the target with lots of time remaining, we're
# uncertain — the price could swing either way. If it's far from target with
# little time left, we're more confident. This table helps quantify that.
_COIN_VOL_PER_MIN = {
    # Expressed as a DECIMAL fraction per minute (not percent).
    # Derived from annual vol divided by sqrt(trading minutes per year):
    #   sqrt(365 * 24 * 60) = sqrt(525,600) ≈ 725
    # BTC: 55% annual → 0.55/725 ≈ 0.00076   ETH: 70% → 0.00097   XRP: 85% → 0.00117
    # SOL: 110% annual → 1.10/725 ≈ 0.00152  DOGE: 130% → 0.00179
    # BNB:  75% annual → 0.75/725 ≈ 0.00103  HYPE: 180% → 0.00248
    # ADA: 100% annual → 1.00/725 ≈ 0.00138  AVAX: 115% → 0.00159
    "bitcoin":     0.00076,
    "ethereum":    0.00097,
    "ripple":      0.00117,
    "solana":      0.00152,
    "dogecoin":    0.00179,
    "binancecoin": 0.00103,
    "hyperliquid": 0.00248,
    "cardano":     0.00138,
    "avalanche-2": 0.00159,
}

# ── BRTI constituent exchange fetch functions ────────────────────────────────
# WHAT IS BRTI?
# CF Benchmarks Bitcoin Real-Time Index (BRTI) is the official price index that
# Kalshi uses to settle its 15-minute Bitcoin markets. It's calculated as a
# 60-second volume-weighted average price (VWAP) from four major exchanges:
# Bitstamp, Coinbase, Gemini, and Kraken.
#
# WHY FETCH FROM THESE SPECIFIC EXCHANGES?
# Because that's EXACTLY what Kalshi uses for settlement. If we fetched prices
# from other exchanges (Binance, OKX, etc.), we'd get slightly different numbers
# that don't perfectly predict how Kalshi will settle.
#
# WHAT IS PARALLEL FETCHING?
# Instead of fetching from Bitstamp, then waiting for it to finish, then fetching
# from Coinbase, etc. (sequential = slow), we start all four requests simultaneously
# and collect whichever ones finish. This cuts total fetch time from ~2 seconds
# down to ~0.5 seconds (just the slowest individual exchange).
#
# This is done using Python's concurrent.futures.ThreadPoolExecutor — a way to
# run multiple tasks at the same time in separate "threads" (lightweight workers).
#
# CF Benchmarks BRTI uses: Bitstamp, Coinbase, Gemini, Kraken.
# We fetch from all four in parallel and average the results, giving us the
# closest possible proxy to the actual BRTI settlement price. This is a strict
# improvement over CoinGecko, which aggregates hundreds of exchanges including
# illiquid ones that don't affect BRTI.

_EXCHANGE_PAIRS = {
    # coin key -> (bitstamp_pair, kraken_pair, gemini_pair, coinbase_pair)
    # None entries = exchange doesn't list this coin; fetcher skips gracefully
    "bitcoin":     ("btcusd",  "XBTUSD",  "btcusd",  "BTC-USD"),
    "ethereum":    ("ethusd",  "ETHUSD",  "ethusd",  "ETH-USD"),
    "ripple":      ("xrpusd",  "XRPUSD",  "xrpusd",  "XRP-USD"),
    "solana":      (None,      "SOLUSD",  "solusd",  "SOL-USD"),
    "dogecoin":    (None,      "DOGEUSD", None,      "DOGE-USD"),
    "binancecoin": (None,      None,      None,      "BNB-USD"),
    "hyperliquid": (None,      None,      None,      "HYPE-USD"),
    "cardano":     (None,      "ADAUSD",  None,      "ADA-USD"),
    "avalanche-2": (None,      "AVAXUSD", None,      "AVAX-USD"),
}


def _fetch_bitstamp(pair: str) -> Optional[float]:
    """Fetch the latest trade price from Bitstamp for a given trading pair.

    Args:
        pair: Bitstamp pair format, e.g. "btcusd" (lowercase, no separator)

    Returns:
        The last traded price as a float, or None if the request fails.
        "last" in Bitstamp's API = price of the most recent completed trade.
    """
    try:
        r = httpx.get(f"https://www.bitstamp.net/api/v2/ticker/{pair}/", timeout=3.0)
        r.raise_for_status()
        # r.json() parses the JSON response body into a Python dict
        # ["last"] extracts the last-trade price (as a string like "87500.00")
        # float() converts the string to a Python floating-point number
        return float(r.json()["last"])
    except Exception:
        return None  # Return None on any error so callers can handle gracefully


def _fetch_kraken(pair: str) -> Optional[float]:
    """Fetch the latest price from Kraken for a given pair.

    Kraken's API wraps results in a "result" object keyed by their internal
    symbol format. "c" field = [last_price, lot_volume]. We want c[0].

    Args:
        pair: Kraken pair format, e.g. "XBTUSD" (note: BTC is "XBT" on Kraken)

    Returns:
        The last traded price as a float, or None on failure.
    """
    try:
        r = httpx.get(f"https://api.kraken.com/0/public/Ticker?pair={pair}", timeout=3.0)
        r.raise_for_status()
        data = r.json()
        # Kraken returns errors in the "error" list even with a 200 status
        if data.get("error"):
            return None
        result = data.get("result", {})
        # next(iter(result), None) gets the first key from the result dict
        # (Kraken uses its own internal symbol as the key, which may differ from input pair)
        key = next(iter(result), None)
        if not key:
            return None
        # "c" = [close_price, lot_volume]. c[0] is the last trade price.
        return float(result[key]["c"][0])
    except Exception:
        return None


def _fetch_gemini(pair: str) -> Optional[float]:
    """Fetch the latest price from Gemini for a given pair.

    Args:
        pair: Gemini pair format, e.g. "btcusd" (lowercase, no separator)

    Returns:
        The last traded price as a float, or None on failure.
    """
    try:
        r = httpx.get(f"https://api.gemini.com/v1/pubticker/{pair}", timeout=3.0)
        r.raise_for_status()
        return float(r.json()["last"])
    except Exception:
        return None


def _fetch_coinbase(pair: str) -> Optional[float]:
    """Fetch the current spot price from Coinbase for a given pair.

    SPOT PRICE vs. LAST TRADE PRICE:
    Coinbase's /spot endpoint returns the current mid-market price, which is
    slightly different from "last trade price" — it's the midpoint between
    the best bid and ask. For our purposes (detecting direction vs. a target),
    this is fine.

    Args:
        pair: Coinbase pair format, e.g. "BTC-USD" (uppercase with dash)

    Returns:
        The spot price as a float, or None on failure.
    """
    try:
        r = httpx.get(f"https://api.coinbase.com/v2/prices/{pair}/spot", timeout=3.0)
        r.raise_for_status()
        # Coinbase wraps the result: {"data": {"amount": "87500.00", "currency": "USD"}}
        return float(r.json()["data"]["amount"])
    except Exception:
        return None


def get_current_prices() -> dict[str, float]:
    """Fetch BTC, ETH, XRP spot prices from BRTI constituent exchanges.

    CF Benchmarks BRTI (the index Kalshi uses for 15-min crypto settlement) is
    computed from: Bitstamp, Coinbase, Gemini, Kraken. We fetch from all four
    in parallel and average the results — this is the closest freely-available
    proxy to the actual settlement price.

    Falls back to CoinGecko if fewer than 2 constituent exchanges respond.

    Caching: 10-second TTL for near-realtime accuracy on 15-min markets.

    Returns:
        Dict mapping coin key to USD price: {"bitcoin": 87500.0, ...}
    """
    # `global` tells Python we want to modify the module-level variables, not create local copies.
    # Without `global`, writing to _price_cache would create a new local variable.
    global _price_cache, _price_cache_time

    now = time.time()

    # ── 1. Prefer Binance WebSocket prices if WS is connected ───────────────────
    # WS is event-driven: latest price in buffer IS the current price — no age
    # check needed or wanted. Only gate on _ws_connected so a dead socket falls
    # through to REST. HYPE has no Binance listing so it always comes from REST.
    if _ws_connected and _ws_prices and len(_ws_prices) >= 4:
        return {**_price_cache, **_ws_prices}  # WS wins, REST fills gaps (HYPE)

    # ── 2. Return REST cache if still fresh ────────────────────────────────────
    if _price_cache and (now - _price_cache_time) < _PRICE_CACHE_TTL:
        return _price_cache

    # concurrent.futures allows running multiple tasks simultaneously in background threads.
    # We import it here (not at the top) since this function is called frequently and the
    # import is cached by Python after the first call anyway.
    import concurrent.futures

    prices: dict[str, float] = {}

    def _fetch_coin(coin: str) -> tuple[str, Optional[float]]:
        """Fetch price for one coin from all BRTI constituent exchanges in parallel.

        For each coin, we know which exchanges list it (from _EXCHANGE_PAIRS).
        We submit all applicable fetch functions to a thread pool simultaneously.
        Whichever exchanges respond first get included in the average.

        Returns: (coin_name, average_price) or (coin_name, None) if not enough sources responded.
        """
        pairs = _EXCHANGE_PAIRS.get(coin)
        if not pairs:
            return coin, None
        bs_pair, kr_pair, gm_pair, cb_pair = pairs
        # Build a list of (name, fetch_function) tuples, skipping exchanges where pair is None
        # (e.g., Bitstamp doesn't list DOGE, so bs_pair=None → skip Bitstamp for DOGE)
        # The lambda p=bs_pair: ... pattern captures the pair value at loop time (avoids closure bug)
        fetchers = [
            (name, fn) for name, fn, pair in [
                ("bitstamp", lambda p=bs_pair: _fetch_bitstamp(p), bs_pair),
                ("kraken",   lambda p=kr_pair: _fetch_kraken(p),   kr_pair),
                ("gemini",   lambda p=gm_pair: _fetch_gemini(p),   gm_pair),
                ("coinbase", lambda p=cb_pair: _fetch_coinbase(p), cb_pair),
            ] if pair is not None  # Only include exchanges that list this coin
        ]
        if not fetchers:
            return coin, None
        results = []
        # ThreadPoolExecutor runs all fetchers simultaneously in background threads.
        # max_workers=4 means up to 4 fetch functions run at once.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            # ex.submit(fn) starts fn running in the background, returns a "Future" object
            # A Future is a placeholder — it will hold the result once the function finishes
            futures = {ex.submit(fn): name for name, fn in fetchers}
            # as_completed() yields futures as they finish (fastest first)
            # timeout=4.0 means we give up waiting after 4 seconds
            for fut in concurrent.futures.as_completed(futures, timeout=4.0):
                val = fut.result()
                if val and val > 0:
                    results.append(val)
        # Require at least 2 sources for major coins (BTC/ETH/XRP); 1 is ok for alt coins with fewer listings
        min_sources = 1 if len(fetchers) <= 2 else 2
        if len(results) >= min_sources:
            # Average the prices from all responding exchanges = our BRTI proxy
            avg = sum(results) / len(results)
            logger.debug(
                f"[crypto_feed] {_COIN_TO_SYMBOL.get(coin, coin)} "
                f"BRTI avg={avg:,.4f} from {len(results)}/{len(fetchers)} exchanges"
            )
            return coin, avg
        return coin, None  # Not enough exchanges responded — skip this coin

    try:
        _all_coins = list(_EXCHANGE_PAIRS.keys())
        # Outer thread pool: fetch all coins simultaneously (9 coins × 4 exchanges = 36 concurrent requests)
        # This is the "parallel parallel" pattern — both the coin loop and exchange loop are parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(_all_coins)) as ex:
            coin_futures = {ex.submit(_fetch_coin, c): c for c in _all_coins}
            for fut in concurrent.futures.as_completed(coin_futures, timeout=6.0):
                coin, val = fut.result()
                if val:
                    prices[coin] = val
    except Exception as e:
        logger.warning(f"[crypto_feed] BRTI exchange fetch error: {e}")

    # If we got prices from at least 2 coins, consider the fetch successful
    if len(prices) >= 2:
        _price_cache = prices
        _price_cache_time = now
        return _price_cache

    # ── Fallback: CoinGecko (all tracked coins) ───────────────────────────────
    # If fewer than 2 BRTI exchanges responded (network issues, geo-blocking, etc.),
    # fall back to CoinGecko which aggregates hundreds of exchanges.
    # CoinGecko is less precise than BRTI constituent exchanges but widely available.
    logger.warning("[crypto_feed] Fewer than 2 BRTI exchanges responded — falling back to CoinGecko")
    _all_coin_ids = list(_EXCHANGE_PAIRS.keys())  # all 9 coins, not just BTC/ETH/XRP
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        # CoinGecko's /simple/price endpoint returns prices for multiple coins in one request.
        # params dict becomes URL query string: ?ids=bitcoin,ethereum,...&vs_currencies=usd
        resp = httpx.get(
            url,
            params={"ids": ",".join(_all_coin_ids), "vs_currencies": "usd"},
            timeout=5.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # CoinGecko response: {"bitcoin": {"usd": 87500.0}, "ethereum": {"usd": 2100.0}, ...}
        fallback: dict[str, float] = {}
        for coin_id in _all_coin_ids:
            if coin_id in data and "usd" in data[coin_id]:
                fallback[coin_id] = float(data[coin_id]["usd"])
        if fallback:
            _price_cache = fallback
            _price_cache_time = now
    except Exception as e:
        logger.warning(f"[crypto_feed] CoinGecko fallback also failed: {e}")

    return _price_cache  # Return whatever we have (might be stale if both sources failed)


_PRICE_STALE_THRESHOLD = 30  # seconds — reject prices older than this when WS is down

def get_price_cache_age() -> float:
    """Return seconds since last successful REST price fetch.
    Returns 0.0 if Binance WebSocket is connected (WS prices are always fresh).
    Returns 999.0 if no price fetch has ever succeeded.
    """
    if _ws_connected and _ws_prices and len(_ws_prices) >= 4:
        return 0.0  # WS is live — prices are real-time, no staleness concern
    if _price_cache_time == 0.0:
        return 999.0  # Never fetched
    return time.time() - _price_cache_time


# ── Binance WebSocket price feed ─────────────────────────────────────────────

async def _binance_ws_loop() -> None:
    """Maintain a persistent Binance aggTrade WebSocket for real-time spot prices.

    Subscribes to aggTrade streams for all tracked coins. Each trade event
    carries the execution price, which we store in _ws_prices. Latency from
    a Binance trade to _ws_prices update is ~10ms vs ~5s for REST polling.

    Auto-reconnects with 5s back-off on any error (disconnect, parse failure, etc.).
    The combined-stream endpoint lets us receive all coins over one connection.
    """
    global _ws_connected, _ws_prices, _ws_prices_ts
    import websockets  # type: ignore

    streams = "/".join(f"{sym}@aggTrade" for sym in _COIN_TO_BINANCE_STREAM.values())
    url = f"wss://stream.binance.us:9443/stream?streams={streams}"

    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                compression=None,   # disable per-message deflate — tiny messages, zero compression benefit, pure latency
            ) as ws:
                _ws_connected = True
                logger.info("[binance_ws] Connected — real-time price ticks active")
                async for raw in ws:
                    try:
                        msg = _json.loads(raw)
                        data = msg.get("data", {})
                        symbol = (data.get("s") or "").lower()
                        price_str = data.get("p")
                        if symbol and price_str:
                            coin = _BINANCE_STREAM_TO_COIN.get(symbol)
                            if coin:
                                _ws_prices[coin] = float(price_str)
                                _ws_prices_ts[coin] = time.time()
                    except Exception:
                        pass
        except Exception as e:
            _ws_connected = False
            logger.warning(f"[binance_ws] Disconnected ({e}) — reconnecting in 5s")
            await asyncio.sleep(5)


async def _binance_futures_ws_loop() -> None:
    """Maintain a persistent Binance futures markPrice WebSocket for real-time mark prices.

    Subscribes to @markPrice@1s streams for all tracked perpetuals. Updates _futures_cache
    in real-time (~1s latency) so the spot/mark blend in compute_fair_probability() uses
    fresh mark prices instead of REST-polled data (5s cache).

    Auto-reconnects with 5s back-off on any error.
    """
    global _futures_ws_connected, _futures_cache, _futures_cache_time
    import websockets  # type: ignore

    # Coins with Kalshi markets + Binance perpetuals (ADA/AVAX/HYPE excluded — no Kalshi market or no perp)
    _FUTURES_STREAM_MAP: dict[str, str] = {
        "btcusdt":  "bitcoin",
        "ethusdt":  "ethereum",
        "xrpusdt":  "ripple",
        "solusdt":  "solana",
        "dogeusdt": "dogecoin",
        "bnbusdt":  "binancecoin",
    }
    streams = "/".join(f"{sym}@markPrice@1s" for sym in _FUTURES_STREAM_MAP)
    url = f"wss://fstream.binance.com/stream?streams={streams}"

    while True:
        try:
            import websockets as _ws
            async with _ws.connect(
                url,
                ping_interval=20,
                ping_timeout=10,
                compression=None,   # disable deflate — tiny messages, no benefit
            ) as ws:
                _futures_ws_connected = True
                logger.info("[futures_ws] Connected — real-time mark price ticks active")
                async for raw in ws:
                    try:
                        msg = _json.loads(raw)
                        data = msg.get("data", {})
                        symbol = (data.get("s") or "").lower()
                        coin = _FUTURES_STREAM_MAP.get(symbol)
                        if coin:
                            mark  = float(data.get("p") or 0)
                            index = float(data.get("i") or 0)
                            rate  = float(data.get("r") or 0)
                            if mark > 0:
                                existing = _futures_cache.get(coin, {})
                                _futures_cache[coin] = {
                                    "mark_price":   mark,
                                    "index_price":  index if index > 0 else existing.get("index_price", 0),
                                    "funding_rate": rate,
                                }
                    except Exception:
                        pass
        except Exception as e:
            _futures_ws_connected = False
            logger.warning(f"[futures_ws] Disconnected ({e}) — reconnecting in 5s")
            await asyncio.sleep(5)


async def start_price_websocket() -> None:
    """Launch the Binance WebSocket price feed as a persistent background task.

    Call once from server lifespan. The task runs forever and auto-reconnects.
    Prices become available in get_current_prices() within one trade tick (~ms).
    """
    asyncio.create_task(_binance_ws_loop())
    asyncio.create_task(_binance_futures_ws_loop())
    logger.info("[binance_ws] Price + futures mark-price WebSocket tasks created")


def get_futures_data() -> dict[str, dict]:
    """Fetch BTC/ETH/XRP perpetual futures mark price, index price, and funding rate
    from Binance in a single free API call.

    The Binance mark price is the fair value their risk engine uses for liquidations.
    It's computed from a composite of spot exchanges + funding adjustments, making it
    a better real-time reference than CoinGecko for pricing longer-horizon markets
    (KXBTCD/KXETHD/KXXRPD).

    mark_price  = perpetual futures fair value (blends spot + funding premium)
    index_price = Binance's composite spot index (similar to our BRTI multi-exchange avg)
    funding_rate = current 8h funding rate (positive = longs pay, negative = shorts pay)

    Returns:
        {
            "bitcoin":  {"mark_price": 69500.0, "index_price": 69480.0, "funding_rate": 0.0003},
            "ethereum": {"mark_price": 2100.0,  "index_price": 2098.5,  "funding_rate": 0.0002},
            "ripple":   {"mark_price": 1.42,    "index_price": 1.419,   "funding_rate": 0.0001},
        }
        Returns cached data on error; empty dict if never successfully fetched.
    """
    global _futures_cache, _futures_cache_time

    now = time.time()
    # WS is live — cache is updated every 1s in real-time, return immediately
    if _futures_ws_connected and _futures_cache:
        return _futures_cache
    # WS not connected — fall back to TTL-cached REST data
    if _futures_cache and (now - _futures_cache_time) < _FUTURES_CACHE_TTL:
        return _futures_cache

    # Binance uses "USDT" settlement perpetuals (e.g., BTCUSDT = Bitcoin priced in Tether USD).
    # "USDT" = USD Tether, a stablecoin pegged 1:1 to the US dollar.
    # HYPE has no Binance perpetual contract (Hyperliquid is too new), so it's excluded.
    _COIN_TO_BINANCE = {
        "bitcoin":     "BTCUSDT",
        "ethereum":    "ETHUSDT",
        "ripple":      "XRPUSDT",
        "solana":      "SOLUSDT",
        "binancecoin": "BNBUSDT",
        "dogecoin":    "DOGEUSDT",
        # ADA/AVAX: no Kalshi market; HYPE: no Binance perp — all omitted intentionally
    }

    try:
        # Fetching /fapi/v1/premiumIndex with NO symbol parameter returns ALL perpetual
        # contracts at once in one API call — more efficient than one call per coin.
        # "fapi" = futures API (vs "api" = spot API)
        resp = httpx.get("https://fapi.binance.com/fapi/v1/premiumIndex", timeout=5.0)
        resp.raise_for_status()
        # Build a lookup dictionary: {"BTCUSDT": {...}, "ETHUSDT": {...}, ...}
        # This makes it O(1) to look up any symbol instead of scanning the whole list
        by_symbol = {item["symbol"]: item for item in resp.json() if isinstance(item, dict)}

        result: dict[str, dict] = {}
        for coin, sym in _COIN_TO_BINANCE.items():
            if sym in by_symbol:
                item = by_symbol[sym]
                mark  = float(item.get("markPrice",      0) or 0)   # Futures fair value price
                index = float(item.get("indexPrice",     0) or 0)   # Spot composite index price
                rate  = float(item.get("lastFundingRate", 0) or 0)  # Most recent 8h funding rate
                if mark > 0:
                    result[coin] = {"mark_price": mark, "index_price": index, "funding_rate": rate}
                    logger.debug(
                        f"[futures] {_COIN_TO_SYMBOL.get(coin, coin)} "
                        f"mark={mark:,.4f} index={index:,.4f} funding={rate:.6f}"
                    )

        if result:
            _futures_cache = result
            _futures_cache_time = now
            return _futures_cache
    except Exception as e:
        err_str = str(e)
        # HTTP 451 = "Unavailable For Legal Reasons" (geo-blocking — Binance blocks some regions)
        # HTTP 403 = Forbidden (similar geo-restriction enforcement)
        # Don't log these as warnings — they happen every call in blocked regions and spam the log
        if "451" in err_str or "403" in err_str:
            logger.debug(f"[futures] Binance geo-blocked: {e}")
        else:
            logger.debug(f"[futures] Binance futures fetch failed: {e}")

    return _futures_cache  # Return stale cache on error — better than nothing


def get_realized_volatility() -> dict[str, float]:
    """Compute HAR-RV daily volatility estimate for BTC/ETH/XRP from Binance 5-min klines.

    WHY HAR-RV OVER FIXED ANNUAL VOL:
    Fixed annual vol (e.g., 55% for BTC) ignores volatility regimes. After a news
    shock or ETF-related volume spike, actual vol can be 2–3× higher than the
    long-run average. Using a stale vol estimate causes systematic mispricing in
    the Student-t model for KXBTCD/KXETHD/KXXRPD daily markets.

    HAR (Heterogeneous Autoregressive) Realized Volatility uses intraday price data
    to forecast next-day vol: σ_day = 0.6×RV_1day + 0.2×RV_3day + 0.2×RV_5day
    where RV = annualized realized vol computed from 5-minute bar returns.

    Returns: dict coin_id -> annualized daily vol estimate (e.g., 0.65 = 65%)
    Falls back to fixed annual vol constants if Binance API fails.
    """
    import math
    import time as _time  # Renamed to _time to avoid conflict with the module-level `time` import

    global _rv_cache, _rv_cache_time
    now = _time.time()
    # Return cached volatility estimates if they're still fresh (1-hour TTL)
    if _rv_cache and (now - _rv_cache_time) < _RV_CACHE_TTL:
        return _rv_cache

    # Fixed fallback vols (annual %) — used if Binance API fails.
    # These are long-run historical averages; they'll be wrong during extreme vol regimes
    # but are better than nothing when real-time data is unavailable.
    _fallback = {
        "bitcoin": 0.55, "ethereum": 0.70, "ripple": 0.85,
        "solana": 1.10, "dogecoin": 1.30, "binancecoin": 0.75,
        "hyperliquid": 1.80, "cardano": 1.00, "avalanche-2": 1.15,
    }

    # Binance spot API symbols for each coin (all 9 coins tracked on Kalshi).
    # Note: we use the spot API (api.binance.com) here, not the futures API (fapi.binance.com),
    # because spot klines are available globally (futures API is geo-blocked in some regions).
    _symbols = {
        "bitcoin":     "BTCUSDT",
        "ethereum":    "ETHUSDT",
        "ripple":      "XRPUSDT",
        "solana":      "SOLUSDT",
        "dogecoin":    "DOGEUSDT",
        "binancecoin": "BNBUSDT",
        "hyperliquid": "HYPEUSDT",
        "cardano":     "ADAUSDT",
        "avalanche-2": "AVAXUSDT",
    }

    result: dict[str, float] = {}
    try:
        # urllib.request is Python's built-in HTTP library — used here to avoid httpx overhead.
        # json is imported as _json to avoid shadowing the global `json` name if it existed.
        import urllib.request, json as _json

        for coin_id, symbol in _symbols.items():
            try:
                # WHAT ARE KLINES?
                # "Klines" = candlestick bars. Each 5-minute kline contains:
                # [open_time, open, high, low, close, volume, close_time, ...]
                # We only need the close price (index 4) to compute returns.
                #
                # limit=288 fetches the last 288 bars × 5 minutes = 1,440 minutes = 24 hours.
                # Binance rate limit: 1200 weight/min; klines endpoint = 2 weight. Fine here.
                _url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={symbol}&interval=5m&limit=288"
                )
                with urllib.request.urlopen(_url, timeout=5) as resp:
                    klines = _json.loads(resp.read())

                if len(klines) < 50:
                    # Not enough data to compute meaningful volatility — use fallback
                    result[coin_id] = _fallback[coin_id]
                    continue

                # Extract closing prices from each kline.
                # k[4] = close price (5th element, 0-indexed). float() converts string to number.
                closes = [float(k[4]) for k in klines]

                # WHAT ARE LOG RETURNS?
                # A "return" is how much the price changed. Instead of percent change
                # (price2 - price1) / price1, we use LOG returns: ln(price2 / price1).
                # Log returns are better for financial math because they're additive across
                # time periods and handle compounding correctly.
                # Example: price goes 100 → 102 → 101 → 103
                #   log return 1 = ln(102/100) = 0.0198 (+1.98%)
                #   log return 2 = ln(101/102) = -0.0099 (-0.99%)
                #   log return 3 = ln(103/101) = +0.0197 (+1.97%)
                log_returns = [
                    math.log(closes[i] / closes[i - 1])
                    for i in range(1, len(closes))
                ]

                # REALIZED VARIANCE FORMULA:
                # RV = sum of squared log returns × annualization factor
                # Why squared? Because variance = average of squared deviations from mean.
                # (For short intervals, the mean is ~0, so we just sum squared returns.)
                # Why annualize? To express vol on a per-year scale (standard convention).
                # 252 trading days/year × 288 five-minute bars/day = 72,576 bars/year
                _annualize = 252.0 * 288.0
                n = len(log_returns)

                # HAR-RV uses 3 different time horizons to capture short, medium, long volatility.
                # The idea: volatility is "heterogeneous" — short-term bursts (traders),
                # medium-term swings (institutions), long-term trends (fundamentals).
                # Each component captures a different type of market participant.

                # RV over last 1 day (last 287 bars ≈ 1 full day of 5-min bars)
                # math.sqrt converts variance to standard deviation (volatility)
                rv_1d = math.sqrt(sum(r**2 for r in log_returns[-287:]) * _annualize)
                # RV over last 3 days (last 863 bars — but we only have 287, use what we have)
                rv_3d = math.sqrt(sum(r**2 for r in log_returns[-min(863, n):]) * _annualize)
                # RV over all 5 days (all available bars)
                rv_5d = math.sqrt(sum(r**2 for r in log_returns) * _annualize)

                # HAR-RV weighted average: 60% weight on most recent day (most predictive),
                # 20% on medium horizon, 20% on slow/long horizon.
                har_rv = 0.60 * rv_1d + 0.20 * rv_3d + 0.20 * rv_5d

                # Sanity clamp: crypto daily vol rarely below 20% (stablecoin territory)
                # or above 300% (even DOGE in a mania doesn't sustain that).
                result[coin_id] = max(0.20, min(3.00, har_rv))

            except Exception:
                # If any individual coin fails, use its historical fallback
                result[coin_id] = _fallback[coin_id]

        # Store result in cache. If result is empty (all coins failed), use fallback dict.
        _rv_cache = result if result else _fallback.copy()
        _rv_cache_time = now
        return _rv_cache

    except Exception:
        # Outer exception: something very wrong (e.g., urllib not available). Use all fallbacks.
        return _fallback.copy()


def get_funding_rates() -> dict[str, float]:
    """Fetch the most recent BTC perpetual funding rate from Binance Futures.

    WHAT IS FUNDING RATE?
    On perpetual futures exchanges, longs pay shorts (or vice versa) every 8 hours
    to keep the futures price anchored to spot. The funding rate reflects this:
      - Positive funding (e.g., +0.03%) = longs pay shorts = market is net long
      - Negative funding (e.g., -0.03%) = shorts pay longs = market is net short

    WHY THIS MATTERS FOR TRADING:
    Extreme funding rates signal overleveraged positioning. When too many traders
    are long (high positive funding), the market is vulnerable to a liquidation
    cascade downward — and vice versa. We use this as a contrarian indicator:
      - High positive funding -> slight bearish bias (overleveraged longs)
      - High negative funding -> slight bullish bias (overleveraged shorts)

    This is Signal #4 in the trading system, used by compute_funding_rate_bias().

    Returns:
        Dict mapping symbol to funding rate as a decimal, e.g.:
        {"BTCUSDT": 0.0003}  (0.03%)
        Returns cached values on error; returns empty dict if never fetched.
    """
    global _funding_cache, _funding_cache_time

    now = time.time()
    # Return cached rate if still fresh (within 30-second TTL).
    # Funding rates only change every 8 hours, so even 30s is very fresh.
    if _funding_cache and (now - _funding_cache_time) < _FUNDING_CACHE_TTL:
        return _funding_cache

    try:
        # Binance Futures API — returns the most recent funding rate for BTCUSDT.
        # Funding rates update every 8 hours (00:00, 08:00, 16:00 UTC).
        # We only fetch BTC because Kalshi's crypto markets are BTC-dominated
        # and BTC funding sentiment broadly affects ETH/XRP direction too.
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": "BTCUSDT", "limit": 1}  # limit=1 = only the most recent rate
        resp = httpx.get(url, params=params, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        # Response format: [{"symbol": "BTCUSDT", "fundingRate": "0.00030000", ...}]
        # It's a list because you can fetch rate history; limit=1 gives just the latest.

        rates = {}
        if data and isinstance(data, list) and len(data) > 0:
            # data[0] = first (and only, since limit=1) element
            # .get("fundingRate", 0) safely handles missing field
            rate = float(data[0].get("fundingRate", 0))
            rates["BTCUSDT"] = rate
            logger.debug(f"[crypto_feed] Funding rate BTCUSDT: {rate:.6f}")

        # Only update cache if we got a valid rate (don't cache empty result)
        if rates:
            _funding_cache = rates
            _funding_cache_time = now
        return _funding_cache

    except Exception as e:
        # Binance geo-blocks some regions (HTTP 451 = "Unavailable For Legal Reasons").
        # This is a known hard failure — don't spam WARNING logs every 30s. Debug only.
        err_str = str(e)
        if "451" in err_str or "403" in err_str or "Unavailable" in err_str:
            logger.debug(f"[crypto_feed] Binance funding rate geo-blocked: {e}")
        else:
            logger.warning(f"[crypto_feed] Binance funding rate fetch failed: {e}")
        return _funding_cache  # Return stale cache on error — better than no signal


def parse_target_price_from_title(title: str, subtitle: str = "") -> Optional[float]:
    """Extract the target (strike) price from a Kalshi crypto market title.

    Kalshi 15-min crypto markets phrase their question as a binary yes/no, e.g.:
      "Bitcoin above $87,500.00?"  ->  target = 87500.00
      "Ethereum above $2,150.00?"  ->  target = 2150.00
      "XRP above $0.5500?"         ->  target = 0.55

    The target price is the "strike" — the price level the crypto must be above
    (per the BRTI TWAP at settlement) for the YES contract to pay out.

    We compare the live CoinGecko price to this target to generate Signal #1.
    If live price > target, the market should lean YES; if below, lean NO.

    Falls back to parsing the subtitle if the title doesn't contain a price
    (some Kalshi markets put the price in the subtitle field).

    Returns:
        The target price as a float, or None if parsing fails.
    """
    # Try the title first; if no price found, try the subtitle as fallback.
    for text in (title, subtitle):
        if not text:
            continue
        # WHAT IS A REGEX?
        # A "regular expression" is a pattern for matching text. This pattern:
        #   r'\$([0-9,]+\.?\d*)'
        # Means: find a dollar sign ($) followed by digits (0-9), optional commas,
        # optional decimal point (.), optional more digits. The () captures just the number.
        #
        # Examples of what this matches:
        #   "$87,500.00" → captures "87,500.00"
        #   "$0.5500"    → captures "0.5500"
        #   "$2,150"     → captures "2,150"
        #
        # re.search() finds the FIRST match anywhere in the string.
        # Regex matches dollar amounts like $87,500.00 or $0.5500 or $2,150
        # The comma-separated thousands and optional decimal handle all crypto price formats
        match = re.search(r'\$([0-9,]+\.?\d*)', text)
        if match:
            try:
                # match.group(1) = the captured group (the number after $, without the $)
                # .replace(",", "") removes thousand separators: "87,500.00" → "87500.00"
                # float() converts the string to a number: "87500.00" → 87500.0
                price_str = match.group(1).replace(",", "")
                return float(price_str)
            except (ValueError, TypeError):
                continue  # Move on to the subtitle if title parsing fails
    return None  # No dollar amount found in either title or subtitle


def get_coingecko_id_for_ticker(ticker: str) -> Optional[str]:
    """Map a Kalshi ticker prefix to a CoinGecko coin ID.

    This bridges the gap between Kalshi's ticker naming convention and CoinGecko's
    API. For example, "KXBTC15M-26MAR230915-B87500" starts with "KXBTC15M", which
    maps to "bitcoin" — telling us to fetch Bitcoin's price from CoinGecko.

    WHY NOT JUST USE "BTC" EVERYWHERE?
    Different systems use different names for the same coin:
      - Kalshi uses: "KXBTC15M" (exchange prefix + symbol + timeframe)
      - CoinGecko uses: "bitcoin" (full English name)
      - Trading convention uses: "BTC" (3-letter ticker)
    This function is the translator between Kalshi's world and CoinGecko's world.

    HOW ticker.startswith() WORKS:
    It checks if the ticker begins with the prefix string. Python tries each prefix
    in dictionary order. Because longer prefixes are listed first in _PREFIX_TO_COIN
    (e.g., "KXBTCD" before "KXBTC-"), we correctly match the most specific prefix.

    Args:
        ticker: Full Kalshi ticker, e.g. "KXBTC15M-26MAR230915-15"

    Returns:
        CoinGecko ID like "bitcoin", or None if the ticker isn't a recognized
        crypto market (meaning the bot should skip it — no price signal available).
    """
    for prefix, coin_id in _PREFIX_TO_COINGECKO.items():
        if ticker.startswith(prefix):
            return coin_id
    return None  # Not a recognized crypto market (might be politics, econ, sports, etc.)


def compute_live_price_signal(
    ticker: str,
    title: str,
    subtitle: str = "",
    prices: Optional[dict[str, float]] = None,
    floor_strike: float = 0.0,
) -> float:
    """SIGNAL #1: Compare live crypto price to the market's target price.

    This is the strongest and most important signal in the trading system.
    The logic is straightforward:
      - If BTC is trading at $88,000 and the market asks "Bitcoin above $87,500?",
        the live price is $500 ABOVE the target, so we should lean YES.
      - If BTC is at $87,000 (below target), we should lean NO.

    HOW THE MATH WORKS:
      distance_pct = (live_price - target) / target

    This gives us the percentage distance from the strike. For the example above:
      distance_pct = (88000 - 87500) / 87500 = +0.0057 (0.57% above target)

    The signal is capped at +/-0.15 to prevent a single extreme reading from
    dominating the combined probability. In server.py, this signal gets added
    directly to the market consensus (mid-price / 100) to form the combined
    probability estimate.

    WHY THIS WORKS FOR BRTI RESOLUTION:
    CF Benchmarks BRTI uses a 60-second TWAP for settlement. If the live spot
    price is meaningfully above/below the target with minutes remaining, the TWAP
    is very likely to settle on the same side. The further away from the target,
    the more confident we can be — hence using distance as the signal magnitude.

    Returns a value between -0.15 and +0.15:
      - Positive: live price is ABOVE target (favors YES)
      - Negative: live price is BELOW target (favors NO)
      - 0.0 if data is unavailable (no signal = no adjustment)
    """
    # Use provided prices or fetch fresh ones.
    # server.py passes its cached prices to avoid duplicate API calls within one scan cycle.
    # Passing None here means "fetch fresh prices now" — useful for standalone testing.
    if prices is None:
        prices = get_current_prices()

    # Look up which coin this market is about (bitcoin, ethereum, ripple, etc.)
    coin_id = get_coingecko_id_for_ticker(ticker)
    if not coin_id or coin_id not in prices:
        return 0.0  # Unknown coin or no price data — return neutral signal (no opinion)

    # Determine the target (strike) price.
    # floor_strike comes directly from the Kalshi API and is more reliable.
    # Parsing from the title is a fallback for older market formats.
    # If we can't determine the target, we have no signal.
    target = floor_strike if floor_strike > 0 else parse_target_price_from_title(title, subtitle)
    if target is None or target <= 0:
        return 0.0  # Couldn't determine target — return neutral signal

    live_price = prices[coin_id]
    if live_price <= 0:
        return 0.0  # Invalid price data — return neutral signal

    # THE CORE SIGNAL CALCULATION:
    # distance_pct = how far the live price is from the target, as a FRACTION of the target.
    # This is essentially "how far past the line is the ball?"
    #
    # Positive = live price is ABOVE target → favors YES resolution
    # Negative = live price is BELOW target → favors NO resolution
    #
    # Example: live=$88,000, target=$87,500
    #   distance_pct = (88000 - 87500) / 87500 = +0.00571 (+0.57%)
    #
    # Example: live=$87,000, target=$87,500
    #   distance_pct = (87000 - 87500) / 87500 = -0.00571 (-0.57%)
    distance_pct = (live_price - target) / target

    # Cap the signal at +/-0.15 to avoid over-weighting a single signal.
    # Even if BTC is 5% above target, we cap the signal at 0.15 because the
    # combined probability shouldn't swing too far from market consensus on
    # a single data point. Market makers also see this price data.
    # max(-0.15, min(0.15, x)) clamps x to the range [-0.15, +0.15]
    signal = max(-0.15, min(0.15, distance_pct))

    symbol = _COINGECKO_TO_SYMBOL.get(coin_id, coin_id)
    logger.debug(
        f"[crypto_feed] {symbol} live=${live_price:,.2f} target=${target:,.2f} "
        f"distance={distance_pct:+.4f} signal={signal:+.4f}"
    )
    return signal


def compute_fair_probability(
    ticker: str,
    floor_strike: float,
    secs_left: float,
    prices: Optional[dict[str, float]] = None,
) -> float:
    """Estimate the true probability that this market settles YES.

    METHODOLOGY — Normal distribution model of remaining price movement:

    Kalshi 15-min markets settle on the CF Benchmarks BRTI 60-second TWAP in the
    final minute. Given the live price P and the floor_strike K, we model the
    settlement probability as:

        z = (P - K) / (K * vol_per_min * sqrt(T_remaining_minutes))
        fair_prob = NormalCDF(z)

    Where:
      - (P - K) / K  = current distance from strike as a fraction (e.g., +0.003 = 0.3% above)
      - vol_per_min   = per-coin 1-minute price volatility (calibrated constant)
      - sqrt(T)       = uncertainty scales with square root of remaining time
      - NormalCDF(z)  = probability a standard normal random variable is <= z

    WHY THIS IS NOT GAMBLING:
    This gives a calibrated probability, not a biased guess. The key insight is that
    the current live price IS information — it tells us which side of the floor_strike
    we're on. The further away and the less time remaining, the more certain the outcome.

    Examples (BTC, vol=0.07%/min):
      - Price 0.5% above strike, 3 min left:  z=0.5/(0.07*1.73)=4.1  → P=~99.9%  ✓ bet YES
      - Price 0.3% above strike, 8 min left:  z=0.3/(0.07*2.83)=1.5  → P=~93%    ✓ bet YES
      - Price 0.1% above strike, 10 min left: z=0.1/(0.07*3.16)=0.45 → P=~67%    ✗ skip (too uncertain)
      - Price 0.0% above strike, any time:    z=0                     → P=50%     ✗ never trade

    Returns:
        Probability in [0.0, 1.0] that market settles YES.
        Returns 0.5 (neutral/no signal) if data is unavailable.
    """
    import math
    # scipy.stats.t = the Student's t-distribution (a bell curve with fatter tails than normal)
    # Imported as _student_t to avoid confusion with the built-in `t` variable name
    from scipy.stats import t as _student_t

    if prices is None:
        prices = get_current_prices()

    coin_id = get_coingecko_id_for_ticker(ticker)
    if not coin_id or coin_id not in prices:
        return 0.5  # No coin data = no opinion = 50% (completely uncertain)

    if floor_strike <= 0:
        return 0.5  # No strike price to compare against

    live_price = prices[coin_id]
    if live_price <= 0:
        return 0.5  # Bad price data

    # LOG-DISTANCE FROM STRIKE:
    # We use log(S/K) instead of (S-K)/K because prices follow a log-normal distribution.
    # For small moves they're nearly identical, but log is more accurate for large moves.
    # Example: live=$88,000, strike=$87,500 → ln(88000/87500) = +0.00571
    # Example: live=$87,000, strike=$87,500 → ln(87000/87500) = -0.00574
    price_pct = math.log(live_price / floor_strike)

    # Time remaining in minutes. Use minimum 0.5 min to avoid division-by-zero at expiry.
    # Near settlement, the uncertainty approaches zero and the price signal dominates.
    time_min = max(secs_left / 60.0, 0.5)

    # Per-coin volatility (per-minute 1-sigma).
    # Use HAR-RV realized vol (from get_realized_volatility(), cached hourly)
    # so the model adapts to current volatility regime instead of using stale annual averages.
    #
    # Convert annual vol → per-minute: ann_vol / sqrt(525,600 minutes/year)
    # 525,600 = 365 days × 24 hours × 60 minutes
    _rv = get_realized_volatility()
    _ann_vol = _rv.get(coin_id) or _COIN_VOL_PER_MIN.get(coin_id, 0.0008) * 725.0
    vol_per_min = _ann_vol / 725.0  # 725 ≈ sqrt(525,600)

    # TOTAL UNCERTAINTY OVER REMAINING TIME:
    # The "random walk" property says uncertainty scales with sqrt(time).
    # If per-minute vol is 0.076%, over 9 minutes: 0.076% × sqrt(9) = 0.228%
    # More time remaining → more potential for price to swing → more uncertainty.
    vol_remaining = vol_per_min * math.sqrt(time_min)

    if vol_remaining <= 0:
        # Zero uncertainty (right at settlement) — deterministic based on current price
        return 1.0 if price_pct > 0 else 0.0

    # Z-SCORE: how many standard deviations is the live price above the strike?
    # z > 0 → live price above strike → probability > 50% of settling YES
    # z < 0 → live price below strike → probability < 50% of settling YES
    # z = 0 → live price exactly at strike → probability = exactly 50%
    #
    # Example: live 0.57% above strike, 9 min left:
    #   vol_remaining = 0.00076 × sqrt(9) = 0.00228
    #   z = 0.00571 / 0.00228 = 2.5 → ~99% probability YES
    z = price_pct / vol_remaining

    # WHY STUDENT'S T-DISTRIBUTION INSTEAD OF NORMAL (BELL CURVE)?
    # Crypto prices have "fat tails" — extreme moves happen MORE often than a normal
    # distribution predicts. The Student's t-distribution accounts for this by having
    # heavier tails than a normal distribution.
    #
    # "Degrees of freedom" (df / ν) controls how fat the tails are:
    #   ν = 1   → Cauchy distribution (extremely fat tails)
    #   ν = 3-4 → Typical for crypto (moderately fat tails)
    #   ν → ∞  → Approaches normal distribution (thin tails)
    #
    # BTC has slightly thinner tails than XRP because BTC is more mature/liquid.
    # Lower df = fatter tails = more probability mass in extreme outcomes.
    #
    # Normal CDF (ν→∞) underestimates tail probabilities by 20-40% for |z| > 1.5.
    _df_15m = {
        "bitcoin": 4, "ethereum": 3.5, "ripple": 3,
        "solana": 3.0, "dogecoin": 2.5, "binancecoin": 4.0,
        "hyperliquid": 2.5, "cardano": 3.0, "avalanche-2": 3.0,
    }.get(coin_id, 3.0)

    # CDF = Cumulative Distribution Function.
    # CDF(z) = probability that a random draw from the distribution is ≤ z.
    # For our z-score, this gives P(price remains above strike at settlement).
    prob = float(_student_t.cdf(z, df=_df_15m))

    symbol = _COIN_TO_SYMBOL.get(coin_id, coin_id)
    logger.debug(
        f"[fair_prob] {symbol} live={live_price:,.4f} strike={floor_strike:,.4f} "
        f"dist={price_pct:+.4f} vol={vol_remaining:.4f} z={z:.2f} df={_df_15m} P={prob:.3f} "
        f"T={time_min:.1f}min"
    )
    return prob


def compute_fair_probability_v2(
    ticker: str,
    floor_strike: float,
    secs_left: float,
    prices: Optional[dict[str, float]] = None,
    spread_cents: float = 5.0,
) -> float:
    """GARCH-proxy: spread-adjusted Student-t. Wide bid-ask spread signals elevated
    market-maker uncertainty (high vol regime). Scales volatility estimate by the
    spread relative to a 5¢ baseline — wider spread → higher vol → wider uncertainty
    band → more conservative probability estimate near the strike.

    V2 vs V1: same Student-t CDF core, but vol is multiplied by spread_factor =
    clamp(spread_cents / 5.0, 0.6, 2.0). In quiet markets (spread=3¢) it narrows
    uncertainty (→ higher conviction); in volatile markets (spread=15¢) it widens it.
    """
    import math
    from scipy.stats import t as _student_t

    if prices is None:
        prices = get_current_prices()

    coin_id = get_coingecko_id_for_ticker(ticker)
    if not coin_id or coin_id not in prices:
        return 0.5
    if floor_strike <= 0:
        return 0.5
    live_price = prices[coin_id]
    if live_price <= 0:
        return 0.5

    price_pct = math.log(live_price / floor_strike)
    time_min  = max(secs_left / 60.0, 0.5)

    _rv      = get_realized_volatility()
    _ann_vol = _rv.get(coin_id) or _COIN_VOL_PER_MIN.get(coin_id, 0.0008) * 725.0
    vol_per_min = _ann_vol / 725.0

    # GARCH-proxy: bid-ask spread is a real-time market-maker vol signal.
    # Baseline spread = 5¢ (typical calm 15M crypto market).
    # spread=3¢ → 0.6× (quiet, narrow uncertainty), spread=10¢ → 2.0× (stressed).
    spread_factor = max(0.6, min(2.0, spread_cents / 5.0))
    vol_remaining = vol_per_min * math.sqrt(time_min) * spread_factor

    if vol_remaining <= 0:
        return 1.0 if price_pct > 0 else 0.0

    z = price_pct / vol_remaining
    _df_15m = {
        "bitcoin": 4, "ethereum": 3.5, "ripple": 3,
        "solana": 3.0, "dogecoin": 2.5, "binancecoin": 4.0,
        "hyperliquid": 2.5, "cardano": 3.0, "avalanche-2": 3.0,
    }.get(coin_id, 3.0)
    return float(_student_t.cdf(z, df=_df_15m))


def compute_fair_probability_v3(
    ticker: str,
    floor_strike: float,
    secs_left: float,
    prices: Optional[dict[str, float]] = None,
) -> float:
    """Jump-diffusion (Merton model): accounts for crypto's frequent sudden price jumps.

    Crypto prices jump 3–8× more often than traditional assets. Standard diffusion
    models (V1/V2) ignore jump risk entirely. This uses a Poisson-weighted sum over
    possible jump counts in the remaining window:

        P(S_T > K) = Σ_{n=0}^5 P(N=n) × Φ(z_n)

    where P(N=n) is Poisson(λτ) and z_n = log(S/K) / √(σ²τ + n·σ_j²).

    Each additional jump widens the uncertainty band, which shifts probability mass
    toward 0.5 for near-ATM contracts — making the model more conservative on
    borderline signals and more confident on deeply ITM/OTM contracts.
    """
    import math
    from scipy.stats import norm

    if prices is None:
        prices = get_current_prices()

    coin_id = get_coingecko_id_for_ticker(ticker)
    if not coin_id or coin_id not in prices:
        return 0.5
    if floor_strike <= 0:
        return 0.5
    live_price = prices[coin_id]
    if live_price <= 0:
        return 0.5

    price_pct = math.log(live_price / floor_strike)
    time_min  = max(secs_left / 60.0, 0.5)

    _rv      = get_realized_volatility()
    _ann_vol = _rv.get(coin_id) or _COIN_VOL_PER_MIN.get(coin_id, 0.0008) * 725.0
    vol_per_min = _ann_vol / 725.0

    # Jump intensity (jumps per hour, calibrated to crypto intraday behavior).
    _lambda_hour = {
        "bitcoin": 3.0, "ethereum": 4.0, "ripple": 6.0,
        "dogecoin": 8.0, "solana": 6.0, "binancecoin": 4.0,
        "hyperliquid": 8.0, "cardano": 5.0, "avalanche-2": 5.0,
    }.get(coin_id, 4.0)
    # Typical intraday jump size (std as fraction of price).
    _sigma_j = {
        "bitcoin": 0.006, "ethereum": 0.008, "ripple": 0.012,
        "dogecoin": 0.015, "solana": 0.012, "binancecoin": 0.008,
        "hyperliquid": 0.018, "cardano": 0.010, "avalanche-2": 0.012,
    }.get(coin_id, 0.010)

    lambda_tau   = _lambda_hour * (secs_left / 3600.0)   # E[jumps] in window
    var_diffusion = (vol_per_min ** 2) * time_min

    # Poisson-weighted sum over 0–5 jumps (covers >99.9% of probability mass
    # for λτ < 2, which covers all 15M windows with our calibrated parameters).
    prob_jd = 0.0
    for n in range(6):
        pn = math.exp(-lambda_tau) * (lambda_tau ** n) / math.factorial(n)
        var_n = var_diffusion + n * (_sigma_j ** 2)
        vol_n = math.sqrt(var_n) if var_n > 0 else 1e-8
        z_n   = price_pct / vol_n
        prob_jd += pn * float(norm.cdf(z_n))

    return max(0.01, min(0.99, prob_jd))


def compute_range_probability(
    coin_id: str,
    spot: float,
    bracket_low: float,
    bracket_high: float,
    secs_left: float,
) -> float:
    """Estimate the probability that the settlement price lands inside [bracket_low, bracket_high].

    WHAT IS A RANGE/BRACKET MARKET?
    Instead of "will BTC be ABOVE $87,500?", some Kalshi markets ask:
    "will BTC settle BETWEEN $87,000 and $88,000?"

    These are called "bracket" markets, and they require a different calculation.
    Instead of one boundary (the strike), we have TWO boundaries (the bracket edges).

    HOW THE MATH WORKS:
    Probability of landing INSIDE the bracket = P(price ≤ high) - P(price ≤ low)
    Which in z-score terms:
        P(bracket) = CDF(z_high) - CDF(z_low)

    Think of it as: "the probability of being below the ceiling" minus
    "the probability of being below the floor" = probability of being in the room.

    Used for Kalshi hourly range/bracket markets (KXBTC-*, KXETH-*, etc.).
    Each bracket market asks: will BTC settle between $X and $Y?

    Models: P(in bracket) = CDF(z_high) - CDF(z_low) where z is log-distance
    from bracket bounds scaled by time-remaining volatility.
    Returns 0.5 (no signal) on bad inputs.
    """
    import math
    from scipy.stats import t as _student_t

    if spot <= 0 or bracket_low <= 0 or bracket_high <= bracket_low:
        return 0.5  # Invalid inputs — return neutral probability

    time_min = max(secs_left / 60.0, 0.5)
    _rv = get_realized_volatility()
    _ann_vol = _rv.get(coin_id) or _COIN_VOL_PER_MIN.get(coin_id, 0.0008) * 725.0
    vol_per_min = _ann_vol / 725.0
    vol_remaining = vol_per_min * math.sqrt(time_min)
    if vol_remaining <= 0:
        # Zero time remaining — deterministic based on current price position
        return 1.0 if bracket_low <= spot <= bracket_high else 0.0

    # Per-coin fat-tail degrees of freedom (same as in compute_fair_probability)
    _df = {
        "bitcoin": 4, "ethereum": 3.5, "ripple": 3,
        "solana": 3.0, "dogecoin": 2.5, "binancecoin": 4.0,
        "hyperliquid": 2.5, "cardano": 3.0, "avalanche-2": 3.0,
    }.get(coin_id, 3.0)

    # Z-scores for the bracket boundaries.
    # z_high: how many standard deviations is the bracket ceiling above current price?
    # z_low:  how many standard deviations is the bracket floor above current price?
    # Using log-distance (more accurate than linear for large price moves).
    z_high = math.log(bracket_high / spot) / vol_remaining
    z_low  = math.log(bracket_low  / spot) / vol_remaining
    # P(in bracket) = P(≤ ceiling) - P(≤ floor) = area under the t-curve between the two z-scores
    prob = float(_student_t.cdf(z_high, df=_df)) - float(_student_t.cdf(z_low, df=_df))
    # Clamp to [0.01, 0.99] — probabilities of exactly 0 or 1 are unrealistic and create problems
    return max(0.01, min(0.99, prob))


def compute_time_of_day_bias() -> float:
    """Time-of-day bias — disabled. Returns 0.0 always.

    Hardcoded session biases (+0.05 evening, -0.05 Asian morning) were removed
    because they are not derived from current market data. The live hourly win-rate
    tracker (_hourly_stats) provides a data-driven replacement once 15+ trades have
    settled in a given hour. Until then, 0.0 is used (no opinion from static rules).
    """
    return 0.0


def compute_funding_rate_bias(rates: Optional[dict[str, float]] = None) -> float:
    """SIGNAL #4: Return a contrarian bias based on Binance BTC funding rate.

    HOW FUNDING RATE SIGNALS WORK:
    Perpetual futures funding rates reflect market leverage sentiment:
      - Funding > +0.05% (0.0005): Too many leveraged longs.
        These traders will get liquidated if price drops, causing a cascade.
        We bias slightly toward NO (bearish) as a contrarian play.
      - Funding < -0.05% (-0.0005): Too many leveraged shorts.
        A price uptick can trigger a short squeeze (forced buying).
        We bias slightly toward YES (bullish) as a contrarian play.
      - Funding between -0.05% and +0.05%: Market is balanced, no signal.

    WHY +/-0.03?
    The bias is small (+/-0.03) because funding rate is a slow-moving indicator
    (updates every 8 hours) and isn't directly predictive of 15-minute outcomes.
    It's a background sentiment signal, not a primary driver. Combined with the
    other signals, it can tip a borderline trade into or out of the 65% threshold.

    Args:
        rates: Pre-fetched funding rates dict, or None to fetch fresh.

    Returns:
        -0.03 if funding is strongly positive (overleveraged longs -> bearish bias)
        +0.03 if funding is strongly negative (overleveraged shorts -> bullish bias)
         0.00 if funding is neutral or data is unavailable
    """
    if rates is None:
        rates = get_funding_rates()  # Fetch fresh or return cached rates

    # Get BTC funding rate; default to 0.0 if not available (neutral = no bias)
    btc_funding = rates.get("BTCUSDT", 0.0)

    # THRESHOLD: 0.0005 = 0.05% per 8 hours.
    # Below this threshold, funding is "normal" and not a strong signal.
    # At 0.05%, traders are paying $0.50 per $1,000 to hold leveraged positions every 8 hours —
    # not extreme yet, but starting to signal one-sided positioning.
    if btc_funding > 0.0005:  # > 0.05% — many leveraged longs, slightly crowded trade
        # CONTRARIAN LOGIC: Overleveraged longs are vulnerable to forced selling.
        # When prices dip, margin calls force these longs to sell, accelerating the drop.
        # We bet slightly against the crowd: bearish bias (lean NO).
        return -0.03  # -3% nudge toward NO
    elif btc_funding < -0.0005:  # < -0.05% — many leveraged shorts, also crowded
        # CONTRARIAN LOGIC: Overleveraged shorts are vulnerable to short squeezes.
        # When prices rise, forced short covering accelerates the move up.
        # Contrarian: bullish bias (lean YES).
        return 0.03  # +3% nudge toward YES
    return 0.0  # Neutral funding — no contrarian signal


def get_per_coin_funding_bias(coin_id: str, futures_data: Optional[dict] = None) -> float:
    """Per-coin contrarian funding bias using Binance perpetual funding rates.

    Unlike compute_funding_rate_bias() which uses BTC funding for all coins,
    this uses each coin's own futures funding rate when available, falling back
    to BTC for correlated assets (ETH, XRP) and 0.0 for uncorrelated ones (HYPE).

    WHY PER-COIN INSTEAD OF JUST BTC?
    While BTC dominates crypto sentiment, each coin can have its own funding dynamics.
    For example, XRP might have high funding while BTC's is neutral — meaning XRP-specific
    markets are overleveraged regardless of what BTC traders are doing.
    Using each coin's own funding rate gives a more precise contrarian signal.

    HYPERLIQUID (HYPE) SPECIAL CASE:
    HYPE has no Binance perpetual futures contract (it's too new/small). For HYPE,
    we use BTC funding at 50% weight (0.5×) as a rough proxy for market-wide sentiment,
    but attenuated because BTC funding doesn't directly represent HYPE positioning.

    Returns -0.03 to +0.03: negative = bearish bias, positive = bullish bias.
    """
    if futures_data is None:
        futures_data = get_futures_data()  # Fetch fresh or return cached

    # Use the coin's own funding rate if available in the futures data
    coin_data = futures_data.get(coin_id, {})
    rate = coin_data.get("funding_rate", None)

    # For HYPE (no Binance perp), use a muted BTC signal (0.5x weight)
    if rate is None and coin_id == "hyperliquid":
        btc_data = futures_data.get("bitcoin", {})
        # 0.5× weight because BTC funding is only loosely related to HYPE positioning
        rate = btc_data.get("funding_rate", 0.0) * 0.5

    if rate is None:
        rate = 0.0  # No data available — neutral (no opinion)

    if rate > 0.0005:    # > 0.05% → too many leveraged longs → bearish contrarian
        return -0.03
    elif rate < -0.0005:  # < -0.05% → too many leveraged shorts → bullish contrarian
        return 0.03
    return 0.0  # Neutral funding — no contrarian signal


def kelly_criterion_size(
    win_probability: float,
    price_cents: int,
    min_bet_cents: int = 200,
    max_bet_cents: int = 1500,
    kelly_fraction: float = 0.25,
    fee_pct: float = 0.07,
    bankroll_cents: int = 6000,
) -> int:
    """SIGNAL #5: Calculate optimal position size using the Kelly Criterion.

    THE KELLY CRITERION EXPLAINED:
    The Kelly Criterion is a formula from information theory (developed by John Kelly
    at Bell Labs, 1956) that calculates the optimal fraction of your bankroll to bet
    in order to maximize long-term growth. The full formula is:

        f* = (p * b - q) / b

    Where:
      f* = fraction of bankroll to wager
      p  = probability of winning (our combined probability estimate)
      q  = probability of losing (1 - p)
      b  = payout odds (net profit per dollar risked if you win)

    HOW IT APPLIES TO KALSHI BINARY MARKETS WITH FEES:
    Kalshi charges approximately 7% of gross profit on winning trades. This reduces
    the effective payout and must be factored into the Kelly calculation or we will
    systematically oversize every bet.

    Correct Kalshi fee formula: Fee = 0.07 × P × (1-P) per contract.
    Example at 40c entry:
      - Gross profit if YES wins: 100 - 40 = 60 cents
      - Fee: 0.07 × 40 × 60 / 100 = 1.68 cents
      - Net profit: 60 - 1.68 = 58.32 cents
      - Adjusted b = 58.32 / 40 = 1.458  (vs 1.5 without fees)
    (NOT 7% of gross profit — the old "fee = gross * 0.07" was 2-3x too high)

    Also accounts for 1c slippage: we place limit orders at bid+1, so the
    effective entry is already 1c above the best bid. price_cents passed in
    should reflect the actual fill price (bid+1), not the mid.

    WHY QUARTER-KELLY (0.25x)?
    Full Kelly assumes perfectly accurate probabilities. Quarter-Kelly (0.25x)
    provides a buffer against estimation errors, reducing variance significantly
    while capturing ~75% of the log-growth rate of full Kelly.

    WHEN KELLY RETURNS 0:
    If f* <= 0, the expected value after fees is negative — don't bet even if
    our combined_prob clears the 65% threshold. This is the final fee-aware gate.

    Args:
        win_probability: Our estimated probability of winning (0 to 1).
        price_cents: Actual fill price in cents (should be bid+1 for limit orders).
        min_bet_cents: Floor for total bet ($2.00 = 200 cents).
        max_bet_cents: Ceiling for total bet ($15.00 = 1500 cents).
        kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly).
        fee_pct: Kalshi's fee as a fraction of gross profit (default 0.07 = 7%).

    Returns:
        Total bet size in cents, clamped to [min_bet_cents, max_bet_cents].
        Returns 0 if Kelly says there's no edge after fees.
        Returns min_bet_cents if inputs are invalid.
    """
    # Guard rails: invalid inputs fall back to minimum bet
    if price_cents <= 0 or price_cents >= 100:
        return min_bet_cents  # Price must be 1-99 cents
    if win_probability <= 0 or win_probability >= 1:
        return min_bet_cents  # Probability must be between 0 and 1 (exclusive)

    p = win_probability         # Our estimated probability of winning
    q = 1.0 - p                 # Probability of losing (complement)

    # HOW KALSHI PAYOUTS WORK:
    # Each contract pays $1.00 (= 100 cents) if it resolves in your favor.
    # If you buy a YES contract at 40¢, your gross profit if YES wins = 100 - 40 = 60¢.
    # (And you lose your 40¢ stake if NO wins.)
    gross_profit = 100 - price_cents  # Cents earned per contract if we win

    # ── CORRECT Kalshi fee formula: 0.07 × P × (1-P) per contract ──────────
    # NOT a flat 7% of gross profit. The actual fee is probability-weighted:
    #   Fee = fee_pct × price_cents × (100 - price_cents) / 100
    #   At 40¢: 0.07 × 40 × 60 / 100 = 1.68¢  (was wrongly computed as 4.2¢)
    #   At 50¢: 0.07 × 50 × 50 / 100 = 1.75¢  (maximum, at ATM)
    #   At 20¢: 0.07 × 20 × 80 / 100 = 1.12¢  (lower-priced contracts pay less)
    # The old formula was 2–3× too expensive for sub-40¢ contracts, causing Kelly
    # to incorrectly return 0 (no edge) on profitable trades.
    fee_cents = fee_pct * price_cents * (100 - price_cents) / 100
    net_profit = gross_profit - fee_cents  # What we actually keep after Kalshi's cut

    if net_profit <= 0:
        return 0  # Fee consumes entire profit — no edge possible even if we win

    # PAYOUT ODDS (b) for the Kelly formula:
    # b = how many dollars we net per dollar bet if we win.
    # Example: 40¢ entry, 1.68¢ fee → net profit = 58.32¢
    #   b = 58.32 / 40 = 1.458  (we win 1.458× our stake after fees)
    # Compare: without fees b = 60/40 = 1.5
    b = net_profit / price_cents  # Net payout ratio

    if b <= 0:
        return min_bet_cents

    # THE KELLY FORMULA:
    # f* = (p × b - q) / b
    # where f* is the fraction of bankroll to bet.
    #
    # HOW TO INTERPRET:
    # - If p=0.65 and b=1.458: f* = (0.65×1.458 - 0.35) / 1.458 = (0.948 - 0.35) / 1.458 = 0.41
    #   This says: bet 41% of your bankroll. With quarter-Kelly: bet 10.25%.
    #   On $100 bankroll: bet $10.25 (which fits within $2-$15 range).
    #
    # - If p=0.60 and b=1.458: f* = (0.60×1.458 - 0.40) / 1.458 = 0.189
    #   Quarter-Kelly: bet 4.7%. On $100 bankroll: $4.72.
    #
    # - If p=0.50 (coin flip) and b=1.458: f* = (0.5×1.458 - 0.5) / 1.458 = 0.186/1.458 = 0.13
    #   But once fees are applied, this goes NEGATIVE for near-50/50 markets — correctly
    #   telling us not to bet on markets with no edge.
    kelly_f = (p * b - q) / b

    if kelly_f <= 0:
        return 0  # Kelly says the expected value is NEGATIVE after fees — skip this trade

    # Apply the fractional Kelly multiplier (0.25 = quarter-Kelly).
    # Reduces bet size to account for uncertainty in our probability estimates.
    bet_fraction = kelly_f * kelly_fraction

    # Convert fraction to dollar amount: fraction × total bankroll.
    # bankroll_cents is passed from the caller (current account balance).
    bet_cents = int(bet_fraction * bankroll_cents)

    # Clamp to [min_bet_cents, max_bet_cents] — the bot never bets less than $2 or more than $15.
    # int() truncates (rounds down) — we never bet MORE than Kelly says.
    bet_cents = max(min_bet_cents, min(max_bet_cents, bet_cents))
    return bet_cents


# ── Hurst Exponent (regime detection) ────────────────────────────────────────
# WHAT IS A MARKET REGIME?
# Financial markets don't behave the same way all the time. They switch between
# different "regimes" (modes of behavior):
#
#   TRENDING REGIME: Prices tend to keep moving in the same direction.
#   "Bitcoin went up today → probably goes up tomorrow too."
#   This is called "momentum" or "persistence" — past moves predict future moves.
#
#   MEAN-REVERTING REGIME: Prices tend to snap back after moving away from average.
#   "Bitcoin went up a lot → probably comes back down soon."
#   This is the "rubber band" effect — stretching creates pressure to return.
#
#   RANDOM WALK REGIME: Price moves are unpredictable — no edge from recent history.
#   "Past moves tell you nothing about future moves." Coin-flip territory.
#
# The Hurst Exponent measures which regime the market is currently in.
# Knowing the regime helps calibrate our probability estimates.

def _compute_hurst_rs(log_returns: list[float]) -> float:
    """Compute Hurst exponent via Rescaled Range (R/S) analysis.

    H > 0.55: trending (momentum) regime — price moves are persistent
    H < 0.45: mean-reverting regime — price moves tend to reverse
    H ≈ 0.50: random walk — no reliable directional bias

    Uses ordinary least squares on log(lag) vs log(R/S) across multiple lag windows.
    Pure Python — no numpy dependency.

    WHAT IS R/S ANALYSIS?
    R/S = "Rescaled Range". The algorithm works by:
    1. Taking a window of returns (e.g., the last 20 price changes)
    2. Computing R = range of cumulative deviations (max - min of running total)
    3. Computing S = standard deviation of those returns
    4. R/S tells us how "wild" or "persistent" the movements were in that window
    5. Repeat for many different window sizes (lags)
    6. The slope of log(lag) vs log(R/S) = Hurst exponent

    For a random walk, R/S grows as sqrt(lag), giving slope = 0.5.
    For a trending series, R/S grows faster, giving slope > 0.5.
    For a mean-reverting series, R/S grows slower, giving slope < 0.5.

    WHY OLS (ORDINARY LEAST SQUARES)?
    OLS finds the "best fit line" through a set of points — it minimizes the
    sum of squared vertical distances from each point to the line.
    The slope of this line IS the Hurst exponent.

    WHY PURE PYTHON (NO NUMPY)?
    numpy is a numerical library that's fast but adds a dependency. For this
    calculation, pure Python is fast enough and avoids the complexity.
    """
    import math
    n = len(log_returns)
    if n < 20:
        return 0.5  # Not enough data — return neutral (random walk assumption)

    # Choose range of window sizes (lags) to analyze.
    # min_lag: smallest window (at least 8 bars to get meaningful stats)
    # max_lag: largest window (at most half the data, up to 80 bars)
    # step: spacing between lags (evenly spread across the range)
    min_lag = max(8, n // 12)
    max_lag = min(n // 2, 80)
    step = max(1, (max_lag - min_lag) // 15)
    lags = list(range(min_lag, max_lag + 1, step))

    # Accumulate (log_lag, log_RS) pairs for OLS regression
    log_rs_pairs: list[tuple[float, float]] = []
    for lag in lags:
        rs_vals: list[float] = []
        # Slice the return series into non-overlapping chunks of length `lag`
        for start in range(0, n - lag, lag):
            chunk = log_returns[start: start + lag]
            if len(chunk) < 4:
                continue
            # mu = mean return of this chunk
            mu = sum(chunk) / len(chunk)
            # adj = returns adjusted by mean (deviation from average)
            adj = [x - mu for x in chunk]
            # Cumulative deviation profile: running sum of deviations
            # This shows how far the cumulative path wanders from zero
            profile: list[float] = []
            cum = 0.0
            for x in adj:
                cum += x
                profile.append(cum)
            # R = range of the profile = how far it wandered (max - min)
            R = max(profile) - min(profile)
            # S = standard deviation of returns in this chunk (scale factor)
            var = sum((x - mu) ** 2 for x in chunk) / len(chunk)
            S = var ** 0.5
            if S > 1e-12:  # Avoid division by near-zero (constant returns)
                rs_vals.append(R / S)  # R/S = rescaled range for this chunk
        if rs_vals:
            avg_rs = sum(rs_vals) / len(rs_vals)  # Average R/S across all chunks at this lag
            if avg_rs > 0:
                # Take logs: the relationship is linear in log-log space
                log_rs_pairs.append((math.log(lag), math.log(avg_rs)))

    if len(log_rs_pairs) < 5:
        return 0.5  # Not enough lag points for reliable regression

    # OLS (Ordinary Least Squares) linear regression to find the slope.
    # slope = Hurst exponent
    xs = [p[0] for p in log_rs_pairs]  # log(lag) values
    ys = [p[1] for p in log_rs_pairs]  # log(R/S) values
    n_pts = len(xs)
    mean_x = sum(xs) / n_pts
    mean_y = sum(ys) / n_pts
    # Numerator = sum of (x - mean_x) × (y - mean_y) = covariance (scaled)
    num = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n_pts))
    # Denominator = sum of (x - mean_x)^2 = variance of x (scaled)
    den = sum((xs[i] - mean_x) ** 2 for i in range(n_pts))
    if den < 1e-12:
        return 0.5  # All lags are identical (shouldn't happen) — return neutral
    # slope = covariance / variance = OLS slope = Hurst exponent
    h = num / den
    # Clamp to [0.1, 0.9]: values outside this range are numerical artifacts, not real signals
    return max(0.1, min(0.9, h))


def get_hurst_exponent(n_bars: int = 100) -> dict[str, float]:
    """Compute Hurst exponent for BTC/ETH/XRP from recent 5-minute Binance klines.

    Uses Rescaled Range (R/S) analysis on the last ~8 hours of 5-min returns.
    Results are cached for 15 minutes (regime shifts are slow-moving).

    Returns:
        dict mapping coin_id -> Hurst exponent in [0.1, 0.9].
        Falls back to 0.5 (neutral/random walk) on any fetch or compute failure.
    """
    import math
    global _hurst_cache, _hurst_cache_time

    now = time.time()
    # Return cached Hurst values if still fresh (15-minute TTL).
    # .copy() returns a new dict so the caller can't accidentally modify our cache.
    if now - _hurst_cache_time < _HURST_CACHE_TTL and _hurst_cache:
        return _hurst_cache.copy()

    # Default to 0.5 (random walk / no regime signal) for all three major coins.
    # These defaults are used if data fetching fails.
    result: dict[str, float] = {"bitcoin": 0.5, "ethereum": 0.5, "ripple": 0.5}
    # Map our internal coin IDs to Binance's futures symbol format
    _SYMBOL_MAP = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "ripple": "XRPUSDT"}

    # Using `requests` library here (different from `httpx` used elsewhere).
    # Both do the same thing (HTTP requests), but requests is already available
    # in this environment and is slightly simpler for synchronous calls.
    import requests as _req
    for coin_id, symbol in _SYMBOL_MAP.items():
        try:
            # Fetch recent 5-minute candlestick data from Binance Futures.
            # n_bars=100 gives ~8.3 hours of 5-minute bars — enough to detect recent regime.
            # Using futures klines (fapi.binance.com) because futures data is more
            # liquid and informative than spot for regime detection.
            url = (
                "https://fapi.binance.com/fapi/v1/klines"
                f"?symbol={symbol}&interval=5m&limit={n_bars}"
            )
            resp = _req.get(url, timeout=8)
            if resp.status_code != 200:
                continue  # Geo-blocked or network error — skip this coin
            bars = resp.json()
            if len(bars) < 20:
                continue  # Not enough data for meaningful Hurst calculation
            # Extract close prices (index 4 in each kline array)
            closes = [float(b[4]) for b in bars]
            # Compute log returns. The `if` condition skips any zero prices (shouldn't happen but safe).
            log_rets = [
                math.log(closes[i] / closes[i - 1])
                for i in range(1, len(closes))
                if closes[i - 1] > 0 and closes[i] > 0
            ]
            # Run R/S analysis to get the Hurst exponent
            h = _compute_hurst_rs(log_rets)
            result[coin_id] = h
            logger.debug(f"[hurst] {coin_id}: H={h:.3f}")
        except Exception as exc:
            logger.debug(f"[hurst] {coin_id} fetch/compute failed: {exc}")
            # Keep the default 0.5 for this coin — safe fallback

    # Store results in cache for next 15 minutes
    _hurst_cache = result.copy()
    _hurst_cache_time = now
    return result
