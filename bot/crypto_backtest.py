"""
Crypto quant strategy backtester.

Replays the full entry/exit logic (Student-t fair prob, logit-averaging,
Kelly sizing, stop-loss, take-profit, de-risk) against Binance historical
15-minute klines for BTC, ETH, XRP.

How it works
------------
1.  Fetch 3 months of 15m klines from Binance for BTCUSDT / ETHUSDT / XRPUSDT.
2.  For every closed 15-min candle:
    a. Simulate the Kalshi KXBTC15M/KXETH15M/KXXRP15M market that was live
       during that candle.  Strike = open price of the candle.
    b. Compute fair_prob with our Student-t model at entry (simulated mid-candle
       snapshot, at T = 7.5 minutes remaining).
    c. Derive p_funding (random walk centred at 0 since we have no Binance
       futures history), p_imbalance (neutral 0.5), p_tod (time-of-day from
       candle open timestamp), p_consensus = market mid (approximated).
    d. Apply logit-averaging to get combined fair_prob.
    e. Check entry gate (≥ 0.65 or ≤ 0.35, longshot ≥ 15c).
    f. Size with Kelly (quarter-Kelly tier based on win_prob, $2–$15 cap).
    g. Simulate settlement at candle close.
    h. Apply 7% Kalshi fee formula.
3.  Report:  win rate, total PnL, Sharpe ratio, max drawdown, profit factor,
    per-coin breakdown, and equity curve.

Notes
-----
- We use open price as the strike so the candle fully tests the 15-min window.
- Funding / LOB signals are set neutral (0.5) because we don't have historical
  Binance futures data archived; the backtest therefore UNDER-estimates real
  performance when those signals add edge.
- Execution is assumed at mid (no slippage); 1¢ slippage penalty added.
"""

import math
import time
import statistics
from datetime import datetime, timezone
from typing import Any

import requests
from scipy.stats import t as student_t


# ─── Signal combination (mirrors server.py exactly) ──────────────────────────

def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1.0 - p))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _combined_fair(p_price, p_funding, p_imbalance, p_consensus, p_tod) -> float:
    raw = _sigmoid(
        0.62 * _logit(p_price) +
        0.20 * _logit(p_funding) +
        0.08 * _logit(p_imbalance) +
        0.05 * _logit(p_consensus) +
        0.05 * _logit(p_tod)
    )
    return max(0.01, min(0.99, raw))


# ─── Student-t fair probability (mirrors crypto_feed.py) ─────────────────────

# Base 3 coins (live bot)
_DF = {
    "bitcoin": 4, "ethereum": 3.5, "ripple": 3,
    # Extended
    "solana": 3.0, "dogecoin": 2.5, "litecoin": 4.0,
    "cardano": 3.0, "avalanche": 3.0,
    "bnb": 4.0, "hyperliquid": 2.5,
}
_ANN_VOL = {
    "bitcoin": 0.75, "ethereum": 0.95, "ripple": 1.15,
    # Extended — higher vol / fatter tails for alt coins
    "solana": 1.25, "dogecoin": 1.50, "litecoin": 0.90,
    "cardano": 1.20, "avalanche": 1.30,
    "bnb": 0.85,  "hyperliquid": 2.00,  # HYPE is very new / volatile
}

def _fair_prob_student_t(spot: float, strike: float, secs_remaining: float, coin: str,
                         vol_scale: float = 1.0) -> float:
    ann_vol = _ANN_VOL[coin] * vol_scale
    days_left = max(secs_remaining / 86400, 1 / 1440)
    vol_horizon = ann_vol * (days_left / 365.0) ** 0.5
    if vol_horizon <= 0:
        return 0.5
    z = math.log(spot / strike) / vol_horizon
    df = _DF[coin]
    return float(student_t.cdf(z, df=df))


def _fair_prob_range(spot: float, strike: float, secs_remaining: float,
                     coin: str, range_pct: float = 0.005, vol_scale: float = 1.0) -> float:
    """
    P(price closes within ±range_pct of strike).
    Models the Kalshi hourly-range market: YES settles if spot ends inside the band.
    Uses two-sided Student-t: P = 2*CDF(z_bound) - 1.
    """
    ann_vol = _ANN_VOL[coin] * vol_scale
    days_left = max(secs_remaining / 86400, 1 / 1440)
    vol_horizon = ann_vol * (days_left / 365.0) ** 0.5
    if vol_horizon <= 0:
        return 0.5
    z_bound = math.log(1.0 + range_pct) / vol_horizon   # z for upper edge of band
    df = _DF[coin]
    prob = 2.0 * float(student_t.cdf(z_bound, df=df)) - 1.0
    return max(0.01, min(0.99, prob))


# ─── Kelly sizing (mirrors crypto_feed.py exactly) ────────────────────────────

def _kelly_size(win_prob: float, price_cents: int,
                min_bet: int = 200, max_bet: int = 1500,
                kelly_fraction: float = 0.25,
                bankroll_cents: int = 10_000) -> int:
    if price_cents <= 0 or price_cents >= 100:
        return 0
    gross_profit = 100 - price_cents
    fee_cents = 0.07 * price_cents * (100 - price_cents) / 100
    net_profit = gross_profit - fee_cents
    if net_profit <= 0:
        return 0
    loss = price_cents
    edge = win_prob * net_profit - (1 - win_prob) * loss
    if edge <= 0:
        return 0
    full_kelly = edge / net_profit
    fraction = full_kelly * kelly_fraction
    # Size proportional to current bankroll (compounds as balance grows)
    bet = int(fraction * bankroll_cents)
    # Dynamic cap: 15% of current bankroll, floored at fixed max_bet, hard-capped at liquidity limit
    dynamic_max = min(MAX_BET_ABS, max(max_bet, int(bankroll_cents * 0.15)))
    return max(0, min(dynamic_max, max(min_bet if bet >= min_bet else 0, bet)))


# ─── Time-of-day bias (mirrors server.py) ────────────────────────────────────

def _tod_bias(hour_utc: int) -> float:
    """Static ±0.05 time-of-day bias."""
    if 12 <= hour_utc < 20:   # US trading hours → slight upward bias
        return 0.02
    elif 0 <= hour_utc < 6:   # Asian hours → slight downward
        return -0.02
    return 0.0


# ─── Hurst threshold (assume neutral 0.5 without live calc) ──────────────────

def _entry_threshold(hurst: float = 0.5) -> float:
    if hurst > 0.55:
        return max(0.62, 0.65 * 0.85)
    elif hurst < 0.45:
        return min(0.70, 0.65 * 1.20)
    return 0.65


# ─── Fetch Binance klines ─────────────────────────────────────────────────────

def _fetch_klines(symbol: str, interval: str = "15m", limit: int = 1000,
                  end_time_ms: int | None = None) -> list[list]:
    url = "https://api.binance.us/api/v3/klines"
    params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms:
        params["endTime"] = end_time_ms
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def fetch_all_klines(symbol: str, n_days: int = 90, interval: str = "15m") -> list[dict]:
    """Fetch n_days of klines for symbol at given interval."""
    candles_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48, "1h": 24, "4h": 6, "1d": 1}.get(interval, 96)
    print(f"  Fetching {n_days}d of {interval} klines for {symbol}...", flush=True)
    candles = []
    limit = 1000
    end_ms = None
    target = n_days * candles_per_day

    while len(candles) < target:
        batch = _fetch_klines(symbol, interval, limit, end_ms)
        if not batch:
            break
        # Each row: [open_time, open, high, low, close, volume, close_time, ...]
        for row in batch:
            candles.append({
                "open_time_ms":  int(row[0]),
                "open":          float(row[1]),
                "high":          float(row[2]),
                "low":           float(row[3]),
                "close":         float(row[4]),
                "volume":        float(row[5]),
                "close_time_ms": int(row[6]),
            })
        end_ms = batch[0][0] - 1  # paginate backward
        if len(batch) < limit:
            break
        time.sleep(0.1)  # polite rate limit

    candles.sort(key=lambda x: x["open_time_ms"])
    # Drop duplicates
    seen = set()
    unique = []
    for c in candles:
        if c["open_time_ms"] not in seen:
            seen.add(c["open_time_ms"])
            unique.append(c)
    print(f"    → {len(unique)} candles ({unique[0]['open_time_ms']} to {unique[-1]['close_time_ms']})", flush=True)
    return unique[-target:]  # keep most recent n_days worth


# ─── Main backtest ────────────────────────────────────────────────────────────

COINS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "ripple": "XRPUSDT",
}

COINS_EXTENDED = {
    "bitcoin":      "BTCUSDT",
    "ethereum":     "ETHUSDT",
    "ripple":       "XRPUSDT",
    "solana":       "SOLUSDT",
    "dogecoin":     "DOGEUSDT",
    "bnb":          "BNBUSDT",
    "hyperliquid":  "HYPEUSDT",
    "litecoin":     "LTCUSDT",
    "cardano":      "ADAUSDT",
    "avalanche":    "AVAXUSDT",
}

# 7-coin target set — per-coin thresholds applied in run_backtest
COINS_TARGET = {
    "bitcoin":      "BTCUSDT",
    "ethereum":     "ETHUSDT",
    "ripple":       "XRPUSDT",
    "solana":       "SOLUSDT",
    "dogecoin":     "DOGEUSDT",
    "bnb":          "BNBUSDT",
    "hyperliquid":  "HYPEUSDT",
}

# Per-coin minimum fair probability — higher for lower-WR coins
# BNB (81.8% WR, -$98) and HYPE (81.4% WR) need higher certainty to trigger
COIN_MIN_PROB: dict[str, float] = {
    "bitcoin":     0.60,
    "ethereum":    0.60,
    "ripple":      0.60,
    "solana":      0.60,
    "dogecoin":    0.60,
    "bnb":         0.75,   # was net loser — only take very high-confidence trades
    "hyperliquid": 0.72,   # borderline — slightly elevated threshold
    "litecoin":    0.80,   # known loser
    "cardano":     0.60,
    "avalanche":   0.60,
}

LONGSHOT_MIN  = 15   # ¢ — skip entries below this
SLIPPAGE      = 1    # cents per contract
MKTEFF        = 0.80 # 80% market efficiency — 20% lag = our edge (conservative)
START_BAL     = 17_200  # $172 in cents (actual account balance)
MAX_BET       = 1500  # $15 default max per trade
MAX_BET_ABS   = 30_000  # $300 absolute ceiling (Kalshi market liquidity limit)

# ── Default parameters (overridden in sensitivity sweep) ─────────────────────
MAX_POSITIONS  = 5    # max simultaneous open positions
MIN_BET        = 200  # $2 min per trade
MIN_FAIR_PROB  = 0.65 # entry threshold
MIN_EDGE       = 0.03 # minimum (win_prob - entry_price) gap required


def run_backtest(n_days: int = 90,
                 coin_candles: dict | None = None,
                 max_positions: int = MAX_POSITIONS,
                 min_bet: int = MIN_BET,
                 min_fair_prob: float = MIN_FAIR_PROB,
                 min_edge: float = MIN_EDGE,
                 kelly_mult: float = 1.0,
                 max_bet_override: int | None = None,
                 longshot_min: int = LONGSHOT_MIN,
                 secs_remaining_override: float | None = None,
                 mkteff_override: float | None = None,
                 vol_scale: float = 1.0,
                 market_type: str = "15m",        # "15m" | "1h" | "1h_range" | "5m_realistic" | "combined"
                 coins_override: dict | None = None,
                 range_pct: float = 0.005,
                 range_pcts: list | None = None,  # for combined: list of range band sizes to include
                 ) -> dict:
    min_fair_no = round(1.0 - min_fair_prob, 4)

    # ── Resolve sub-market-types for "combined" mode ──────────────────────────
    # "combined" runs 15m directional + 1h directional + 1h range bands in one
    # unified timeline with shared balance & position tracker.
    _range_pcts = range_pcts or [range_pct]
    if market_type == "combined":
        _sub_types = (
            [("15m",     "15m", 1, rp) for rp in [range_pct]]
          + [("1h",      "1h",  1, rp) for rp in [range_pct]]
          + [("1h_range","1h",  1, rp) for rp in _range_pcts]
        )
    else:
        _interval_map = {"5m_realistic": "5m", "1h": "1h", "1h_range": "1h"}
        _ci = _interval_map.get(market_type, "15m")
        _stride = 3 if market_type == "5m_realistic" else 1
        _sub_types = [(market_type, _ci, _stride, range_pct)]

    # ── Resolve coin set ──────────────────────────────────────────────────────
    active_coins = coins_override if coins_override is not None else COINS

    # ── Shared balance and position tracker across all coins & market types ───
    balance     = START_BAL
    open_pos: list[dict] = []
    all_trades: list[dict] = []

    # ── Fetch candles for every unique interval needed ────────────────────────
    _intervals_needed = set(ci for _, ci, _, _ in _sub_types)
    # coin_candles_by_interval: {"15m": {coin: [...]}, "1h": {coin: [...]}, ...}
    coin_candles_by_interval: dict[str, dict] = {}
    for ci in _intervals_needed:
        if coin_candles is not None and ci == _sub_types[0][1]:
            # reuse pre-fetched candles for the primary interval
            coin_candles_by_interval[ci] = coin_candles
        else:
            coin_candles_by_interval[ci] = {}
            for coin, symbol in active_coins.items():
                try:
                    candles = fetch_all_klines(symbol, n_days, interval=ci)
                    if candles:
                        coin_candles_by_interval[ci][coin] = candles
                    else:
                        print(f"  [skip] {coin} ({symbol}) @{ci}: no candles", flush=True)
                except Exception as e:
                    print(f"  [skip] {coin} ({symbol}) @{ci}: {e}", flush=True)

    # Default coin_candles for backward-compat (primary interval)
    _primary_ci = _sub_types[0][1]
    coin_candles = coin_candles_by_interval[_primary_ci]

    # Only track per-coin stats for coins we actually have data for
    per_coin = {c: {"trades": 0, "wins": 0, "pnl": 0}
                for c in active_coins if c in coin_candles}

    # Build unified timeline across all sub_types and intervals
    # Each entry: (open_time_ms, coin, candle_idx, sub_market_type, candle_interval, sub_range_pct)
    timeline: list[tuple[int, str, int, str, str, float]] = []
    for _mt, _ci, _stride, _rp in _sub_types:
        _candles_map = coin_candles_by_interval.get(_ci, {})
        for coin, candles in _candles_map.items():
            if coin not in active_coins:
                continue
            for i in range(0, len(candles) - _stride, _stride):
                timeline.append((candles[i]["open_time_ms"], coin, i, _mt, _ci, _rp))
    timeline.sort(key=lambda x: x[0])

    for ts_ms, coin, i, _cur_mt, _cur_ci, _cur_rp in timeline:
        candles = coin_candles_by_interval[_cur_ci][coin]
        c = candles[i]

        if _cur_mt == "5m_realistic":
            if i + 2 >= len(candles):
                continue
            c_settle = candles[i + 2]
            c_next   = candles[i + 1]
        else:
            if i + 1 >= len(candles):
                continue
            c_next   = candles[i + 1]
            c_settle = c_next

        # Gap check
        gap_ms = c_next["open_time_ms"] - c["close_time_ms"]
        if gap_ms > 5_000:
            continue

        # ── Settle any open positions that expired at this timestamp ──────
        still_open = []
        for pos in open_pos:
            if pos["settle_ts_ms"] <= ts_ms:
                coin_c = pos["coin"]
                res_candle_idx = pos["settle_candle_idx"]
                pos_ci = pos.get("candle_interval", _primary_ci)
                res_c = coin_candles_by_interval[pos_ci][coin_c][res_candle_idx]
                settle_price = res_c["close"]
                if pos.get("market_type") == "1h_range":
                    # Range market: YES wins if close is within ±range_pct of strike
                    rp = pos.get("range_pct", 0.005)
                    settled_yes = abs(settle_price - pos["strike"]) / pos["strike"] <= rp
                else:
                    settled_yes = settle_price > pos["strike"]
                won = (pos["side"] == "yes" and settled_yes) or \
                      (pos["side"] == "no"  and not settled_yes)
                entry_c   = pos["entry"]
                fee_total = 0.07 * entry_c * (100 - entry_c) / 100 * pos["n"]
                # Entry cost was already deducted at order time.
                # On win: receive 100c/contract, pay fee.
                # On loss: receive 0c/contract, pay fee. Entry cost already gone.
                pnl_per = (100 - entry_c - fee_total / pos["n"]) if won else (-entry_c - fee_total / pos["n"])
                pnl     = int(pnl_per * pos["n"])
                if won:
                    balance += int(100 * pos["n"] - fee_total)
                else:
                    balance -= int(fee_total)
                all_trades.append({**pos, "won": won, "pnl": pnl,
                                    "settle_price": settle_price})
                per_coin[pos["coin"]]["trades"] += 1
                per_coin[pos["coin"]]["wins"]   += 1 if won else 0
                per_coin[pos["coin"]]["pnl"]    += pnl
            else:
                still_open.append(pos)
        open_pos = still_open

        # ── Skip entry if at position limit or insufficient funds ─────────
        if len(open_pos) >= max_positions:
            continue
        crypto_budget   = int(balance * 0.80)
        crypto_exposure = sum(p["entry"] * p["n"] for p in open_pos)
        available       = max(0, crypto_budget - crypto_exposure)
        if available < min_bet:
            continue

        # ── Signal computation ────────────────────────────────────────────
        strike = c["open"]
        spot   = c["close"]
        if strike <= 0 or spot <= 0:
            continue

        # Per-coin minimum probability (higher for lower-WR coins like BNB/HYPE)
        _coin_min_prob = COIN_MIN_PROB.get(coin, min_fair_prob)
        _effective_min_prob = max(min_fair_prob, _coin_min_prob)
        _effective_min_no   = round(1.0 - _effective_min_prob, 4)

        # Seconds remaining: 15m=900s, 1h=3600s, 5m_realistic=600s
        _default_secs_map = {"15m": 900.0, "1h": 3600.0, "1h_range": 3600.0, "5m_realistic": 600.0}
        secs_remaining = secs_remaining_override if secs_remaining_override is not None \
                         else _default_secs_map.get(_cur_mt, 900.0)

        open_ms  = c["open_time_ms"]
        hour_utc = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).hour

        if _cur_mt == "1h_range":
            fair_prob = _fair_prob_range(spot, strike, secs_remaining, coin,
                                         range_pct=_cur_rp, vol_scale=vol_scale)
        else:
            p_price     = _fair_prob_student_t(spot, strike, secs_remaining, coin,
                                               vol_scale=vol_scale)
            p_tod       = max(0.01, min(0.99, 0.5 + _tod_bias(hour_utc)))
            p_funding   = 0.5
            p_imbalance = 0.5
            p_consensus = max(0.01, min(0.99, p_price))
            fair_prob   = _combined_fair(p_price, p_funding, p_imbalance,
                                         p_consensus, p_tod)

        # ── Entry gate (per-coin threshold) ───────────────────────────────
        _eff = mkteff_override if mkteff_override is not None else MKTEFF
        if _cur_mt == "1h_range":
            if fair_prob < _effective_min_prob:
                continue
            side        = "yes"
            mkt_mid     = 0.5 + _eff * (fair_prob - 0.5)
            entry_cents = min(98, int(mkt_mid * 100) + 2 + SLIPPAGE)
            win_prob    = fair_prob
        elif fair_prob >= _effective_min_prob:
            side        = "yes"
            mkt_mid     = 0.5 + _eff * (fair_prob - 0.5)
            entry_cents = min(98, int(mkt_mid * 100) + 2 + SLIPPAGE)
            win_prob    = fair_prob
        elif fair_prob <= _effective_min_no:
            side        = "no"
            no_fair     = 1.0 - fair_prob
            mkt_mid_no  = 0.5 + _eff * (no_fair - 0.5)
            entry_cents = min(98, int(mkt_mid_no * 100) + 2 + SLIPPAGE)
            win_prob    = no_fair
        else:
            continue

        if entry_cents < longshot_min:
            continue

        edge_gap = win_prob - entry_cents / 100.0
        if edge_gap < min_edge:
            continue

        # ── Kelly sizing (proportional to current balance) ────────────────
        if win_prob >= 0.80:
            kelly_frac = 0.50 * 0.25 * kelly_mult
        elif win_prob >= 0.70:
            kelly_frac = 0.35 * 0.25 * kelly_mult
        else:
            kelly_frac = 0.20 * 0.25 * kelly_mult

        _max_bet_eff = max_bet_override if max_bet_override is not None else MAX_BET
        bet_cents = _kelly_size(win_prob, entry_cents, min_bet,
                                min(_max_bet_eff, available), kelly_frac,
                                bankroll_cents=balance)
        if bet_cents <= 0:
            continue
        n_contracts = max(1, bet_cents // max(entry_cents, 1))
        cost        = entry_cents * n_contracts

        if cost > available:
            continue

        balance -= cost

        _settle_idx = i + 2 if _cur_mt == "5m_realistic" else i + 1
        open_pos.append({
            "coin":            coin,
            "side":            side,
            "strike":          strike,
            "entry":           entry_cents,
            "n":               n_contracts,
            "fair_prob":       round(fair_prob, 3),
            "win_prob":        round(win_prob, 3),
            "hour":            hour_utc,
            "spot":            spot,
            "settle_ts_ms":    c_settle["close_time_ms"],
            "settle_candle_idx": _settle_idx,
            "candle_interval": _cur_ci,
            "market_type":     _cur_mt,
            "range_pct":       _cur_rp,
        })

    # Settle any remaining open positions at last available price
    for pos in open_pos:
        _pos_ci = pos.get("candle_interval", _primary_ci)
        res_c = coin_candles_by_interval[_pos_ci][pos["coin"]][pos["settle_candle_idx"]]
        settle_price = res_c["close"]
        if pos.get("market_type") == "1h_range":
            rp = pos.get("range_pct", 0.005)
            settled_yes = abs(settle_price - pos["strike"]) / pos["strike"] <= rp
        else:
            settled_yes = settle_price > pos["strike"]
        won  = (pos["side"] == "yes" and settled_yes) or \
               (pos["side"] == "no" and not settled_yes)
        entry_c = pos["entry"]
        fee     = 0.07 * entry_c * (100 - entry_c) / 100
        fee_total = 0.07 * entry_c * (100 - entry_c) / 100 * pos["n"]
        pnl_per   = (100 - entry_c - fee_total/pos["n"]) if won else (-entry_c - fee_total/pos["n"])
        pnl       = int(pnl_per * pos["n"])
        if won:
            balance += int(100 * pos["n"] - fee_total)
        else:
            balance -= int(fee_total)
        all_trades.append({**pos, "won": won, "pnl": pnl,
                            "settle_price": settle_price})
        per_coin[pos["coin"]]["trades"] += 1
        per_coin[pos["coin"]]["wins"]   += 1 if won else 0
        per_coin[pos["coin"]]["pnl"]    += pnl

    if not all_trades:
        return {}

    # ── Aggregate stats ───────────────────────────────────────────────────────
    total     = len(all_trades)
    wins      = sum(1 for t in all_trades if t["won"])
    total_pnl = balance - START_BAL  # use actual balance delta (most accurate)
    wr = wins / total if total else 0

    gross_wins  = sum(t["pnl"] for t in all_trades if t["won"])
    gross_losses = abs(sum(t["pnl"] for t in all_trades if not t["won"]))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Equity curve + Sharpe
    equity = START_BAL
    peak   = equity
    max_dd = 0
    pnl_series = []
    equity_curve = []
    for t in all_trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        dd = peak - equity
        max_dd = max(max_dd, dd)
        pnl_series.append(t["pnl"])
        equity_curve.append(equity)

    avg_pnl = statistics.mean(pnl_series)
    std_pnl = statistics.stdev(pnl_series) if len(pnl_series) > 1 else 1
    sharpe  = (avg_pnl / std_pnl) * (total / n_days * 365) ** 0.5 if std_pnl > 0 else 0
    # Simplified annualised Sharpe ≈ per-trade Sharpe × √(trades/year)
    ann_sharpe = (avg_pnl / std_pnl) * math.sqrt(total * 365 / n_days) if std_pnl > 0 else 0

    # Calmar ratio
    calmar = (total_pnl / max_dd) if max_dd > 0 else float("inf")

    # Win-probability by confidence tier
    high_conf  = [t for t in all_trades if t["win_prob"] >= 0.80]
    mid_conf   = [t for t in all_trades if 0.70 <= t["win_prob"] < 0.80]
    low_conf   = [t for t in all_trades if t["win_prob"] < 0.70]

    def _wr(lst):
        return sum(1 for t in lst if t["won"]) / len(lst) if lst else 0

    result = {
        "n_days":             n_days,
        "total_trades":       total,
        "wins":               wins,
        "losses":             total - wins,
        "win_rate":           wr,
        "total_pnl_cents":    total_pnl,
        "final_balance_cents": balance,
        "profit_factor":      profit_factor,
        "max_drawdown_cents": max_dd,
        "ann_sharpe":         ann_sharpe,
        "calmar":             calmar,
        "per_coin":           per_coin,
        "tier_wr": {
            "high_conf_wr": _wr(high_conf),
            "mid_conf_wr":  _wr(mid_conf),
            "low_conf_wr":  _wr(low_conf),
            "high_conf_n":  len(high_conf),
            "mid_conf_n":   len(mid_conf),
            "low_conf_n":   len(low_conf),
        },
        "equity_curve": equity_curve[-100:],
    }
    return result


def print_report(r: dict, label: str = ""):
    if not r:
        return
    tag = f" [{label}]" if label else ""
    print(f"\n{'='*64}")
    print(f"  BACKTEST RESULTS — {r['n_days']}-day window{tag}")
    print(f"  Market efficiency assumption: {int(MKTEFF*100)}% (conservative)")
    print(f"{'='*64}")
    print(f"  Trades:         {r['total_trades']}  ({r['total_trades']/r['n_days']:.1f}/day)")
    print(f"  Win rate:       {r['win_rate']:.1%}  ({r['wins']}W / {r['losses']}L)")
    print(f"  Total PnL:      ${r['total_pnl_cents']/100:+.2f}")
    print(f"  Final balance:  ${r['final_balance_cents']/100:.2f}  (started ${START_BAL/100:.2f})")
    print(f"  Return:         {r['total_pnl_cents']/START_BAL*100:+.1f}%  over {r['n_days']} days")
    print(f"  Ann. return:    {r['total_pnl_cents']/START_BAL*100*365/r['n_days']:+.1f}%")
    print(f"  Profit factor:  {r['profit_factor']:.2f}x")
    print(f"  Max drawdown:   ${r['max_drawdown_cents']/100:.2f}")
    print(f"  Ann. Sharpe:    {r['ann_sharpe']:.2f}")
    print(f"  Calmar ratio:   {r['calmar']:.2f}")
    print()
    print(f"  ── Per-coin breakdown ──")
    for coin, s in r['per_coin'].items():
        n  = s['trades']
        wr_c = s['wins'] / n if n else 0
        print(f"  {coin:10s}  {n:4d} trades  WR={wr_c:.1%}  PnL=${s['pnl']/100:+.2f}")
    print()
    print(f"  ── Confidence tier win rates ──")
    t = r['tier_wr']
    print(f"  ≥80% conf:  WR={t['high_conf_wr']:.1%}  (n={t['high_conf_n']})")
    print(f"  70–79%:     WR={t['mid_conf_wr']:.1%}  (n={t['mid_conf_n']})")
    print(f"  65–69%:     WR={t['low_conf_wr']:.1%}  (n={t['low_conf_n']})")
    print(f"{'='*64}\n")


def run_sensitivity(n_days: int = 90):
    """
    Sweep MAX_POSITIONS × MIN_BET × MIN_FAIR_PROB to find the config that
    maximises trade frequency while keeping positive EV.

    Candles are fetched ONCE and reused across all combos.
    """
    import sys as _sys
    sweep_mode = next((a for a in _sys.argv[1:] if not a.startswith("-")), "edge")

    # Pre-fetch candles based on sweep mode to avoid re-fetching per combo
    print(f"\nFetching {n_days} days of candle data (fetched once, reused for all combos)...")
    cached_candles: dict[str, list[dict]] = {}
    for coin, symbol in COINS.items():
        cached_candles[coin] = fetch_all_klines(symbol, n_days)

    # For 5m-based sweeps, also pre-fetch 5m candles for target coins
    cached_5m_target: dict[str, list[dict]] = {}
    if sweep_mode in ("5m_realistic", "risky"):
        print(f"  Also fetching 5m candles for target coins (cached for all combos)...")
        for coin, symbol in COINS_TARGET.items():
            try:
                cached_5m_target[coin] = fetch_all_klines(symbol, n_days, interval="5m")
            except Exception as e:
                print(f"  [skip] {coin}: {e}")

    if sweep_mode == "edge":
        # Sweep 1: MIN_EDGE (established — run for reference)
        max_pos_options   = [8]
        min_bet_options   = [200]
        min_fair_options  = [0.60]
        min_edge_options  = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]
        extra_keys: list[str] = []
        combos_raw = [
            {"max_positions": mp, "min_bet": mb, "min_fair_prob": mf, "min_edge": me}
            for mp in max_pos_options for mb in min_bet_options
            for mf in min_fair_options for me in min_edge_options
        ]
        label_fn = lambda c: f"edge≥{c['min_edge']:.0%}"

    elif sweep_mode == "kelly":
        # Sweep 2: Kelly fraction tiers + MAX_BET
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_kelly_mult": km, "_max_bet": mb}
            for km in [0.25, 0.50, 0.75, 1.00]   # multiplier on base quarter-Kelly
            for mb in [1500, 2500, 5000]           # $15, $25, $50 max bet
        ]
        label_fn = lambda c: f"Kelly×{c['_kelly_mult']:.2f} MaxBet=${c['_max_bet']//100}"

    elif sweep_mode == "positions":
        # Sweep 3: MAX_POSITIONS
        combos_raw = [
            {"max_positions": mp, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0}
            for mp in [3, 5, 8, 10, 15, 20]
        ]
        label_fn = lambda c: f"MaxPos={c['max_positions']}"

    elif sweep_mode == "longshot":
        # Sweep 4: Longshot gate — minimum entry price in cents
        # Lower = lets in more cheap contracts; higher = only high-priced, near-certain trades
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_longshot": ls}
            for ls in [5, 10, 15, 20, 25, 30]
        ]
        label_fn = lambda c: f"LongshotMin={c['_longshot']}¢"

    elif sweep_mode == "timing":
        # Sweep 5: Seconds remaining when we enter — simulates entering at different
        # points in the 15-min window.  900s = just opened; 300s = only 5min left.
        # Lower secs = tighter vol_horizon = stronger signal (closer to binary outcome).
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_secs": s}
            for s in [900, 750, 600, 450, 300, 180]
        ]
        label_fn = lambda c: f"SecsLeft={c['_secs']}s ({c['_secs']//60}m{c['_secs']%60:02d}s)"

    elif sweep_mode == "mkteff":
        # Sweep 6: Market efficiency assumption — how much Kalshi has already priced in our signal.
        # 100% = no edge; 50% = Kalshi is very slow.  Tests break-even point.
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_mkteff": e}
            for e in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        ]
        label_fn = lambda c: f"MktEff={int(c['_mkteff']*100)}%"

    elif sweep_mode == "volatility":
        # Sweep 7: Wide vol_scale sweep to find the 75% WR sweet spot.
        # High vol_scale → model less certain → lower entry prices → lower WR but more EV per $.
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.55, "min_edge": 0.0,
             "_vol_scale": vs}
            for vs in [0.60, 0.80, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 8.00]
        ]
        label_fn = lambda c: f"VolScale={c['_vol_scale']:.2f}x"

    elif sweep_mode == "winrate":
        # Target ~75% WR: vol_scale × min_fair_prob × secs_remaining grid
        # Higher vol_scale = less certain model = lower WR + lower entry prices = higher EV/dollar
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": fp, "min_edge": 0.0,
             "_vol_scale": vs, "_secs": sr}
            for vs in [2.0, 3.0, 4.0, 5.0]
            for fp in [0.55, 0.57, 0.60]
            for sr in [450, 900]
        ]
        label_fn = lambda c: f"vol={c['_vol_scale']:.1f}x fp={c['min_fair_prob']:.2f} s={c['_secs']}"

    elif sweep_mode == "optimal":
        # Sweep 8: 2D grid — entry timing × vol scale
        # Goal: maximize PnL while keeping max drawdown < $40 (40% of $100 starting capital)
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_secs": s, "_vol_scale": vs}
            for s  in [900, 600, 450, 300, 180]
            for vs in [1.00, 0.90, 0.80, 0.70, 0.60]
        ]
        label_fn = lambda c: f"secs={c['_secs']:>4}s vol={c['_vol_scale']:.2f}x"

    elif sweep_mode == "markets":
        # Sweep 9: Market type comparison — 15m directional vs 1h directional vs 1h range
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_market_type": mt, "_range_pct": rp}
            for mt, rp in [
                ("15m",      0.005),
                ("1h",       0.005),
                ("1h_range", 0.003),   # ±0.3% range (tight — more often NO)
                ("1h_range", 0.005),   # ±0.5% range (typical)
                ("1h_range", 0.010),   # ±1.0% range (wide — more often YES)
            ]
        ]
        label_fn = lambda c: f"{c['_market_type']} range±{c['_range_pct']*100:.1f}%" if "range" in c["_market_type"] else c["_market_type"]

    elif sweep_mode == "coins":
        # Sweep 10: Coin set comparison — base 3 vs target 7 (BTC/ETH/XRP/SOL/DOGE/BNB/HYPE) vs all 10
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_coin_set": cs, "_market_type": "15m", "_range_pct": 0.005}
            for cs in ["base", "target", "extended"]
        ]
        label_fn = lambda c: f"coins={c['_coin_set']}"

    elif sweep_mode == "all_markets":
        # Sweep 11: Full cross — all market types × coin sets
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": 0.60, "min_edge": 0.0,
             "_market_type": mt, "_coin_set": cs, "_range_pct": rp}
            for mt, rp in [("15m", 0.005), ("1h", 0.005)]
            for cs in ["base", "target", "extended"]
        ]
        label_fn = lambda c: f"{c['_market_type']} {c['_coin_set']:8s}"

    elif sweep_mode == "5m_realistic":
        # Sweep 12: 5m kline realistic simulation — genuine 75-80% WR
        combos_raw = [
            {"max_positions": 8, "min_bet": 200, "min_fair_prob": fp, "min_edge": 0.0,
             "_market_type": "5m_realistic", "_coin_set": cs, "_range_pct": 0.005}
            for cs in ["base", "target"]
            for fp in [0.55, 0.60, 0.65]
        ]
        label_fn = lambda c: f"5m coins={c['_coin_set']} fp={c['min_fair_prob']:.2f}"

    elif sweep_mode == "risky":
        # Sweep 13: Maximum profit mode — lower WR, cheaper entries, bigger Kelly, more positions
        # Axes: min_fair_prob × kelly_mult × max_positions × longshot_min
        combos_raw = [
            {"max_positions": mp, "min_bet": 200, "min_fair_prob": fp, "min_edge": 0.0,
             "_kelly_mult": km, "_longshot": ls, "_market_type": "5m_realistic",
             "_coin_set": "target", "_range_pct": 0.005}
            for fp  in [0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60]
            for km  in [1.0, 2.0, 3.0, 4.0]
            for mp  in [8, 15, 25]
            for ls  in [5, 10, 15]
        ]
        label_fn = lambda c: f"fp={c['min_fair_prob']:.2f} K×{c['_kelly_mult']:.1f} pos={c['max_positions']} ls={c['_longshot']}¢"

    else:
        raise ValueError(f"Unknown sweep mode. Use: edge | kelly | positions | longshot | timing | mkteff | volatility | optimal | markets | coins | all_markets | 5m_realistic | risky")

    combos = combos_raw

    print(f"\nRunning {len(combos)} parameter combinations...\n")
    header = f"{'Config':<28} {'Trades':>7} {'T/day':>6} {'WR':>7} {'PnL':>9} {'EV/T':>7} {'MaxDD':>7} {'DD%':>5} {'PF':>5} {'Sharpe':>7}"
    print(header)
    print("─" * len(header))

    best_pnl   = None
    best_combo = None

    for c in combos:
        _mtype     = c.get("_market_type", "15m")
        _coin_set  = c.get("_coin_set", "base")
        _coins_arg = (COINS_EXTENDED if _coin_set == "extended"
                      else COINS_TARGET if _coin_set == "target"
                      else None)
        # Reuse pre-fetched candle cache based on interval + coin set
        if _mtype == "15m" and _coin_set in ("base", None):
            _use_cache = cached_candles
        elif _mtype == "5m_realistic" and _coin_set == "target" and cached_5m_target:
            _use_cache = cached_5m_target
        else:
            _use_cache = None
        r = run_backtest(
            n_days=n_days,
            coin_candles=_use_cache,
            max_positions=c.get("max_positions", MAX_POSITIONS),
            min_bet=c.get("min_bet", MIN_BET),
            min_fair_prob=c.get("min_fair_prob", MIN_FAIR_PROB),
            min_edge=c.get("min_edge", MIN_EDGE),
            kelly_mult=c.get("_kelly_mult", 1.0),
            max_bet_override=c.get("_max_bet"),
            longshot_min=c.get("_longshot", LONGSHOT_MIN),
            secs_remaining_override=c.get("_secs"),
            mkteff_override=c.get("_mkteff"),
            vol_scale=c.get("_vol_scale", 1.0),
            market_type=_mtype,
            coins_override=_coins_arg,
            range_pct=c.get("_range_pct", 0.005),
        )
        lbl = label_fn(c)
        if not r:
            print(f"  {lbl:<28}  NO TRADES")
            continue

        total  = r["total_trades"]
        wr     = r["win_rate"]
        pnl    = r["total_pnl_cents"] / 100
        ev_t   = pnl / total if total else 0
        max_dd = r["max_drawdown_cents"] / 100
        pf     = r["profit_factor"]
        tpd    = total / n_days
        sharpe = r["ann_sharpe"]

        marker = ""
        if best_pnl is None or pnl > best_pnl:
            best_pnl   = pnl
            best_combo = c
            marker = " ◄ best"

        dd_pct = max_dd / (START_BAL / 100) * 100  # drawdown as % of starting $100
        dd_flag = " ⚠" if dd_pct > 40 else ""
        print(f"  {lbl:<28} {total:>6}  {tpd:>5.1f}  {wr:>6.1%}  ${pnl:>+8.2f}  ${ev_t:>+5.3f}  ${max_dd:>5.2f}  {dd_pct:>4.0f}%{dd_flag}  {pf:>4.2f}x  {sharpe:>6.1f}{marker}")

    print()
    if best_combo:
        lbl = label_fn(best_combo)
        print(f"  Best: {lbl}")
        print()
        print("  Running full report for best config...")
        _bmtype    = best_combo.get("_market_type", "15m")
        _bcs       = best_combo.get("_coin_set", "base")
        _bcoins    = (COINS_EXTENDED if _bcs == "extended"
                      else COINS_TARGET if _bcs == "target"
                      else None)
        if _bmtype == "15m" and _bcs in ("base", None):
            _bcache = cached_candles
        elif _bmtype == "5m_realistic" and _bcs == "target" and cached_5m_target:
            _bcache = cached_5m_target
        else:
            _bcache = None
        r = run_backtest(
            n_days=n_days, coin_candles=_bcache,
            max_positions=best_combo.get("max_positions", MAX_POSITIONS),
            min_bet=best_combo.get("min_bet", MIN_BET),
            min_fair_prob=best_combo.get("min_fair_prob", MIN_FAIR_PROB),
            min_edge=best_combo.get("min_edge", MIN_EDGE),
            kelly_mult=best_combo.get("_kelly_mult", 1.0),
            max_bet_override=best_combo.get("_max_bet"),
            longshot_min=best_combo.get("_longshot", LONGSHOT_MIN),
            secs_remaining_override=best_combo.get("_secs"),
            mkteff_override=best_combo.get("_mkteff"),
            vol_scale=best_combo.get("_vol_scale", 1.0),
            market_type=_bmtype,
            coins_override=_bcoins,
            range_pct=best_combo.get("_range_pct", 0.005),
        )
        print_report(r, label=lbl)


if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        run_sensitivity(n_days=90)
    else:
        # Single run — 7-coin target, combined 15m+1h+1h_range, $172 starting, proportional Kelly
        print(f"\n{'='*64}")
        print(f"  CRYPTO QUANT BACKTEST  (90 days, combined markets, 7 coins)")
        print(f"  Starting balance: ${START_BAL/100:.2f}  |  Kelly proportional to balance")
        print(f"  Per-coin thresholds: BNB≥75%, HYPE≥72%, others≥60%")
        print(f"{'='*64}\n")
        result = run_backtest(
            n_days=90,
            coins_override=COINS_TARGET,
            market_type="combined",
            range_pcts=[0.005, 0.010, 0.020],  # ±0.5%, ±1%, ±2% range bands
        )
        print_report(result)
