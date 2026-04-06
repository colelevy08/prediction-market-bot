"""
Quant model backtest for BTC/ETH/XRP 15-minute Kalshi markets.

Fetches settled crypto markets directly from /events API (no per-market trade
fetches), simulates compute_fair_probability using market last_price as the
implied probability, and measures actual win rate at various fair_prob thresholds.

This answers: "when the quant model says ≥67%, how often does YES win?"
"""

import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))

from statistics import NormalDist
from collections import defaultdict
from bot.kalshi_client import KalshiClient

# ── Constants from crypto_feed.py ────────────────────────────────────────────
_COIN_VOL_PER_MIN = {
    "bitcoin":  0.00070,
    "ethereum": 0.00090,
    "ripple":   0.00120,
}

CRYPTO_PREFIXES = ("KXBTC15M", "KXETH15M", "KXXRP15M")

def ticker_to_coin(ticker: str) -> str:
    if ticker.startswith("KXBTC15M"): return "bitcoin"
    if ticker.startswith("KXETH15M"): return "ethereum"
    if ticker.startswith("KXXRP15M"): return "ripple"
    return ""

def quant_fair_prob(floor_strike: float, coin: str, secs_left: float,
                    price_pct_above: float) -> float:
    """Compute fair prob given known price distance from strike."""
    vol_per_min = _COIN_VOL_PER_MIN.get(coin, 0.00080)
    time_min = max(secs_left / 60.0, 0.5)
    vol_remaining = vol_per_min * math.sqrt(time_min)
    if vol_remaining <= 0:
        return 1.0 if price_pct_above > 0 else 0.0
    z = price_pct_above / vol_remaining
    return NormalDist().cdf(z)

def market_price_to_implied_pct_above(market_prob: float, coin: str, secs_left: float) -> float:
    """Invert: given market probability, what % above/below strike is implied?"""
    vol_per_min = _COIN_VOL_PER_MIN.get(coin, 0.00080)
    time_min = max(secs_left / 60.0, 0.5)
    vol_remaining = vol_per_min * math.sqrt(time_min)
    z = NormalDist().inv_cdf(max(0.001, min(0.999, market_prob)))
    return z * vol_remaining

def fetch_crypto_markets(kalshi: KalshiClient, max_pages: int = 50) -> list[dict]:
    """
    Fetch settled crypto 15m markets directly from /events API.
    Returns flat dicts with ticker, result, floor_strike, last_price.
    Much faster than HistoricalDataFetcher — no per-market trade history fetches.
    """
    markets = []
    cursor = None
    pages_fetched = 0

    for _ in range(max_pages):
        params = {
            "limit": 200,
            "status": "settled",
            "with_nested_markets": "true",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            data = kalshi._request("GET", "/events", params=params)
        except Exception as e:
            print(f"API error: {e}", flush=True)
            time.sleep(2)
            continue

        events = data.get("events", [])
        if not events:
            break

        for ev in events:
            for m in ev.get("markets", []):
                ticker = m.get("ticker", "")
                if not any(ticker.startswith(p) for p in CRYPTO_PREFIXES):
                    continue
                result = m.get("result", "")
                if result not in ("yes", "no"):
                    continue

                # last_price in dollars (0.0–1.0) or cents (0–100)?
                # Kalshi v2 returns last_price_dollars as a string decimal
                lp_dollars = m.get("last_price_dollars") or m.get("last_price") or 0
                try:
                    lp_dollars = float(lp_dollars)
                except (TypeError, ValueError):
                    lp_dollars = 0

                # Convert to cents if needed
                if 0 < lp_dollars <= 1.0:
                    last_price_c = int(round(lp_dollars * 100))
                elif 1 < lp_dollars <= 100:
                    last_price_c = int(round(lp_dollars))
                else:
                    continue

                if last_price_c <= 0 or last_price_c >= 100:
                    continue

                # floor_strike
                fs = m.get("floor_strike") or m.get("strike_type") or 0
                try:
                    fs = float(fs)
                except (TypeError, ValueError):
                    fs = 0.0

                markets.append({
                    "ticker": ticker,
                    "result": result,
                    "floor_strike": fs,
                    "last_price": last_price_c,
                })

        cursor = data.get("cursor")
        pages_fetched += 1
        if not cursor:
            break

        print(f"  fetched {len(markets)} crypto markets so far (page {pages_fetched})...", flush=True)
        time.sleep(0.3)  # gentle rate limiting

    return markets

def run_backtest():
    kalshi = KalshiClient()

    print("Fetching settled BTC/ETH/XRP 15m markets (direct API, fast)...", flush=True)
    crypto = fetch_crypto_markets(kalshi, max_pages=100)

    print(f"Found {len(crypto)} crypto 15m settled markets with results", flush=True)

    if not crypto:
        print("No crypto markets found — check API connection or tickers")
        return

    # --- Market calibration (market price → actual win rate) ---
    buckets = defaultdict(lambda: {"yes": 0, "no": 0})
    for m in crypto:
        lp = m["last_price"]
        result = m["result"]
        b = (lp // 10) * 10
        buckets[b][result] += 1

    print("\n=== Market Calibration (Kalshi price → actual win rate) ===")
    print(f"{'Bucket':>10} {'YES':>6} {'NO':>6} {'Total':>7} {'Win%':>7}")
    for b in sorted(buckets):
        d = buckets[b]
        total = d["yes"] + d["no"]
        pct = d["yes"] / total * 100 if total else 0
        print(f"  {b:3d}–{b+9:3d}c  {d['yes']:>6} {d['no']:>6} {total:>7}  {pct:>6.1f}%")

    # --- Quant model simulation ---
    S_MIN_FAIR = 0.67
    S_MIN_FAIR_NO = 0.33
    S_MIN_EDGE = 0.12
    KELLY_FRACTION = 0.25
    FEE_PCT = 0.07
    INITIAL_BALANCE = 5000  # cents ($50)
    MIN_BET = 200
    MAX_BET = 1500

    balance = INITIAL_BALANCE
    trades = []
    skipped_no_data = 0

    # Assume typical entry at ~10 min remaining (600s) when signal fires
    ASSUMED_SECS_LEFT = 600.0

    for m in crypto:
        coin = ticker_to_coin(m["ticker"])
        if not coin:
            continue
        lp = m["last_price"]
        if lp <= 0:
            skipped_no_data += 1
            continue

        market_prob = lp / 100.0
        implied_pct = market_price_to_implied_pct_above(market_prob, coin, ASSUMED_SECS_LEFT)
        fair_prob = quant_fair_prob(m["floor_strike"] or 1, coin, ASSUMED_SECS_LEFT, implied_pct)
        edge = fair_prob - market_prob
        result = m["result"]

        if fair_prob >= S_MIN_FAIR and edge >= S_MIN_EDGE:
            side = "yes"
            win = (result == "yes")
        elif fair_prob <= S_MIN_FAIR_NO and edge <= -S_MIN_EDGE:
            side = "no"
            win = (result == "no")
        else:
            continue

        win_prob = fair_prob if side == "yes" else (1.0 - fair_prob)
        price_c = int(market_prob * 100) if side == "yes" else int((1 - market_prob) * 100)
        price_c = max(1, min(70, price_c))
        payout = 100 - price_c
        kelly_raw = (win_prob * payout - (1 - win_prob) * price_c) / payout
        kelly_raw = max(0, kelly_raw) * KELLY_FRACTION
        bet_cents = int(balance * kelly_raw)
        bet_cents = max(MIN_BET, min(MAX_BET, bet_cents))
        if bet_cents > balance:
            bet_cents = balance

        contracts = max(1, bet_cents // price_c)
        cost = price_c * contracts

        if win:
            gross = contracts * payout
            net = int(gross * (1 - FEE_PCT))
            balance += net
            pnl = net
        else:
            balance -= cost
            pnl = -cost

        trades.append({
            "ticker": m["ticker"],
            "side": side,
            "fair": round(fair_prob, 3),
            "mkt": round(market_prob, 3),
            "edge": round(edge, 3),
            "win": win,
            "pnl": pnl,
            "balance": balance,
        })

    if not trades:
        print(f"\nNo qualifying trades found at thresholds {S_MIN_FAIR}/{S_MIN_FAIR_NO} edge≥{S_MIN_EDGE}")
        print(f"(Skipped {skipped_no_data} markets with no price data)")
        return

    wins = sum(1 for t in trades if t["win"])
    losses = len(trades) - wins
    total_pnl = balance - INITIAL_BALANCE
    win_rate = wins / len(trades) * 100

    print(f"\n=== Quant Model Backtest (threshold {S_MIN_FAIR}/{S_MIN_FAIR_NO}, edge≥{S_MIN_EDGE}) ===")
    print(f"Markets analyzed:  {len(crypto)}")
    print(f"Trades triggered:  {len(trades)} ({len(trades)/len(crypto)*100:.1f}% hit rate)")
    print(f"Win / Loss:        {wins} / {losses}")
    print(f"Win rate:          {win_rate:.1f}%")
    print(f"Starting balance:  ${INITIAL_BALANCE/100:.2f}")
    print(f"Ending balance:    ${balance/100:.2f}")
    print(f"Total P&L:         ${total_pnl/100:+.2f}")
    print(f"Return:            {total_pnl/INITIAL_BALANCE*100:+.1f}%")

    print(f"\n=== Win Rate by Fair Prob at Entry ===")
    thresh_buckets = defaultdict(lambda: {"w":0,"l":0})
    for t in trades:
        b = round(t["fair"] * 10) / 10
        if t["win"]: thresh_buckets[b]["w"] += 1
        else: thresh_buckets[b]["l"] += 1
    for b in sorted(thresh_buckets):
        d = thresh_buckets[b]
        tot = d["w"] + d["l"]
        print(f"  fair≈{b:.1f}  {d['w']}W {d['l']}L  {d['w']/tot*100:.0f}% WR")

    print(f"\n=== Sample Trades ===")
    for t in trades[:10]:
        print(f"  {t['ticker'][:30]:30s} {t['side']:3s} fair={t['fair']} mkt={t['mkt']} edge={t['edge']:+.3f}  {'WIN' if t['win'] else 'LOSS':4s}  P&L=${t['pnl']/100:+.2f}")

if __name__ == "__main__":
    run_backtest()
