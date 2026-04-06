"""
Claude Code Market Scanner — full predictionbot analysis for FREE.

Pulls RF model signals + raw market data from your deployed Railway backend,
then formats everything for Claude Code to analyze with web search. Claude Code
runs on your Max subscription — zero API cost.

This script does TWO things:
1. Triggers an RF model scan on the backend (POST /api/scan) to get model signals,
   near misses, and worst opportunities
2. Fetches raw market data for the top 100 candidates so Claude Code can do
   its own independent research with web search

Usage (just tell Claude Code "scan markets"):
    python scan_claude_code.py                     # Full scan, top 100 for analysis
    python scan_claude_code.py --limit 50          # Fewer candidates
    python scan_claude_code.py --category politics  # Filter by category
    python scan_claude_code.py --ticker KXBTCD     # Deep-dive one market
    python scan_claude_code.py --signals-only      # Just show RF signals, skip raw data
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import httpx

from bot.models import Event, Market

RAILWAY_URL = "https://prediction-market-bot-railway-production.up.railway.app"
KALSHI_MARKET_URL = "https://kalshi.com/markets"


def score_market(market: Market, event: Event) -> float:
    """Microstructure pre-filter score."""
    if market.status not in ("open", "active"):
        return -1.0
    if market.volume < 10:
        return -1.0

    mid = market.mid_price_yes
    spread = market.spread

    spread_score = max(0, 1.0 - spread / 15)
    price_score = 1.0 - abs(mid - 50) / 50
    volume_score = min(market.volume / 5000, 1.0)

    time_score = 0.5
    if market.close_time:
        try:
            close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
            days_left = (close_dt - datetime.now(timezone.utc)).total_seconds() / 86400
            if 1 <= days_left <= 30:
                time_score = 1.0
            elif days_left < 1:
                time_score = 0.3
            else:
                time_score = 0.6
        except Exception:
            pass

    return (
        spread_score * 0.30
        + price_score * 0.25
        + volume_score * 0.25
        + time_score * 0.20
    )


def kalshi_link(ticker: str) -> str:
    """Generate a Kalshi market link from a ticker."""
    # Kalshi URLs use the event ticker (everything before the last hyphen segment)
    # e.g. KXMARMADROUND-26S16-SJU -> kalshi.com/markets/kxmarmadround-26s16/...
    # But the simplest approach: use the search URL
    return f"{KALSHI_MARKET_URL}#markets={ticker}"


def fetch_events(backend_url: str) -> list[Event]:
    """Fetch events from backend."""
    resp = httpx.get(f"{backend_url}/api/events", params={"all": "true"}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [Event(**e) for e in data["events"]]


def trigger_rf_scan(backend_url: str) -> dict:
    """Trigger an RF model scan on the backend and return results."""
    resp = httpx.post(
        f"{backend_url}/api/scan",
        json={"use_ai": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def get_status(backend_url: str) -> dict:
    """Get bot status including model training state."""
    resp = httpx.get(f"{backend_url}/api/status", timeout=30)
    resp.raise_for_status()
    return resp.json()


def time_remaining_str(close_time: str) -> str:
    """Human-readable time remaining."""
    if not close_time:
        return "Unknown"
    try:
        close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        delta = close_dt - datetime.now(timezone.utc)
        if delta.days > 0:
            return f"{delta.days}d {delta.seconds // 3600}h"
        elif delta.seconds > 3600:
            return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
        elif delta.seconds > 0:
            return f"{delta.seconds // 60}m"
        return "Expired"
    except Exception:
        return "Unknown"


def format_signal(sig: dict, rank: int) -> str:
    """Format a signal from the RF model for display."""
    edge_pct = sig.get("edge", 0) * 100
    conf = sig.get("confidence", 0) * 100
    fair = sig.get("fair_probability", 0) * 100
    mkt = sig.get("market_probability", 0) * 100
    side = sig.get("side", "?").upper()
    quality = sig.get("signal_quality", 0)
    size = sig.get("recommended_size_cents", 0)
    ticker = sig.get("ticker", "")
    title = sig.get("market_title", "")
    cat = sig.get("category", "")
    reasoning = sig.get("reasoning", "")

    return (
        f"### {rank}. {ticker}\n"
        f"**{title}**\n"
        f"Category: {cat} | Side: **{side}** | Edge: **{edge_pct:+.1f}%** | "
        f"Confidence: {conf:.0f}% | Quality: {quality:.3f}\n"
        f"Fair: {fair:.1f}% vs Market: {mkt:.1f}% | "
        f"Size: ${size/100:.2f} | EV: {edge_pct * conf / 100:.2f}%\n"
        f"Reasoning: {reasoning[:200]}\n"
        f"Bet: {kalshi_link(ticker)}\n"
    )


def format_near_miss(nm: dict, rank: int) -> str:
    """Format a near miss."""
    edge = nm.get("edge", 0) * 100
    conf = nm.get("confidence", 0) * 100
    side = nm.get("side", "?").upper()
    reason = nm.get("reason", "")
    ticker = nm.get("ticker", "")
    title = nm.get("market_title", "")

    return (
        f"{rank}. **{ticker}** — {title}\n"
        f"   {side} | Edge: {edge:+.1f}% | Conf: {conf:.0f}% | "
        f"Blocked: {reason} | {kalshi_link(ticker)}\n"
    )


def format_worst(w: dict, rank: int) -> str:
    """Format a worst opportunity (overpriced market)."""
    edge = w.get("edge", 0) * 100
    conf = w.get("confidence", 0) * 100
    fair = w.get("fair_probability", 0) * 100
    mkt = w.get("market_probability", 0) * 100
    ticker = w.get("ticker", "")
    title = w.get("market_title", "")

    return (
        f"{rank}. **{ticker}** — {title}\n"
        f"   Overpriced by {abs(edge):.1f}% | Fair: {fair:.1f}% vs Market: {mkt:.1f}% | "
        f"Conf: {conf:.0f}% | {kalshi_link(ticker)}\n"
    )


def format_market_for_analysis(event: Event, market: Market, rank: int, rf_data: dict | None = None) -> str:
    """Format a market with full data for Claude Code deep analysis."""
    mid = market.mid_price_yes
    spread = market.spread
    volume = market.volume
    oi = market.open_interest
    tr = time_remaining_str(market.close_time)

    microprice = mid
    if market.yes_bid > 0 and market.yes_ask > 0:
        microprice = (market.yes_bid + market.yes_ask) / 2

    last_price = market.last_price if market.last_price else mid
    prev_price = market.prev_price if market.prev_price else mid
    price_change = last_price - prev_price

    lines = [
        f"---",
        f"### Candidate #{rank}: {market.ticker}",
        f"**Event:** {event.title}",
        f"**Market:** {market.title}",
        f"**Category:** {market.category or event.category or 'Other'}",
        f"**Closes:** {market.close_time or 'N/A'} ({tr} remaining)",
        f"**Bet link:** {kalshi_link(market.ticker)}",
        f"",
        f"YES mid={mid:.0f}c (bid={market.yes_bid}c / ask={market.yes_ask}c) | "
        f"NO mid={100-mid:.0f}c | Spread={spread}c",
        f"Microprice: {microprice:.1f}c | Last={last_price:.0f}c | "
        f"Prev={prev_price:.0f}c | Change={price_change:+.1f}c",
        f"Volume: {volume:,} | OI: {oi:,} | Turnover: {volume/max(oi,1):.2f}x",
    ]

    if rf_data:
        lines.append(f"**RF Model:** prob={rf_data.get('model_prob', 0):.1f}% | "
                      f"edge={rf_data.get('edge', 0)*100:+.1f}% | "
                      f"conf={rf_data.get('confidence', 0)*100:.0f}% | "
                      f"side={rf_data.get('side', '?').upper()}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Full predictionbot scanner for Claude Code analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend-url", default=RAILWAY_URL, help=f"Backend URL (default: Railway)")
    parser.add_argument("--limit", type=int, default=100, help="Number of candidates for Claude analysis (default: 100)")
    parser.add_argument("--top", type=int, default=10, help="Number of top signals to highlight (default: 10)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--ticker", type=str, default=None, help="Deep-dive a specific ticker")
    parser.add_argument("--signals-only", action="store_true", help="Only show RF signals, skip raw candidates")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    backend = args.backend_url
    print(f"# Prediction Bot Scan — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", file=sys.stderr)
    print(f"Backend: {backend}", file=sys.stderr)

    # Step 1: Check bot status
    try:
        status = get_status(backend)
        model_trained = status.get("paper_trader", {}).get("model_trained", False)
        auto_scan = status.get("auto_scan_enabled", False)
        print(f"Model trained: {model_trained} | Auto-scan: {auto_scan}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: Could not get status: {e}", file=sys.stderr)
        model_trained = False

    # Step 2: Trigger RF scan
    scan_result = None
    try:
        print("Triggering RF model scan...", file=sys.stderr)
        scan_result = trigger_rf_scan(backend)
        n_signals = scan_result.get("rf_signals", 0)
        n_events = scan_result.get("events_scanned", 0)
        n_markets = scan_result.get("markets_scanned", 0)
        duration = scan_result.get("duration_ms", 0)
        print(f"Scan complete: {n_events} events, {n_markets} markets, "
              f"{n_signals} signals in {duration:.0f}ms", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: RF scan failed: {e}", file=sys.stderr)

    # Step 3: Fetch raw market data for Claude Code analysis
    events = []
    try:
        print("Fetching raw market data...", file=sys.stderr)
        events = fetch_events(backend)
        total = sum(len(e.markets) for e in events)
        print(f"Fetched {len(events)} events, {total} markets", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Could not fetch events: {e}", file=sys.stderr)
        if not scan_result:
            sys.exit(1)

    # Build signal lookup for RF data enrichment
    rf_signal_map = {}
    if scan_result:
        for sig in scan_result.get("signals", []):
            rf_signal_map[sig["ticker"]] = sig
        for nm in scan_result.get("near_misses", []):
            rf_signal_map[nm["ticker"]] = nm

    # JSON mode
    if args.json:
        json.dump({
            "scan_result": scan_result,
            "model_trained": model_trained,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, sys.stdout, indent=2)
        return

    # ── Output Report ──

    print(f"# PREDICTION BOT SCAN REPORT")
    print(f"**{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}** | "
          f"Model: {'TRAINED' if model_trained else 'NOT TRAINED'}")
    print()

    # Section 1: Top RF Signals
    signals = scan_result.get("signals", []) if scan_result else []
    signals.sort(key=lambda s: s.get("edge", 0) * s.get("confidence", 0), reverse=True)

    print(f"## TOP {min(args.top, len(signals))} RF MODEL SIGNALS")
    print(f"These passed all filters (edge >= 3%, confidence >= 65%, quality threshold).\n")
    if signals:
        for i, sig in enumerate(signals[:args.top], 1):
            print(format_signal(sig, i))
    else:
        print("No signals passed all filters this scan.\n")

    # Section 2: Near Misses (close but didn't qualify)
    near_misses = scan_result.get("near_misses", []) if scan_result else []
    if near_misses:
        print(f"\n## NEAR MISSES ({len(near_misses)} markets with 3%+ edge but blocked)")
        print("These have edge but failed a secondary filter — worth watching.\n")
        for i, nm in enumerate(near_misses[:10], 1):
            print(format_near_miss(nm, i))

    # Section 3: Worst Opportunities (overpriced)
    worst = scan_result.get("worst_opportunities", []) if scan_result else []
    if worst:
        print(f"\n## WORST OPPORTUNITIES ({len(worst)} overpriced markets)")
        print("Model sees these as significantly overpriced — potential NO plays.\n")
        for i, w in enumerate(worst[:10], 1):
            print(format_worst(w, i))

    if args.signals_only:
        return

    # Section 4: Raw candidates for Claude Code deep analysis
    candidates = []
    for event in events:
        for market in event.markets:
            if args.category and (market.category or event.category or "").lower() != args.category.lower():
                continue
            if args.ticker and market.ticker != args.ticker.upper():
                continue
            score = score_market(market, event)
            if score > 0:
                candidates.append((event, market, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    top_candidates = candidates[:args.limit]

    print(f"\n## RAW MARKET DATA FOR CLAUDE CODE ANALYSIS")
    print(f"Top {len(top_candidates)} candidates by microstructure score (spread, volume, price, time).")
    print(f"Claude Code: research each market with web search, estimate fair probability,")
    print(f"identify mispricings, and select the best 10 to report.\n")

    for i, (event, market, score) in enumerate(top_candidates, 1):
        rf_data = rf_signal_map.get(market.ticker)
        print(format_market_for_analysis(event, market, i, rf_data))

    # Summary footer
    total_candidates = len(candidates)
    print(f"\n---")
    print(f"**Scan complete.** {len(signals)} RF signals | {len(near_misses)} near misses | "
          f"{len(worst)} worst | {len(top_candidates)}/{total_candidates} candidates shown.")
    print(f"\nClaude Code: analyze the candidates above using web search, then report your TOP 10 with:")
    print(f"1. Ticker + market title")
    print(f"2. Your fair probability vs market price")
    print(f"3. Edge (%) and side (YES/NO)")
    print(f"4. Confidence level and reasoning")
    print(f"5. Whether RF model agrees")
    print(f"6. Recommended bet size")
    print(f"7. Kalshi bet link")


if __name__ == "__main__":
    main()
