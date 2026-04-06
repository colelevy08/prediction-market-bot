"""
Local market scanner — run AI analysis using your Claude Max subscription.

Instead of burning Anthropic API credits through the bot's built-in analyzer,
this script fetches markets from your running backend, pre-filters them with
the RF model's microstructure scoring, then calls Claude via the Anthropic
Python client (using ANTHROPIC_API_KEY from your environment). This way you
use the same analysis prompts and logic as the bot, but the API calls go
against your Max subscription quota instead of pay-per-token billing.

Usage:
    python scan_local.py                       # Full scan, analyze top 20
    python scan_local.py --limit 5             # Analyze top 5 only
    python scan_local.py --dry-run             # Show candidates without calling Claude
    python scan_local.py --dry-run --limit 50  # Preview top 50 candidates
    python scan_local.py --backend-url http://my-server:8000
    python scan_local.py --no-push             # Don't POST results to backend
    python scan_local.py --model claude-sonnet-4-6  # Use a specific model

Requires:
    - Backend running (default http://localhost:8000) with Kalshi connected
    - ANTHROPIC_API_KEY set in environment or .env
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import anthropic
import httpx
from dotenv import load_dotenv

from bot.models import Event, Market, Side, TradingSignal
from bot.analyzer import DATA_DRIVEN_PROMPT

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Microstructure pre-filter (same logic as MarketAnalyzer._pre_filter_markets)
# ---------------------------------------------------------------------------

def score_market(market: Market, event: Event) -> float:
    """Score a market by microstructure features for pre-filtering.

    This replicates the Pass 1 logic from MarketAnalyzer._pre_filter_markets
    so we don't need an RF model loaded locally.
    """
    if market.status not in ("open", "active"):
        return -1.0
    if market.volume < 10:
        return -1.0

    mid = market.mid_price_yes
    spread = market.spread

    spread_score = max(0, 1.0 - spread / 15)  # Tighter spread = better
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


# ---------------------------------------------------------------------------
# Data collection (simplified version of MarketAnalyzer._collect_market_data)
# ---------------------------------------------------------------------------

def collect_market_data(event: Event, market: Market) -> dict:
    """Collect quantitative data for a market to fill the analysis prompt.

    This is a standalone version of MarketAnalyzer._collect_market_data that
    works without the RF model or history cache. Fields that require the RF
    model get sensible defaults.
    """
    yes_mid = market.mid_price_yes
    no_mid = 100 - yes_mid
    spread = market.spread
    volume = market.volume
    oi = market.open_interest

    # Time remaining
    time_remaining = "Unknown"
    days_left = 30.0
    hours_left = 720.0
    if market.close_time:
        try:
            close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
            delta = close_dt - datetime.now(timezone.utc)
            days_left = max(0, delta.total_seconds() / 86400)
            hours_left = max(0, delta.total_seconds() / 3600)
            if delta.days > 0:
                time_remaining = f"{delta.days}d {delta.seconds // 3600}h"
            elif delta.seconds > 3600:
                time_remaining = f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
            elif delta.seconds > 0:
                time_remaining = f"{delta.seconds // 60}m"
            else:
                time_remaining = "Expired"
        except Exception:
            pass

    # Without RF model, default to market price as model estimate
    model_prob = yes_mid / 100
    model_confidence = max(model_prob, 1 - model_prob) * 100
    prediction_std = 15.0
    model_edge = 0.0
    model_side = "YES" if model_prob > 0.5 else "NO"

    # Orderbook (simplified — no RF analyzer)
    ob_imbalance = 0.0
    bid_pressure = 0.0
    ask_pressure = 0.0
    ob_signal = "neutral"

    # Microprice
    microprice = yes_mid
    if market.yes_bid > 0 and market.yes_ask > 0:
        microprice = (market.yes_bid + market.yes_ask) / 2

    # Price change
    last_price = market.last_price if market.last_price else yes_mid
    prev_price = market.prev_price if market.prev_price else yes_mid
    price_change = last_price - prev_price

    dollar_volume = volume * yes_mid / 100
    turnover = volume / max(oi, 1)
    volume_intensity = volume / max(hours_left, 1)
    liquidity_score = volume * (100 - spread) / 100 if spread <= 100 else 0

    # Quant signals summary
    quant_lines = []
    if spread > 5:
        quant_lines.append(f"- Wide spread ({spread}c) suggests disagreement or low liquidity — potential inefficiency")
    if volume < 100:
        quant_lines.append(f"- Very low volume ({volume}) — market may be inefficiently priced")
    elif volume > 10000:
        quant_lines.append(f"- High volume ({volume:,}) — market is well-studied, edge harder to find")
    if days_left < 1:
        quant_lines.append(f"- Expiring in {time_remaining} — time pressure increases volatility")
    elif days_left < 7:
        quant_lines.append(f"- Expiring soon ({time_remaining}) — convergence pressure building")

    return {
        "event_title": event.title,
        "market_title": market.title,
        "category": market.category or event.category or "Other",
        "close_time": market.close_time or "Unknown",
        "time_remaining": time_remaining,
        "yes_bid": market.yes_bid,
        "yes_ask": market.yes_ask,
        "yes_mid": yes_mid,
        "no_bid": market.no_bid,
        "no_ask": market.no_ask,
        "no_mid": no_mid,
        "spread": spread,
        "spread_pct": (spread / max(yes_mid, 1)) * 100,
        "last_price": last_price,
        "prev_price": prev_price,
        "price_change": price_change,
        "volume": volume,
        "dollar_volume": dollar_volume,
        "open_interest": oi,
        "turnover": turnover,
        "liquidity_score": liquidity_score,
        "volume_intensity": volume_intensity,
        "bid_pressure": bid_pressure,
        "ask_pressure": ask_pressure,
        "ob_imbalance": ob_imbalance,
        "ob_signal": ob_signal,
        "microprice": microprice,
        "microprice_edge": microprice - yes_mid,
        "model_prob": model_prob * 100,
        "market_prob": yes_mid,
        "model_edge": model_edge,
        "model_side": model_side,
        "model_confidence": model_confidence,
        "prediction_std": prediction_std,
        "n_snapshots": 0,
        "momentum_summary": "- No history available (local scan without RF model)",
        "quant_signals": "\n".join(quant_lines) if quant_lines else "- No strong quantitative signals detected",
    }


# ---------------------------------------------------------------------------
# Claude analysis
# ---------------------------------------------------------------------------

def analyze_market(
    client: anthropic.Anthropic,
    event: Event,
    market: Market,
    model: str,
) -> dict | None:
    """Send market data to Claude and parse the structured JSON response."""
    data = collect_market_data(event, market)
    prompt = DATA_DRIVEN_PROMPT.format(**data)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=8000,
            temperature=0.2,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        text = ""
        search_count = 0
        for block in response.content:
            if hasattr(block, "type") and block.type == "server_tool_use":
                search_count += 1
            if hasattr(block, "text") and block.text.strip():
                text = block.text.strip()

        if not text:
            print(f"  [!] No text response ({search_count} web searches performed)")
            return None

        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        analysis = json.loads(text)
        analysis["_search_count"] = search_count
        analysis["_market_data"] = data
        return analysis

    except json.JSONDecodeError as e:
        print(f"  [!] Failed to parse JSON response: {e}")
        print(f"      Raw text: {text[:200]}...")
        return None
    except anthropic.APIError as e:
        print(f"  [!] Anthropic API error: {e}")
        return None


def build_signal_from_analysis(
    analysis: dict, market: Market, data: dict
) -> dict:
    """Convert a Claude analysis response into a signal dict for the backend."""
    fair_prob = max(0.01, min(0.99, float(analysis["fair_probability"])))
    market_prob = data["yes_mid"] / 100
    side = analysis["side"]

    if side == "yes":
        edge = fair_prob - market_prob
    else:
        edge = (1 - fair_prob) - (1 - market_prob)

    confidence = max(0.0, min(1.0, float(analysis["confidence"])))

    # Quarter-Kelly sizing (same as MarketAnalyzer._calculate_size)
    max_bet = 2500  # $25 default
    if edge > 0 and confidence >= 0.5:
        kelly = edge * confidence
        size = int(max_bet * kelly * 0.25)
        size = max(0, min(size, max_bet))
    else:
        size = 0

    reasoning_parts = [analysis.get("reasoning", "")]
    if analysis.get("key_factors"):
        reasoning_parts.append(f"Key: {'; '.join(analysis['key_factors'][:3])}")
    if analysis.get("risk_factors"):
        reasoning_parts.append(f"Risks: {'; '.join(analysis['risk_factors'][:3])}")
    if not analysis.get("agrees_with_model", True):
        reasoning_parts.append("AI DISAGREES with ML model")
    full_reasoning = " | ".join(reasoning_parts)

    return {
        "ticker": market.ticker,
        "market_title": market.title,
        "side": side,
        "confidence": confidence,
        "fair_probability": fair_prob,
        "market_probability": market_prob,
        "edge": edge,
        "reasoning": full_reasoning,
        "recommended_size_cents": size,
        "source": "claude_ai_local",
        "risk_check": {
            "allowed": edge >= 0.03 and confidence >= 0.45,
            "reason": "local_scan" if edge >= 0.03 and confidence >= 0.45 else "below_threshold",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_events(backend_url: str, fetch_all: bool = True) -> list[Event]:
    """Fetch events from the running backend."""
    url = f"{backend_url}/api/events"
    params = {"all": "true"} if fetch_all else {}
    print(f"Fetching events from {url} ...")
    resp = httpx.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    events = [Event(**e) for e in data["events"]]
    total_markets = sum(len(e.markets) for e in events)
    print(f"  Got {len(events)} events with {total_markets} total markets")
    return events


def prefilter(events: list[Event], top_n: int = 100) -> list[tuple[Event, Market, float]]:
    """Score and rank all open markets, return top N candidates."""
    candidates = []
    for event in events:
        for market in event.markets:
            score = score_market(market, event)
            if score > 0:
                candidates.append((event, market, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    result = candidates[:top_n]
    print(f"  Pre-filter: {len(candidates)} eligible markets -> top {len(result)} candidates")
    if result:
        print(f"  Score range: {result[0][2]:.3f} (best) to {result[-1][2]:.3f} (cutoff)")
    return result


def print_dry_run(candidates: list[tuple[Event, Market, float]]):
    """Print candidate markets without calling Claude."""
    print(f"\n{'='*80}")
    print(f"DRY RUN — {len(candidates)} candidates (no Claude calls)")
    print(f"{'='*80}\n")

    for i, (event, market, score) in enumerate(candidates, 1):
        mid = market.mid_price_yes
        spread = market.spread
        print(
            f"  {i:3d}. [{score:.3f}] {market.ticker}\n"
            f"       {market.title}\n"
            f"       Event: {event.title}\n"
            f"       Mid: {mid:.0f}c | Spread: {spread}c | "
            f"Vol: {market.volume:,} | OI: {market.open_interest:,} | "
            f"Closes: {market.close_time or 'N/A'}\n"
        )


def run_analysis(
    candidates: list[tuple[Event, Market, float]],
    model: str,
    limit: int,
) -> list[dict]:
    """Analyze candidates via Claude API and return signal dicts."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment or .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    to_analyze = candidates[:limit]
    signals = []
    print(f"\nAnalyzing {len(to_analyze)} markets with {model} ...\n")

    for i, (event, market, score) in enumerate(to_analyze, 1):
        mid = market.mid_price_yes
        print(f"[{i}/{len(to_analyze)}] {market.ticker} — {market.title} (mid {mid:.0f}c)")
        start = time.time()

        analysis = analyze_market(client, event, market, model)
        elapsed = time.time() - start

        if analysis is None:
            print(f"  Skipped (no valid response) [{elapsed:.1f}s]\n")
            continue

        data = analysis.pop("_market_data", collect_market_data(event, market))
        search_count = analysis.pop("_search_count", 0)

        fair_prob = analysis.get("fair_probability", 0)
        confidence = analysis.get("confidence", 0)
        side = analysis.get("side", "?")
        market_prob = mid / 100

        if side == "yes":
            edge = fair_prob - market_prob
        else:
            edge = (1 - fair_prob) - (1 - market_prob)

        # Color-code the edge
        edge_pct = edge * 100
        if edge_pct >= 5:
            edge_tag = f"*** {edge_pct:+.1f}% ***"
        elif edge_pct >= 3:
            edge_tag = f"  {edge_pct:+.1f}%"
        else:
            edge_tag = f"  {edge_pct:+.1f}% (below threshold)"

        print(f"  Fair: {fair_prob:.0%} | Edge: {edge_tag} | Side: {side.upper()} | "
              f"Conf: {confidence:.0%} | Searches: {search_count} [{elapsed:.1f}s]")
        print(f"  {analysis.get('reasoning', 'N/A')[:120]}")

        sig = build_signal_from_analysis(analysis, market, data)
        signals.append(sig)

        # Show actionable signals prominently
        if edge >= 0.03 and confidence >= 0.45:
            print(f"  >>> ACTIONABLE SIGNAL: {side.upper()} @ {mid:.0f}c, "
                  f"fair {fair_prob:.0%}, size ${sig['recommended_size_cents']/100:.2f}")

        print()

    return signals


def push_signals(backend_url: str, signals: list[dict]):
    """Display summary of signals (backend doesn't have a POST /api/signals endpoint)."""
    actionable = [s for s in signals if s["risk_check"]["allowed"]]
    print(f"\n{'='*80}")
    print(f"SCAN COMPLETE — {len(actionable)} actionable signals out of {len(signals)} analyzed")
    print(f"{'='*80}\n")

    if not actionable:
        print("  No actionable signals found this scan.\n")
        return

    # Sort by expected value (edge * confidence)
    actionable.sort(key=lambda s: s["edge"] * s["confidence"], reverse=True)

    for i, sig in enumerate(actionable, 1):
        print(
            f"  {i}. {sig['ticker']} — {sig['side'].upper()}\n"
            f"     Edge: {sig['edge']:.1%} | Confidence: {sig['confidence']:.0%} | "
            f"Fair: {sig['fair_probability']:.0%} vs Market: {sig['market_probability']:.0%}\n"
            f"     Size: ${sig['recommended_size_cents']/100:.2f} | "
            f"EV: {sig['edge']*sig['confidence']:.1%}\n"
            f"     {sig['reasoning'][:150]}\n"
        )

    # Also dump to JSON for easy import
    out_file = f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump({"signals": actionable, "scan_time": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    print(f"  Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run AI market scans locally using your Anthropic API key.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--backend-url", default="http://localhost:8000",
        help="Backend API URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Max number of markets to analyze with Claude (default: 20)",
    )
    parser.add_argument(
        "--prefilter-top", type=int, default=100,
        help="Number of markets to keep after microstructure pre-filter (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show top candidates without calling Claude",
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Don't display push summary (just show individual results)",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-6",
        help="Claude model to use (default: claude-sonnet-4-6)",
    )
    args = parser.parse_args()

    print(f"scan_local.py — Local AI market scanner")
    print(f"Backend: {args.backend_url} | Model: {args.model} | Limit: {args.limit}")
    print()

    # Step 1: Fetch events from backend
    try:
        events = fetch_events(args.backend_url)
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to backend at {args.backend_url}")
        print("Make sure the backend is running (python -m bot.server)")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"ERROR: Backend returned {e.response.status_code}: {e.response.text[:200]}")
        sys.exit(1)

    # Step 2: Pre-filter
    candidates = prefilter(events, top_n=args.prefilter_top)
    if not candidates:
        print("No eligible markets found.")
        sys.exit(0)

    # Step 3: Dry run or full analysis
    if args.dry_run:
        print_dry_run(candidates[:args.limit])
        sys.exit(0)

    # Step 4: Run Claude analysis
    signals = run_analysis(candidates, model=args.model, limit=args.limit)

    # Step 5: Summary
    if not args.no_push:
        push_signals(args.backend_url, signals)


if __name__ == "__main__":
    main()
