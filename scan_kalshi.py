"""
Kalshi Market Scanner — full predictionbot analysis for FREE.

Pulls RF model signals + raw market data from your deployed Railway backend,
then formats everything for Claude Code to analyze with web search. Claude Code
runs on your Max subscription — zero API cost.

This script does THREE things:
1. Triggers an RF model scan on the backend (POST /api/scan) to get model signals,
   near misses, and worst opportunities
2. Fetches raw market data for the top 100 candidates so Claude Code can do
   its own independent research with web search
3. (Optional) Generates research notes with contextual information for each market

Usage (just tell Claude Code "scan markets"):
    python scan_kalshi.py                          # Full scan, top 100 for analysis
    python scan_kalshi.py --limit 50               # Fewer candidates
    python scan_kalshi.py --category politics       # Filter by category
    python scan_kalshi.py --ticker KXBTCD          # Deep-dive one market
    python scan_kalshi.py --signals-only           # Just show RF signals, skip raw data
    python scan_kalshi.py --research               # Include deeper research notes per market
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

import httpx

from bot.models import Event, Market

RAILWAY_URL = "https://prediction-market-bot-railway-production.up.railway.app"
KALSHI_MARKET_URL = "https://kalshi.com/markets"

# Category keywords for research context
CATEGORY_KEYWORDS = {
    "politics": [
        "election", "president", "senate", "house", "congress", "governor",
        "trump", "biden", "democrat", "republican", "vote", "poll", "primary",
        "cabinet", "impeach", "legislation", "bill", "executive order",
    ],
    "sports": [
        "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball",
        "baseball", "hockey", "tennis", "golf", "ufc", "boxing", "march madness",
        "playoffs", "championship", "super bowl", "world series", "finals",
        "game", "match", "tournament", "ncaa",
    ],
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "token", "blockchain",
        "solana", "sol", "dogecoin", "doge", "xrp",
    ],
    "economics": [
        "fed", "interest rate", "inflation", "cpi", "gdp", "jobs", "unemployment",
        "recession", "housing", "treasury", "fomc", "rate cut", "rate hike",
        "nonfarm", "payroll",
    ],
    "tech": [
        "ai", "artificial intelligence", "openai", "google", "apple", "meta",
        "microsoft", "amazon", "tesla", "ipo", "launch", "release",
    ],
    "weather": [
        "temperature", "hurricane", "tornado", "storm", "rainfall", "snow",
        "heat", "cold", "climate", "weather", "drought", "flood",
    ],
    "entertainment": [
        "oscar", "emmy", "grammy", "box office", "movie", "film", "tv",
        "streaming", "album", "award", "celebrity",
    ],
}


def detect_category(title: str, explicit_category: str | None = None) -> str:
    """Detect the effective category from title keywords and explicit category."""
    if explicit_category:
        cat_lower = explicit_category.lower()
        for cat in CATEGORY_KEYWORDS:
            if cat in cat_lower:
                return cat
    title_lower = title.lower()
    best_cat = "other"
    best_score = 0
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in title_lower)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat if best_score > 0 else "other"


def research_market(ticker: str, title: str, category: str | None = None) -> str:
    """Return relevant context/research notes based on the market category and title.

    This generates contextual research prompts and known factors that Claude Code
    should investigate when analyzing this market. For real-time data, Claude Code
    should use web search.
    """
    effective_cat = detect_category(title, category)
    title_lower = title.lower()
    notes = []

    if effective_cat == "politics":
        notes.append("POLITICAL MARKET — Check recent polling data, FiveThirtyEight/RCP averages.")
        if any(kw in title_lower for kw in ["trump", "biden", "president"]):
            notes.append("Presidential market: look at approval ratings, swing state polls, prediction market consensus.")
        if any(kw in title_lower for kw in ["senate", "house", "congress"]):
            notes.append("Congressional market: check Cook Political Report ratings, fundraising data, generic ballot.")
        if any(kw in title_lower for kw in ["executive order", "legislation", "bill"]):
            notes.append("Policy market: check congressional schedule, whip counts, recent statements from leadership.")
        if any(kw in title_lower for kw in ["cabinet", "nominee", "confirmation"]):
            notes.append("Nomination market: check Senate committee schedules, vote counts, recent hearing testimony.")
        notes.append("Key sources: FiveThirtyEight, RealClearPolitics, Polymarket, PredictIt for cross-reference.")

    elif effective_cat == "sports":
        notes.append("SPORTS MARKET — Check injury reports, recent form, head-to-head records.")
        if any(kw in title_lower for kw in ["nba", "basketball"]):
            notes.append("NBA: check ESPN/NBA.com for injury reports, back-to-back schedules, rest days, betting lines.")
        if any(kw in title_lower for kw in ["nfl", "football", "super bowl"]):
            notes.append("NFL: check weather conditions, injury reports (Wednesday/Thursday/Friday practice reports).")
        if any(kw in title_lower for kw in ["mlb", "baseball", "world series"]):
            notes.append("MLB: check starting pitcher matchup, bullpen usage, park factors, weather.")
        if any(kw in title_lower for kw in ["march madness", "ncaa", "tournament"]):
            notes.append("NCAA Tournament: check KenPom ratings, bracket positioning, tempo/efficiency stats.")
        notes.append("Key sources: ESPN, Vegas consensus lines, team beat reporters on Twitter/X.")

    elif effective_cat == "crypto":
        notes.append("CRYPTO MARKET — Check current price, recent volatility, on-chain metrics.")
        if any(kw in title_lower for kw in ["bitcoin", "btc"]):
            notes.append("Bitcoin: check ETF flows, halving cycle position, hash rate, exchange reserves.")
        if any(kw in title_lower for kw in ["ethereum", "eth"]):
            notes.append("Ethereum: check gas fees, staking rate, L2 activity, upcoming protocol upgrades.")
        notes.append("Key sources: CoinGecko, Glassnode, CryptoQuant, Deribit options for implied vol.")

    elif effective_cat == "economics":
        notes.append("ECONOMIC MARKET — Check Fed dot plot, FOMC minutes, recent data releases.")
        if any(kw in title_lower for kw in ["fed", "interest rate", "fomc", "rate"]):
            notes.append("Fed market: check CME FedWatch tool for rate probabilities, recent Fed speaker comments.")
        if any(kw in title_lower for kw in ["cpi", "inflation"]):
            notes.append("Inflation market: check Cleveland Fed Nowcast, recent PPI/PCE, energy prices, shelter costs.")
        if any(kw in title_lower for kw in ["jobs", "unemployment", "nonfarm", "payroll"]):
            notes.append("Jobs market: check ADP preview, jobless claims trend, ISM employment components.")
        if any(kw in title_lower for kw in ["gdp", "recession"]):
            notes.append("GDP/recession market: check GDPNow (Atlanta Fed), leading economic indicators, yield curve.")
        notes.append("Key sources: FRED, BLS, CME FedWatch, Bloomberg consensus estimates.")

    elif effective_cat == "tech":
        notes.append("TECH MARKET — Check recent news, product announcements, earnings reports.")
        if "ai" in title_lower or "artificial intelligence" in title_lower:
            notes.append("AI market: check recent model releases, regulatory developments, company announcements.")
        notes.append("Key sources: TechCrunch, The Verge, company investor relations pages, SEC filings.")

    elif effective_cat == "weather":
        notes.append("WEATHER MARKET — Check NWS forecasts, ECMWF/GFS models, historical data.")
        if any(kw in title_lower for kw in ["temperature", "heat", "cold"]):
            notes.append("Temperature market: check 7-day/14-day forecasts, Climate Prediction Center outlooks.")
        if any(kw in title_lower for kw in ["hurricane", "storm", "tornado"]):
            notes.append("Storm market: check NHC forecasts, storm track models, historical base rates.")
        notes.append("Key sources: weather.gov, ECMWF, Weather Underground, Climate Prediction Center.")

    elif effective_cat == "entertainment":
        notes.append("ENTERTAINMENT MARKET — Check industry predictions, critic reviews, box office tracking.")
        notes.append("Key sources: Gold Derby, Box Office Mojo, Rotten Tomatoes, Metacritic.")

    else:
        notes.append(f"GENERAL MARKET — Research the specific topic: '{title}'")
        notes.append("Look for recent news, expert opinions, historical base rates for similar events.")

    # Add ticker-specific note
    notes.append(f"Ticker: {ticker} — Cross-reference with Polymarket and other prediction markets for consensus.")

    return "\n".join(f"  - {n}" for n in notes)


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


def format_signal(sig: dict, rank: int, include_research: bool = False) -> str:
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
    spread = sig.get("spread", 0)
    volume = sig.get("volume", 0)
    oi = sig.get("open_interest", 0)

    lines = [
        f"### {rank}. {ticker}",
        f"**{title}**",
        f"Category: {cat} | Side: **{side}** | Edge: **{edge_pct:+.1f}%** | "
        f"Confidence: {conf:.0f}% | Quality: {quality:.3f}",
        f"Model Prob: {fair:.1f}% | Market Price: {mkt:.1f}% | Spread: {spread}c",
        f"Volume: {volume:,} | Open Interest: {oi:,}",
        f"Size: ${size/100:.2f} | EV: {edge_pct * conf / 100:.2f}%",
        f"Reasoning: {reasoning[:200]}",
        f"Bet: {kalshi_link(ticker)}",
    ]

    if include_research:
        lines.append(f"\n**Research Notes:**")
        lines.append(research_market(ticker, title, cat))

    return "\n".join(lines) + "\n"


def format_near_miss(nm: dict, rank: int, include_research: bool = False) -> str:
    """Format a near miss."""
    edge = nm.get("edge", 0) * 100
    conf = nm.get("confidence", 0) * 100
    fair = nm.get("fair_probability", 0) * 100
    mkt = nm.get("market_probability", 0) * 100
    side = nm.get("side", "?").upper()
    reason = nm.get("reason", "")
    ticker = nm.get("ticker", "")
    title = nm.get("market_title", "")
    cat = nm.get("category", "")
    spread = nm.get("spread", 0)
    volume = nm.get("volume", 0)
    oi = nm.get("open_interest", 0)

    lines = [
        f"{rank}. **{ticker}** — {title}",
        f"   {side} | Edge: {edge:+.1f}% | Model Prob: {fair:.1f}% | Market Price: {mkt:.1f}%",
        f"   Spread: {spread}c | Volume: {volume:,} | OI: {oi:,}",
        f"   Conf: {conf:.0f}% | Blocked: {reason}",
        f"   Bet: {kalshi_link(ticker)}",
    ]

    if include_research:
        lines.append(f"   **Research Notes:**")
        lines.append(research_market(ticker, title, cat))

    return "\n".join(lines) + "\n"


def format_worst(w: dict, rank: int, include_research: bool = False) -> str:
    """Format a worst opportunity (overpriced market)."""
    edge = w.get("edge", 0) * 100
    conf = w.get("confidence", 0) * 100
    fair = w.get("fair_probability", 0) * 100
    mkt = w.get("market_probability", 0) * 100
    ticker = w.get("ticker", "")
    title = w.get("market_title", "")
    cat = w.get("category", "")
    spread = w.get("spread", 0)
    volume = w.get("volume", 0)
    oi = w.get("open_interest", 0)

    lines = [
        f"{rank}. **{ticker}** — {title}",
        f"   Overpriced by {abs(edge):.1f}% | Model Prob: {fair:.1f}% | Market Price: {mkt:.1f}%",
        f"   Spread: {spread}c | Volume: {volume:,} | OI: {oi:,}",
        f"   Conf: {conf:.0f}%",
        f"   Bet: {kalshi_link(ticker)}",
    ]

    if include_research:
        lines.append(f"   **Research Notes:**")
        lines.append(research_market(ticker, title, cat))

    return "\n".join(lines) + "\n"


def format_market_for_analysis(event: Event, market: Market, rank: int,
                                rf_data: dict | None = None,
                                include_research: bool = False) -> str:
    """Format a market with full data for Claude Code deep analysis."""
    mid = market.mid_price_yes
    spread = market.spread
    volume = market.volume
    oi = market.open_interest
    tr = time_remaining_str(market.close_time)
    cat = market.category or event.category or "Other"

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
        f"**Category:** {cat}",
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
        model_prob = rf_data.get("fair_probability", rf_data.get("model_prob", 0))
        rf_edge = rf_data.get("edge", 0) * 100
        rf_conf = rf_data.get("confidence", 0) * 100
        rf_side = rf_data.get("side", "?").upper()
        lines.append(f"**RF Model:** prob={model_prob:.1f}% | "
                      f"edge={rf_edge:+.1f}% | "
                      f"conf={rf_conf:.0f}% | "
                      f"side={rf_side}")

    if include_research:
        lines.append(f"\n**Research Notes:**")
        lines.append(research_market(market.ticker, market.title, cat))

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi Market Scanner — full predictionbot analysis for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scan_kalshi.py                          # Full scan, top 100 for analysis
    python scan_kalshi.py --limit 50               # Fewer candidates
    python scan_kalshi.py --category politics       # Filter by category
    python scan_kalshi.py --ticker KXBTCD          # Deep-dive one market
    python scan_kalshi.py --signals-only           # Just show RF signals, skip raw data
    python scan_kalshi.py --research               # Include research notes per market
        """,
    )
    parser.add_argument("--backend-url", default=RAILWAY_URL, help=f"Backend URL (default: Railway)")
    parser.add_argument("--limit", type=int, default=100, help="Number of candidates for Claude analysis (default: 100)")
    parser.add_argument("--top", type=int, default=10, help="Number of top signals to highlight (default: 10)")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    parser.add_argument("--ticker", type=str, default=None, help="Deep-dive a specific ticker")
    parser.add_argument("--signals-only", action="store_true", help="Only show RF signals, skip raw candidates")
    parser.add_argument("--research", action="store_true", help="Include deeper research notes for each market")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    backend = args.backend_url
    include_research = args.research
    print(f"# Kalshi Market Scanner — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", file=sys.stderr)
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

    print(f"# KALSHI MARKET SCANNER REPORT")
    print(f"**{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}** | "
          f"Model: {'TRAINED' if model_trained else 'NOT TRAINED'}"
          f"{' | Research Mode: ON' if include_research else ''}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # Section 1: SIGNALS — Top RF Model Signals
    # ═══════════════════════════════════════════════════════════════
    signals = scan_result.get("signals", []) if scan_result else []
    signals.sort(key=lambda s: s.get("edge", 0) * s.get("confidence", 0), reverse=True)

    print(f"## ═══ SIGNALS ═══")
    print(f"## TOP {min(args.top, len(signals))} RF MODEL SIGNALS")
    print(f"These passed all filters (edge >= 3%, confidence >= 65%, quality threshold).\n")
    if signals:
        for i, sig in enumerate(signals[:args.top], 1):
            print(format_signal(sig, i, include_research))
    else:
        print("No signals passed all filters this scan.\n")

    # ═══════════════════════════════════════════════════════════════
    # Section 2: NEAR MISSES
    # ═══════════════════════════════════════════════════════════════
    near_misses = scan_result.get("near_misses", []) if scan_result else []
    if near_misses:
        print(f"\n## ═══ NEAR MISSES ═══")
        print(f"## {len(near_misses)} markets with 3%+ edge but blocked by secondary filters")
        print("These have edge but failed a secondary filter — worth watching.\n")
        for i, nm in enumerate(near_misses[:10], 1):
            print(format_near_miss(nm, i, include_research))

    # ═══════════════════════════════════════════════════════════════
    # Section 3: WORST — Overpriced Markets
    # ═══════════════════════════════════════════════════════════════
    worst = scan_result.get("worst_opportunities", []) if scan_result else []
    if worst:
        print(f"\n## ═══ WORST (OVERPRICED) ═══")
        print(f"## {len(worst)} overpriced markets — potential NO plays")
        print("Model sees these as significantly overpriced.\n")
        for i, w in enumerate(worst[:10], 1):
            print(format_worst(w, i, include_research))

    if args.signals_only:
        # Still print summary even in signals-only mode
        _print_summary(signals, near_misses, worst, 0, 0)
        return

    # ═══════════════════════════════════════════════════════════════
    # Section 4: Raw candidates for Claude Code deep analysis
    # ═══════════════════════════════════════════════════════════════
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

    print(f"\n## ═══ RAW MARKET DATA FOR ANALYSIS ═══")
    print(f"Top {len(top_candidates)} candidates by microstructure score (spread, volume, price, time).")
    print(f"Claude Code: research each market with web search, estimate fair probability,")
    print(f"identify mispricings, and select the best 10 to report.\n")

    for i, (event, market, score) in enumerate(top_candidates, 1):
        rf_data = rf_signal_map.get(market.ticker)
        print(format_market_for_analysis(event, market, i, rf_data, include_research))

    # ═══════════════════════════════════════════════════════════════
    # Section 5: RESEARCH NOTES (if --research flag is set)
    # ═══════════════════════════════════════════════════════════════
    if include_research:
        print(f"\n## ═══ RESEARCH NOTES ═══")
        print(f"Aggregated research guidance for the top opportunities.\n")

        # Collect unique categories from signals and top candidates
        all_cats = set()
        for sig in signals[:args.top]:
            cat = detect_category(sig.get("market_title", ""), sig.get("category"))
            all_cats.add(cat)
        for event, market, _ in top_candidates[:20]:
            cat = detect_category(market.title, market.category or event.category)
            all_cats.add(cat)

        for cat in sorted(all_cats):
            if cat == "other":
                continue
            print(f"### {cat.upper()} Markets")
            if cat == "politics":
                print("- Check FiveThirtyEight, RealClearPolitics for latest polling averages")
                print("- Cross-reference with Polymarket and PredictIt prices")
                print("- Look for recent endorsements, policy announcements, or scandals")
                print("- Check congressional schedule for upcoming votes")
            elif cat == "sports":
                print("- Check ESPN for injury reports and lineup changes")
                print("- Compare with Vegas consensus lines (DraftKings, FanDuel)")
                print("- Look at recent form (last 5-10 games)")
                print("- Check weather conditions for outdoor sports")
            elif cat == "crypto":
                print("- Check CoinGecko/CoinMarketCap for current prices and 24h trends")
                print("- Look at ETF flow data (for BTC/ETH)")
                print("- Check Deribit for options-implied volatility")
                print("- Monitor regulatory news (SEC, CFTC)")
            elif cat == "economics":
                print("- Check CME FedWatch for rate probabilities")
                print("- Look at Bloomberg/Reuters consensus estimates for data releases")
                print("- Check Atlanta Fed GDPNow and Cleveland Fed inflation nowcast")
                print("- Monitor recent Fed speaker comments")
            elif cat == "tech":
                print("- Check recent earnings reports and guidance")
                print("- Monitor product launch timelines and announcements")
                print("- Look at SEC filings for material events")
            elif cat == "weather":
                print("- Check NWS/weather.gov for official forecasts")
                print("- Compare ECMWF and GFS model runs")
                print("- Look at Climate Prediction Center outlooks")
            elif cat == "entertainment":
                print("- Check Gold Derby for expert predictions and odds")
                print("- Monitor Box Office Mojo for revenue tracking")
                print("- Look at critic consensus on Rotten Tomatoes/Metacritic")
            print()

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    total_candidates = len(candidates)
    _print_summary(signals, near_misses, worst, len(top_candidates), total_candidates)

    print(f"\nClaude Code: analyze the candidates above using web search, then report your TOP 10 with:")
    print(f"1. Ticker + full market title")
    print(f"2. Category")
    print(f"3. Your fair probability vs market price (current price)")
    print(f"4. Edge (%) and side (YES/NO)")
    print(f"5. Spread, volume, open interest")
    print(f"6. Confidence level and reasoning")
    print(f"7. Whether RF model agrees")
    print(f"8. Recommended bet size")
    print(f"9. Kalshi bet link")


def _print_summary(signals: list, near_misses: list, worst: list,
                   candidates_shown: int, total_candidates: int) -> None:
    """Print the summary section."""
    print(f"\n## ═══ SUMMARY ═══")
    print(f"**Total markets scanned:** RF model covered all active markets")
    print(f"**Signals found:** {len(signals)}")
    print(f"**Near misses:** {len(near_misses)}")
    print(f"**Overpriced (worst):** {len(worst)}")
    if total_candidates > 0:
        print(f"**Candidates shown:** {candidates_shown}/{total_candidates}")

    # Best opportunities summary
    if signals:
        best = max(signals, key=lambda s: s.get("edge", 0) * s.get("confidence", 0))
        best_edge = best.get("edge", 0) * 100
        best_ticker = best.get("ticker", "")
        best_title = best.get("market_title", "")
        best_side = best.get("side", "?").upper()
        print(f"\n**Best opportunity:** {best_ticker} — {best_title}")
        print(f"  {best_side} | Edge: {best_edge:+.1f}% | {kalshi_link(best_ticker)}")

    if worst:
        most_overpriced = min(worst, key=lambda w: w.get("edge", 0))
        op_edge = most_overpriced.get("edge", 0) * 100
        op_ticker = most_overpriced.get("ticker", "")
        op_title = most_overpriced.get("market_title", "")
        print(f"\n**Most overpriced:** {op_ticker} — {op_title}")
        print(f"  Overpriced by {abs(op_edge):.1f}% | {kalshi_link(op_ticker)}")

    print()


if __name__ == "__main__":
    main()
