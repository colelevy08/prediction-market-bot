"""
Cross-platform arbitrage detection between Kalshi and DraftKings.

Scans for price discrepancies between Kalshi prediction markets and DraftKings
Predictions markets. When the same event is priced differently on both platforms,
there may be a risk-free (or low-risk) arbitrage opportunity.

Workflow:
  1. find_matching_markets(): Pairs Kalshi and DraftKings markets by title similarity
     using simple keyword overlap (requires >= 3 matching words). This is a heuristic
     approach; fuzzy matching (e.g., fuzzywuzzy) could improve match quality.
  2. detect_arbitrage(): For each matched pair, compares the YES price on both platforms.
     If the absolute spread exceeds min_spread (default 5%), an ArbitrageOpportunity is
     generated with a recommended action (e.g., "Buy YES on Kalshi, sell YES on DraftKings").

Limitations:
  - DraftKings does not have a stable public API, so market data may be incomplete.
  - Actual arbitrage execution on DraftKings must be done manually (no API trading).
  - Title matching is approximate and may produce false positives or miss valid pairs.

Connects to: bot.models (Market, DraftKingsMarket).
Used by: bot.server (GET /api/arbitrage), bot.main (--arbitrage CLI flag).
"""

from __future__ import annotations

from dataclasses import dataclass

from bot.models import DraftKingsMarket, Market


@dataclass
class ArbitrageOpportunity:
    """A detected price discrepancy between Kalshi and DraftKings for the same event.

    The spread_pct represents the absolute difference in YES prices between the two
    platforms. A spread above 5% (the default min_spread) may indicate an exploitable
    arbitrage opportunity.
    """
    kalshi_ticker: str            # Kalshi market ticker
    dk_market_id: str             # DraftKings market/draft group ID
    title: str                    # Market title (from Kalshi)
    kalshi_yes_price: float       # Kalshi YES mid price as decimal (0-1)
    dk_yes_price: float           # DraftKings YES price as decimal (0-1)
    spread_pct: float             # Absolute price difference (0-1)
    recommended_action: str       # Human-readable action, e.g., "Buy YES on Kalshi, sell YES on DK"


def find_matching_markets(
    kalshi_markets: list[Market],
    dk_markets: list[DraftKingsMarket],
) -> list[tuple[Market, DraftKingsMarket]]:
    """Attempt to match Kalshi and DraftKings markets by title keyword overlap.

    Uses a simple heuristic: splits both titles into words and requires >= 3 common
    words for a match. This is O(n*m) where n = Kalshi markets, m = DK markets.
    Could be improved with fuzzy matching (e.g., fuzzywuzzy) or TF-IDF similarity.

    Args:
        kalshi_markets: List of open Kalshi markets.
        dk_markets: List of DraftKings prediction markets.

    Returns:
        List of (Kalshi Market, DraftKings Market) pairs that appear to reference
        the same real-world event.
    """
    matches = []
    for km in kalshi_markets:
        km_words = set(km.title.lower().split())
        for dk in dk_markets:
            dk_words = set(dk.title.lower().split())
            # Require at least 3 overlapping words
            overlap = km_words & dk_words
            if len(overlap) >= 3:
                matches.append((km, dk))
    return matches


def detect_arbitrage(
    kalshi_markets: list[Market],
    dk_markets: list[DraftKingsMarket],
    min_spread: float = 0.05,
) -> list[ArbitrageOpportunity]:
    """Detect cross-platform arbitrage opportunities between Kalshi and DraftKings.

    For each matched market pair, compares YES prices. If the absolute difference
    exceeds min_spread (default 5%), creates an ArbitrageOpportunity with a
    recommended action indicating which platform to buy/sell on.

    Example: Kalshi YES = 40c, DraftKings YES = 50c => spread = 10%.
    Action: "Buy YES on Kalshi @ 40%, sell YES on DraftKings @ 50%"

    Args:
        kalshi_markets: List of open Kalshi markets.
        dk_markets: List of DraftKings prediction markets.
        min_spread: Minimum spread (as decimal) to flag as an opportunity.

    Returns:
        List of ArbitrageOpportunity objects sorted by spread descending.
    """
    opportunities = []
    matches = find_matching_markets(kalshi_markets, dk_markets)

    for km, dk in matches:
        kalshi_yes = km.mid_price_yes / 100
        dk_yes = dk.yes_price

        if dk_yes <= 0 or dk_yes >= 1:
            continue

        spread = abs(kalshi_yes - dk_yes)
        if spread >= min_spread:
            if kalshi_yes < dk_yes:
                action = f"Buy YES on Kalshi @ {kalshi_yes:.0%}, sell YES on DraftKings @ {dk_yes:.0%}"
            else:
                action = f"Buy YES on DraftKings @ {dk_yes:.0%}, sell YES on Kalshi @ {kalshi_yes:.0%}"

            opportunities.append(ArbitrageOpportunity(
                kalshi_ticker=km.ticker,
                dk_market_id=dk.market_id,
                title=km.title,
                kalshi_yes_price=kalshi_yes,
                dk_yes_price=dk_yes,
                spread_pct=spread,
                recommended_action=action,
            ))

    opportunities.sort(key=lambda o: o.spread_pct, reverse=True)
    return opportunities
