"""Cross-platform arbitrage detection between Kalshi and DraftKings."""

from __future__ import annotations

from dataclasses import dataclass

from bot.models import DraftKingsMarket, Market


@dataclass
class ArbitrageOpportunity:
    """A detected price discrepancy between platforms."""
    kalshi_ticker: str
    dk_market_id: str
    title: str
    kalshi_yes_price: float
    dk_yes_price: float
    spread_pct: float
    recommended_action: str  # e.g., "Buy YES on Kalshi, Buy NO on DraftKings"


def find_matching_markets(
    kalshi_markets: list[Market],
    dk_markets: list[DraftKingsMarket],
) -> list[tuple[Market, DraftKingsMarket]]:
    """
    Attempt to match Kalshi and DraftKings markets by title similarity.
    Uses simple keyword overlap - could be improved with fuzzy matching.
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
    """
    Detect cross-platform arbitrage opportunities.

    If Kalshi YES is 40c and DraftKings YES equivalent is 50c,
    there's a potential 10% spread to exploit.
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
