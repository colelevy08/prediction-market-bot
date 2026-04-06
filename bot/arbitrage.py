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

---------------------------------------------------------------------------
EDUCATIONAL OVERVIEW
---------------------------------------------------------------------------

WHAT IS ARBITRAGE?
  In finance, arbitrage is the simultaneous purchase and sale of the same (or
  equivalent) asset in different markets to profit from a price difference.
  Classic example: gold trades at $1,900/oz in New York and $1,905/oz in London.
  An arbitrageur buys in New York and sells in London for a risk-free $5 profit
  per ounce (ignoring transaction costs).

  In prediction markets, arbitrage means finding the same event priced at
  different probabilities on different exchanges. If "Will X happen?" trades
  at 40% on Kalshi but 50% on DraftKings, you could:
    - Buy YES on Kalshi at 40 cents
    - Sell YES on DraftKings at 50 cents (or buy NO at 50 cents = buy it at 50)
  No matter the outcome, you lock in a 10-cent profit on each contract pair.

  The word "arbitrage" comes from French and originally referred to the
  resolution of disputes — both sides are "arbitrated" simultaneously.

WHY ISN'T THIS ALWAYS RISK-FREE?
  True risk-free arbitrage requires:
    1. Simultaneous execution on both platforms (prices may move before you
       finish placing both orders)
    2. The ability to actually trade on both sides (DraftKings has no trading
       API, so the DK leg must be placed manually)
    3. Enough liquidity to fill your order at the quoted price
    4. Transaction costs smaller than the spread

  Because DK has no API, this bot can only DETECT the opportunity and alert
  you — human execution is still required on the DraftKings side.

WHAT IS A DATACLASS?
  Python's @dataclass decorator automatically generates __init__, __repr__,
  and other boilerplate methods for a class based on its field annotations.
  It's lighter-weight than a Pydantic BaseModel: it does no validation, just
  stores data. Used here because ArbitrageOpportunity is a simple output
  container, not user-facing data that needs input validation.

WHAT IS THE O(n*m) NOTATION IN find_matching_markets?
  Big-O notation describes how an algorithm's run time grows with input size.
  O(n*m) means: if Kalshi has n markets and DraftKings has m markets, the
  algorithm does n * m comparisons in the worst case. With 1,000 Kalshi markets
  and 200 DK markets, that's 200,000 comparisons. This is fine for moderate
  sizes but would become slow with millions of markets. A hash-based approach
  (pre-indexing by keywords) could reduce this to O(n + m).
---------------------------------------------------------------------------
"""

from __future__ import annotations

# dataclass: a lightweight decorator that auto-generates __init__ and __repr__
# for a class, avoiding boilerplate code.
from dataclasses import dataclass

from bot.models import DraftKingsMarket, Market


@dataclass
class ArbitrageOpportunity:
    """A detected price discrepancy between Kalshi and DraftKings for the same event.

    The spread_pct represents the absolute difference in YES prices between the two
    platforms. A spread above 5% (the default min_spread) may indicate an exploitable
    arbitrage opportunity.

    FIELDS EXPLAINED:
    - kalshi_yes_price / dk_yes_price: Both expressed as decimals (0.0 to 1.0),
      e.g., 0.40 = 40% implied probability.
    - spread_pct: abs(kalshi_yes_price - dk_yes_price). Always positive — it
      measures the SIZE of the discrepancy, not its direction.
    - recommended_action: A plain-English instruction like "Buy YES on Kalshi @ 40%,
      sell YES on DraftKings @ 50%" that tells a human trader exactly what to do.
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

    WHY KEYWORD MATCHING INSTEAD OF AN EXACT LOOKUP?
    Kalshi and DraftKings use different naming conventions for the same events.
    Kalshi might title a market "Will Bitcoin exceed $100,000 by end of March?"
    while DraftKings titles the same event "Bitcoin > $100K March 2024."
    There's no shared ID or standardised name, so we rely on overlapping words
    as a proxy for "same event." Requiring at least 3 shared words reduces false
    positives (e.g., two markets that both contain only "Will" and "2024" are
    probably not the same event).

    Args:
        kalshi_markets: List of open Kalshi markets.
        dk_markets: List of DraftKings prediction markets.

    Returns:
        List of (Kalshi Market, DraftKings Market) pairs that appear to reference
        the same real-world event.
    """
    matches = []
    for km in kalshi_markets:
        # Convert to lowercase so "Bitcoin" and "bitcoin" count as the same word.
        # split() turns the title string into a list of individual words,
        # then set() removes duplicates (so "will will Bitcoin" doesn't double-count "will").
        km_words = set(km.title.lower().split())
        for dk in dk_markets:
            dk_words = set(dk.title.lower().split())
            # The & operator on sets returns their INTERSECTION — only words present
            # in BOTH sets. This is the "keyword overlap" count.
            overlap = km_words & dk_words
            if len(overlap) >= 3:
                # 3 or more shared words → treat as the same event
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

    HOW TO READ THE RECOMMENDED ACTION:
    "Buy YES on Kalshi @ 40%, sell YES on DraftKings @ 50%" means:
      - You think YES is UNDERPRICED at Kalshi (only 40 cents)
      - You think YES is OVERPRICED at DraftKings (50 cents)
      - By buying the cheap side and selling the expensive side, you lock in
        a theoretical 10-cent profit per contract, independent of the outcome.

    WHY min_spread = 0.05 (5%)?
    Very small spreads (< 5%) are likely to be erased by:
      - Transaction costs (exchange fees, though Kalshi charges none currently)
      - Slippage (you may not get the exact quoted price at execution time)
      - The time delay between placing orders on two separate platforms
    A 5% minimum gives a meaningful margin of safety.

    Args:
        kalshi_markets: List of open Kalshi markets.
        dk_markets: List of DraftKings prediction markets.
        min_spread: Minimum spread (as decimal) to flag as an opportunity.

    Returns:
        List of ArbitrageOpportunity objects sorted by spread descending.
        Sorted largest spread first so the most lucrative opportunities appear at
        the top when displayed in the terminal table.
    """
    opportunities = []
    # Step 1: Find markets that appear to be the same event on both platforms
    matches = find_matching_markets(kalshi_markets, dk_markets)

    for km, dk in matches:
        # Convert Kalshi's integer cent price to a decimal for comparison.
        # mid_price_yes is in cents (0-100), so we divide by 100 to get a
        # probability (0.0-1.0) that can be compared directly with dk.yes_price.
        kalshi_yes = km.mid_price_yes / 100
        dk_yes = dk.yes_price

        # Sanity check: DraftKings price must be a valid probability (0 to 1).
        # A price of 0 or 1 usually means missing data, not a real market.
        if dk_yes <= 0 or dk_yes >= 1:
            continue

        # abs() gives the absolute (positive) difference between the two prices.
        # We don't care which direction the discrepancy goes yet — just whether
        # it's large enough to be worth acting on.
        spread = abs(kalshi_yes - dk_yes)

        if spread >= min_spread:
            # Determine which side is cheaper and therefore which platform to buy on.
            # The cheaper YES should be bought; the more expensive YES should be sold.
            if kalshi_yes < dk_yes:
                # Kalshi prices YES lower than DraftKings:
                # Buy YES on Kalshi (pay less), sell YES on DraftKings (receive more)
                action = f"Buy YES on Kalshi @ {kalshi_yes:.0%}, sell YES on DraftKings @ {dk_yes:.0%}"
            else:
                # DraftKings prices YES lower than Kalshi:
                # Buy YES on DraftKings (pay less), sell YES on Kalshi (receive more)
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

    # Sort so the highest-spread (most profitable) opportunities appear first.
    # lambda o: o.spread_pct is an anonymous function that extracts the sort key.
    # reverse=True means descending order (largest first).
    opportunities.sort(key=lambda o: o.spread_pct, reverse=True)
    return opportunities
