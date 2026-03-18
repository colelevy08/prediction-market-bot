"""
AI-powered market analysis using Anthropic's Claude API.

This module provides an alternative (or supplement) to the Random Forest ensemble
for generating trading signals. It sends structured prompts to Claude asking it to:

  1. Understand what the prediction market event is actually asking.
  2. Assess publicly available information relevant to the outcome.
  3. Estimate a "true" probability independent of the current market price.
  4. Calculate the edge (fair_prob - market_prob) and recommend a side.

The response is parsed as JSON containing: fair_probability, confidence, side, reasoning.
Position sizing uses a simplified quarter-Kelly formula: size = max_bet * edge * confidence * 0.25.

Filtering: Signals are only returned when edge >= min_edge_threshold AND confidence >= 55%.
Markets with volume < 100 are skipped to avoid illiquid contracts.

This analyzer is optional; the bot primarily uses the RF+GB ensemble (rf_model.py).
Claude analysis can be triggered via the "use_ai" flag in the /api/scan endpoint or
from the CLI. It's slower and costs API credits, but provides qualitative reasoning.

Connects to: Anthropic Messages API (Claude claude-sonnet-4-6 model), bot.config (API key, max bet),
bot.models (Event, Market, Side, TradingSignal).
Used by: bot.server (POST /api/scan with use_ai=true), bot.main (CLI scan).
"""

from __future__ import annotations

import json

import anthropic

from bot.config import config
from bot.models import Event, Market, Side, TradingSignal


# Structured prompt template for Claude AI market analysis.
# The prompt asks Claude to reason step-by-step about the true probability of an
# event, compare it to the market price, and return a JSON response with fields:
# fair_probability, confidence, side, and reasoning.
ANALYSIS_PROMPT = """You are a quantitative analyst for prediction markets. Analyze the following market and estimate the TRUE probability of the event occurring.

**Event:** {event_title}
**Market:** {market_title}
**Category:** {category}
**Current YES price:** {yes_price}c (market-implied probability: {yes_price}%)
**Current NO price:** {no_price}c
**Spread:** {spread}c
**Volume:** {volume} contracts
**Closes:** {close_time}

Think step by step:
1. What is this event actually asking?
2. What publicly available information is relevant?
3. What is a reasonable base rate or probability estimate?
4. How does your estimate compare to the market price?
5. How confident are you in your edge?

Respond with ONLY valid JSON (no markdown, no explanation outside JSON):
{{
    "fair_probability": <float 0-1, your estimate of the true YES probability>,
    "confidence": <float 0-1, how confident you are in your estimate>,
    "side": "<'yes' or 'no', which side to trade>",
    "reasoning": "<brief 2-3 sentence explanation>"
}}"""


class MarketAnalyzer:
    """Uses Anthropic's Claude API to analyze prediction markets and generate trading signals.

    Each market analysis costs one Claude API call (~500 tokens output). The analyzer
    is optional and significantly slower than the RF ensemble, but provides qualitative
    reasoning that can complement quantitative signals.
    """

    def __init__(self):
        """Initialize the Anthropic client with the configured API key."""
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def analyze_market(self, event: Event, market: Market) -> TradingSignal | None:
        """Analyze a single market using Claude and return a TradingSignal if there's edge.

        Sends the market details to Claude, parses the JSON response, calculates
        the edge (fair_prob - market_prob), and builds a TradingSignal.

        Args:
            event: Parent event containing the market.
            market: The specific market contract to analyze.

        Returns:
            TradingSignal if analysis succeeds, None if parsing fails or API errors.
        """
        prompt = ANALYSIS_PROMPT.format(
            event_title=event.title,
            market_title=market.title,
            category=market.category or event.category,
            yes_price=market.yes_ask or market.mid_price_yes,
            no_price=market.no_ask or (100 - market.mid_price_yes),
            spread=market.spread,
            volume=market.volume,
            close_time=market.close_time or "Unknown",
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            # Parse JSON response
            analysis = json.loads(text)
            fair_prob = float(analysis["fair_probability"])
            market_prob = (market.yes_ask or market.mid_price_yes) / 100
            side = Side(analysis["side"])

            # Calculate edge
            if side == Side.YES:
                edge = fair_prob - market_prob
            else:
                edge = (1 - fair_prob) - (1 - market_prob)

            return TradingSignal(
                ticker=market.ticker,
                market_title=market.title,
                side=side,
                confidence=float(analysis["confidence"]),
                fair_probability=fair_prob,
                market_probability=market_prob,
                edge=edge,
                reasoning=analysis.get("reasoning", ""),
                recommended_size_cents=self._calculate_size(edge, float(analysis["confidence"])),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"  [!] Failed to parse analysis for {market.ticker}: {e}")
            return None
        except anthropic.APIError as e:
            print(f"  [!] API error analyzing {market.ticker}: {e}")
            return None

    def _calculate_size(self, edge: float, confidence: float) -> int:
        """Kelly-criterion inspired position sizing (quarter-Kelly for extra safety).

        Uses a simplified formula: size = max_bet * edge * confidence * 0.25.
        Quarter-Kelly (rather than the RF model's half-Kelly) is used here because
        Claude's probability estimates are less calibrated than the trained ensemble.
        """
        if edge <= 0 or confidence < 0.5:
            return 0

        # Simplified Kelly fraction: edge * confidence
        kelly_fraction = edge * confidence
        # Use quarter-Kelly for safety
        size = int(config.max_bet_amount_cents * kelly_fraction * 0.25)
        # Enforce min/max
        return max(0, min(size, config.max_bet_amount_cents))

    def analyze_events(self, events: list[Event]) -> list[TradingSignal]:
        """Analyze all markets across events and return actionable signals.

        Filters: skips closed markets and markets with volume < 100 (illiquid).
        Only returns signals where edge >= min_edge_threshold AND confidence >= 55%.
        Results are sorted by expected value (edge * confidence) descending.
        """
        signals = []
        for event in events:
            for market in event.markets:
                if market.status != "open":
                    continue
                if market.volume < 100:  # Skip illiquid markets
                    continue

                signal = self.analyze_market(event, market)
                if signal and signal.edge >= config.min_edge_threshold and signal.confidence >= 0.55:
                    signals.append(signal)

        # Sort by edge * confidence (expected value)
        signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        return signals
