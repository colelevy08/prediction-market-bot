"""AI-powered market analysis using Claude to generate trading signals."""

from __future__ import annotations

import json

import anthropic

from bot.config import config
from bot.models import Event, Market, Side, TradingSignal


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
    """Uses Claude to analyze prediction markets and generate trading signals."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    def analyze_market(self, event: Event, market: Market) -> TradingSignal | None:
        """Analyze a single market and return a trading signal if there's edge."""
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
        """Kelly-criterion inspired position sizing (fractional Kelly at 25%)."""
        if edge <= 0 or confidence < 0.5:
            return 0

        # Simplified Kelly fraction: edge * confidence
        kelly_fraction = edge * confidence
        # Use quarter-Kelly for safety
        size = int(config.max_bet_amount_cents * kelly_fraction * 0.25)
        # Enforce min/max
        return max(0, min(size, config.max_bet_amount_cents))

    def analyze_events(self, events: list[Event]) -> list[TradingSignal]:
        """Analyze all markets across events and return actionable signals."""
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
