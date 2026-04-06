"""
AI-powered market analysis using Anthropic's Claude API.

----------------------------------------------------------------------
WHAT IS THIS MODULE DOING?
----------------------------------------------------------------------
This module is the "brain" layer of the trading bot. Instead of relying
purely on mathematical models, it asks the Claude AI to act as a
professional analyst who reads all the market data and forms a judgment
about what the true probability of an event should be.

Here's the conceptual flow:
  1. GATHER DATA: For each market, collect everything we know —
     prices, bid/ask spread, trading volume, order book imbalance,
     price history over time, and predictions from our ML model.

  2. BUILD A PROMPT: Organize all that data into a structured
     description and hand it to Claude.

  3. ASK CLAUDE TO RESEARCH: Claude has access to a web search tool.
     It searches for the latest news, polls, scores, or whatever is
     relevant to that specific prediction market.

  4. GET STRUCTURED ANALYSIS: Claude returns a JSON object with:
     - fair_probability: What Claude thinks the true odds are (0-1)
     - confidence: How sure Claude is (0-1)
     - side: Whether to bet YES or NO
     - reasoning: Plain-English explanation of the analysis
     - key_factors: The top reasons driving the probability estimate
     - risk_factors: What could go wrong with this analysis
     - edge_type: What kind of mispricing was found (fundamental,
                  timing, liquidity, sentiment, or structural)

  5. BUILD A SIGNAL: The analysis becomes a TradingSignal that
     says "bet YES/NO on this market, here's why."

----------------------------------------------------------------------
WHY COMBINE AI WITH MATHEMATICAL MODELS?
----------------------------------------------------------------------
The mathematical models (Random Forest + Gradient Boosting) are great
at finding patterns in historical price data. But they're blind to:
  - Breaking news that just happened
  - Context that requires world knowledge ("What does 'bipartisan
    support' actually mean for this bill's chances?")
  - Resolution criteria edge cases ("Does this count if it happens
    on the last day of the month?")

Claude provides the QUALITATIVE layer the math models lack. Together:
  - Math model: "Based on historical price patterns, this looks like
    a 60% YES based on numbers alone."
  - Claude + web search: "I just found a news article that says the
    event already happened this morning — this should be 95% YES."
  - Combined: Much stronger signal than either source alone.

----------------------------------------------------------------------
WHAT IS AN "ORDERBOOK" AND "ORDERBOOK IMBALANCE"?
----------------------------------------------------------------------
A market's "order book" is the list of all pending buy and sell orders.
  - BID side: Everyone who wants to BUY and the price they'll pay
  - ASK side: Everyone who wants to SELL and the price they want

"Bid pressure" = how much total buying interest exists
"Ask pressure" = how much total selling interest exists
"Imbalance" = (bids - asks) / (bids + asks)
  - Positive imbalance → more buyers than sellers → price likely to rise
  - Negative imbalance → more sellers than buyers → price likely to fall

This is a SHORT-TERM signal — it measures current supply/demand
pressure but doesn't predict the fundamental truth of the outcome.

----------------------------------------------------------------------
WHAT IS "MICROPRICE"?
----------------------------------------------------------------------
The microprice is a smarter estimate of the "true" price than just
taking the midpoint of the bid and ask.

Standard midpoint: (bid + ask) / 2
Microprice: weighs bid and ask by their SIZES (volume)
  Microprice = (ask × bid_size + bid × ask_size) / (bid_size + ask_size)

If there are 100 orders at the bid but only 10 at the ask, the
microprice leans toward the bid — reflecting that buyers are dominant.

----------------------------------------------------------------------
HOW THE TWO-PASS PRE-FILTER WORKS
----------------------------------------------------------------------
Before involving Claude (which costs API money and takes time), the
bot does a quick screening pass to find the most promising markets.

PASS 1 (fast, free):
  Score every open market on: spread tightness, price away from extremes,
  volume, and time until expiry. Pick the top 200.

PASS 2 (uses ML model):
  Run the Random Forest model on those 200 to find where it sees the
  biggest gap between market price and estimated true probability.
  Composite score = 30% microstructure + 70% ML edge.

TOP 100 go to Claude for deep analysis.

----------------------------------------------------------------------
KNOWLEDGE BASE / CACHING
----------------------------------------------------------------------
Claude analysis is expensive (each call costs time and money). The
knowledge base (_knowledge dict) stores previous analyses so Claude
can build on them rather than starting from scratch every scan.

When a market is re-analyzed, Claude sees its own previous reasoning
and can confirm, update, or reverse its view based on new information.
Markets that have NEVER been analyzed are prioritized over rescans.

Connects to: Anthropic Messages API, bot.config, bot.models, bot.rf_model (features).
Used by: bot.server (every scan), bot.main (CLI scan).
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone

import anthropic

from bot.config import config
from bot.models import Event, Market, Side, TradingSignal

logger = logging.getLogger("predictionbot")


# Data-driven analysis prompt — Claude gets REAL data, not just a title
DATA_DRIVEN_PROMPT = """You are a quantitative analyst at a prediction market fund. You receive REAL market data and must synthesize it with your world knowledge to find mispricings.

## Market Context
**Event:** {event_title}
**Market:** {market_title}
**Category:** {category}
**Closes:** {close_time} ({time_remaining} remaining)

## Live Price Data
- YES: bid={yes_bid}c ask={yes_ask}c mid={yes_mid}c
- NO: bid={no_bid}c ask={no_ask}c mid={no_mid}c
- Spread: {spread}c ({spread_pct:.1f}% of mid)
- Last traded: {last_price}c | Previous close: {prev_price}c
- Price change: {price_change:+.1f}c

## Volume & Liquidity
- Volume: {volume:,} contracts (${dollar_volume:,.0f} notional)
- Open interest: {open_interest:,} contracts
- Turnover rate: {turnover:.2f}x
- Liquidity score: {liquidity_score:.0f}
- Volume intensity: {volume_intensity:.1f} contracts/hour

## Orderbook Analysis
- Bid pressure: {bid_pressure:.2f} | Ask pressure: {ask_pressure:.2f}
- Orderbook imbalance: {ob_imbalance:+.2f} ({ob_signal})
- Microprice: {microprice:.1f}c (vs mid {yes_mid}c → {microprice_edge:+.1f}c skew)

## ML Model Analysis (108-feature RF+GB ensemble)
- Model probability: {model_prob:.1f}% (vs market {market_prob:.1f}%)
- Model edge: {model_edge:+.1f}% ({model_side} side)
- Model confidence: {model_confidence:.0f}%
- Prediction uncertainty: ±{prediction_std:.1f}%

## Price Momentum & History ({n_snapshots} snapshots)
{momentum_summary}

## Key Quantitative Signals
{quant_signals}

## Your Task
Given ALL the data above, determine the TRUE probability of this market resolving YES.

**IMPORTANT: You have web search available. USE IT.** Search for the latest news, polling data, scores, announcements, or any information relevant to this market. Do NOT rely solely on your training data — search for what's happening RIGHT NOW.

You have three advantages over the ML model:
1. **Live web research**: Search Google for breaking news, latest polls, scores, announcements
2. **World knowledge**: Context about the event, historical patterns, domain expertise
3. **Qualitative reasoning**: Resolution criteria edge cases, narrative biases, second-order effects

The ML model only sees numbers. You see the full picture AND can research current events.

**Analysis steps:**
1. Search the web for the latest information about this event/market
2. What does this market actually resolve on? Are there edge cases?
3. Given your research + the quantitative data, what's the true probability?
4. Where might the ML model or market be wrong given current news?
5. Final calibrated probability and confidence.

Respond with ONLY valid JSON:
{{
    "fair_probability": <float 0.0-1.0>,
    "confidence": <float 0.0-1.0>,
    "side": "<'yes' or 'no'>",
    "reasoning": "<3-5 sentences synthesizing data + knowledge>",
    "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
    "risk_factors": ["<risk 1>", "<risk 2>", "<risk 3>"],
    "agrees_with_model": <true or false>,
    "edge_type": "<'fundamental', 'timing', 'liquidity', 'sentiment', or 'structural'>",
    "data_quality": "<'high', 'medium', or 'low'>"
}}"""


class MarketAnalyzer:
    """Data-driven AI analyzer that feeds real market data to Claude.

    This class is the bridge between raw market data and Claude's intelligence.
    It collects all available quantitative information for a market, formats it
    into a structured prompt, sends it to Claude for analysis, parses the
    response, and returns a TradingSignal.

    KNOWLEDGE BASE:
    The _knowledge dictionary acts as an in-memory cache of previous analyses.
    Key = market ticker (e.g., "BTCUSD-24DEC31")
    Value = {
        "signal": the last TradingSignal generated,
        "timestamp": when the analysis was done (Unix timestamp),
        "reasoning": Claude's explanation,
        "fair_prob": Claude's estimated probability,
        "analysis_count": how many times we've analyzed this market,
    }

    This allows Claude to "remember" what it thought previously and update
    its view incrementally rather than starting from scratch every time.

    Maintains a persistent knowledge base of market analyses that grows
    and refreshes over time. Every scan does fresh research — Claude always
    calls the web and thinks from scratch, but gets its prior analysis as
    context so it can build on previous findings.

    The knowledge base prioritizes freshness: markets never analyzed get
    researched first, then the stalest analyses get refreshed.
    """

    def __init__(self, rf_generator=None):
        """Initialize with Anthropic client and optional RF model reference.

        Args:
            rf_generator: Optional reference to the RF (Random Forest) model
                          generator. If provided, Claude gets enriched with
                          108-feature model predictions, price history, and
                          orderbook analysis from that model.
        """
        # The Anthropic Python client — handles API authentication and HTTP calls
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        # Reference to RF model for features/history — can be None if not available
        self.rf_generator = rf_generator
        # Knowledge base: ticker → {signal, timestamp, reasoning, fair_prob, searches, analysis_count}
        self._knowledge: dict[str, dict] = {}

    def _collect_market_data(self, event: Event, market: Market) -> dict:
        """Collect all available quantitative data for a market.

        This function acts as the "data aggregator" — before we ask Claude
        anything, we need to gather every piece of relevant information
        about this market and format it for the prompt.

        What gets collected:
          - PRICES: YES bid/ask/mid, NO bid/ask/mid, spread, price change
          - TIME: How long until the market closes (affects trading dynamics)
          - VOLUME: How many contracts have traded, open interest, turnover
          - ML MODEL: The Random Forest model's probability estimate and confidence
          - ORDERBOOK: Bid pressure, ask pressure, imbalance signal
          - MICROPRICE: Smarter price estimate than simple midpoint
          - MOMENTUM: Recent price history showing direction and volatility
          - QUANTITATIVE SIGNALS: Pre-computed interpretations for Claude

        The result is a flat dictionary that gets injected into the
        DATA_DRIVEN_PROMPT template using Python's .format() method.

        Pulls from: live prices, RF model features, price history cache,
        orderbook analysis, and momentum indicators.
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

        # RF model prediction
        model_prob = yes_mid / 100  # Default to market price
        model_confidence = 0.5
        prediction_std = 0.15
        features = {}
        if self.rf_generator:
            try:
                from bot.rf_model import extract_features
                history = self.rf_generator.history_cache.get(market.ticker)
                features = extract_features(market, event, history)
                prediction = self.rf_generator.model.predict(features)
                model_prob = prediction["probability"]
                prediction_std = prediction.get("prediction_std", 0.15)
                model_confidence = max(model_prob, 1 - model_prob)
            except Exception as e:
                logger.debug(f"Feature extraction failed for {market.ticker}: {e}")

        # Orderbook analysis
        ob_imbalance = 0.0
        bid_pressure = 0.0
        ask_pressure = 0.0
        microprice = yes_mid
        ob_signal = "neutral"
        if self.rf_generator:
            try:
                ob = self.rf_generator.analyze_order_book(market)
                ob_imbalance = ob.get("imbalance", 0.0)
                bid_pressure = ob.get("bid_pressure", 0.0)
                ask_pressure = ob.get("ask_pressure", 0.0)
                ob_signal = ob.get("direction", "neutral")
            except Exception:
                pass

        # Compute microprice from bid/ask
        if market.yes_bid > 0 and market.yes_ask > 0:
            total_size = market.yes_bid + market.yes_ask
            if total_size > 0:
                microprice = (market.yes_ask * market.yes_bid + market.yes_bid * market.yes_ask) / total_size
                # Simplified microprice estimate from available data
                microprice = (market.yes_bid + market.yes_ask) / 2

        # Momentum from history
        momentum_lines = []
        n_snapshots = 0
        if self.rf_generator:
            history = self.rf_generator.history_cache.get(market.ticker, [])
            n_snapshots = len(history)
            if history and len(history) >= 2:
                recent = history[-min(5, len(history)):]
                prices = [h.get("yes_mid", yes_mid) for h in recent]
                vols = [h.get("volume", 0) for h in recent]
                momentum_lines.append(f"- Recent prices: {' → '.join(f'{p:.0f}c' for p in prices)}")
                if len(prices) >= 2:
                    change = prices[-1] - prices[0]
                    momentum_lines.append(f"- Price trend: {change:+.1f}c over {len(prices)} snapshots")
                    volatility = max(prices) - min(prices)
                    momentum_lines.append(f"- Price range: {min(prices):.0f}c - {max(prices):.0f}c (volatility: {volatility:.1f}c)")
                if any(v > 0 for v in vols):
                    momentum_lines.append(f"- Volume trend: {' → '.join(str(int(v)) for v in vols)}")
            else:
                momentum_lines.append("- No history yet (first scan)")

        # Quantitative signals summary
        quant_lines = []
        model_edge = (model_prob - yes_mid / 100) * 100
        if abs(model_edge) > 2:
            quant_lines.append(f"- ML model sees {abs(model_edge):.1f}% edge on {'YES' if model_edge > 0 else 'NO'} side")
        if abs(ob_imbalance) > 0.1:
            quant_lines.append(f"- Orderbook skewed {'bullish' if ob_imbalance > 0 else 'bearish'} (imbalance: {ob_imbalance:+.2f})")
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
        # Feature-based signals
        if features:
            price_extremity = features.get("price_extremity", 0)
            if price_extremity > 40:
                quant_lines.append(f"- Extreme price ({yes_mid}c) — rarely mispriced at extremes, high bar for edge")
            volume_intensity = features.get("volume_intensity", 0)
            if volume_intensity > 10:
                quant_lines.append(f"- High trading intensity ({volume_intensity:.0f}/hr) — active market")

        # Price change
        last_price = market.last_price if hasattr(market, 'last_price') and market.last_price else yes_mid
        prev_price = market.prev_price if hasattr(market, 'prev_price') and market.prev_price else yes_mid
        price_change = last_price - prev_price

        model_side = "YES" if model_prob > 0.5 else "NO"
        dollar_volume = volume * yes_mid / 100
        turnover = volume / max(oi, 1)
        volume_intensity = volume / max(hours_left, 1)
        liquidity_score = volume * (100 - spread) / 100 if spread <= 100 else 0

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
            "model_confidence": model_confidence * 100,
            "prediction_std": prediction_std * 100,
            "n_snapshots": n_snapshots,
            "momentum_summary": "\n".join(momentum_lines) if momentum_lines else "- No data available",
            "quant_signals": "\n".join(quant_lines) if quant_lines else "- No strong quantitative signals detected",
        }

    def analyze_market(self, event: Event, market: Market) -> TradingSignal | None:
        """Analyze a market with fresh Sonnet research, building on prior knowledge.

        This is the core analysis function. It:
          1. Collects all market data via _collect_market_data()
          2. Builds the prompt (including prior analysis if available)
          3. Calls Claude Sonnet with web search enabled (up to 5 searches)
          4. Parses the JSON response into a TradingSignal
          5. Stores the result in the knowledge base for future reference

        TEMPERATURE = 0.2: This is a parameter that controls how "creative"
        Claude is. 0.0 = completely deterministic (same answer every time),
        1.0 = very creative/random. We use 0.2 for slight variability while
        keeping answers mostly consistent and analytical.

        MAX TOKENS = 8000: This limits the length of Claude's response.
        JSON responses are short, but Claude often thinks through the problem
        at length before producing the final JSON, so we give it enough space.

        EDGE CALCULATION:
        After getting Claude's fair_probability, we calculate the edge:
          For YES bets: edge = fair_probability - market_price
          For NO bets:  edge = (1 - fair_probability) - (1 - market_price)
                             = market_price - fair_probability

        If edge is positive, we have an advantage over the market price.
        Example: Claude says 65% chance, market says 45% → YES edge = 20%.

        Always calls Claude Sonnet with web search. If we have a prior analysis,
        Claude gets it as context so it can confirm, update, or reverse its view.
        """
        now = datetime.now(timezone.utc).timestamp()
        prior = self._knowledge.get(market.ticker)

        data = self._collect_market_data(event, market)
        prompt = DATA_DRIVEN_PROMPT.format(**data)

        # If we have prior analysis, give Claude context to build on
        if prior and prior.get("reasoning"):
            age_min = (now - prior["timestamp"]) / 60
            prompt += (
                f"\n\n## Your Prior Analysis ({age_min:.0f} minutes ago, scan #{prior.get('analysis_count', 1)})\n"
                f"- Fair probability: {prior.get('fair_prob', 'N/A')}\n"
                f"- Side: {prior.get('side', 'N/A')} | Edge: {prior.get('edge', 'N/A')}\n"
                f"- Reasoning: {prior['reasoning']}\n"
                f"\n**Has anything changed?** Search for the LATEST news and update your analysis. "
                f"Confirm if your prior view still holds, or revise it based on new information. "
                f"Price may have moved since your last analysis."
            )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
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
                logger.warning(f"No text response for {market.ticker} ({search_count} searches)")
                return prior.get("signal") if prior else None

            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            analysis = json.loads(text)
            fair_prob = float(analysis["fair_probability"])
            fair_prob = max(0.01, min(0.99, fair_prob))
            market_prob = data["yes_mid"] / 100
            side = Side(analysis["side"])

            if side == Side.YES:
                edge = fair_prob - market_prob
            else:
                edge = (1 - fair_prob) - (1 - market_prob)

            reasoning_parts = [analysis.get("reasoning", "")]
            if analysis.get("key_factors"):
                reasoning_parts.append(f"Key: {'; '.join(analysis['key_factors'][:3])}")
            if analysis.get("risk_factors"):
                reasoning_parts.append(f"Risks: {'; '.join(analysis['risk_factors'][:3])}")
            agrees = analysis.get("agrees_with_model", True)
            if not agrees:
                reasoning_parts.append("AI DISAGREES with ML model")
            full_reasoning = " | ".join(reasoning_parts)

            signal = TradingSignal(
                ticker=market.ticker,
                market_title=market.title,
                side=side,
                confidence=min(0.99, max(0.0, float(analysis["confidence"]))),
                fair_probability=fair_prob,
                market_probability=market_prob,
                edge=edge,
                reasoning=full_reasoning,
                recommended_size_cents=self._calculate_size(edge, float(analysis["confidence"])),
            )

            prev_count = prior.get("analysis_count", 0) if prior else 0
            self._knowledge[market.ticker] = {
                "signal": signal,
                "timestamp": now,
                "reasoning": full_reasoning,
                "fair_prob": fair_prob,
                "side": side.value,
                "edge": f"{edge:.1%}",
                "searches": search_count,
                "analysis_count": prev_count + 1,
            }

            rescan_tag = f"rescan #{prev_count + 1}" if prev_count > 0 else "first scan"
            logger.info(
                f"AI: {market.ticker} [{rescan_tag}] | edge={edge:.1%} | conf={analysis['confidence']:.0%} | "
                f"side={side.value} | {search_count} searches | agrees_model={agrees} | "
                f"type={analysis.get('edge_type', '?')} | quality={analysis.get('data_quality', '?')}"
            )

            return signal

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse AI analysis for {market.ticker}: {e}")
            return prior.get("signal") if prior else None
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error for {market.ticker}: {e}")
            return prior.get("signal") if prior else None

    def _calculate_size(self, edge: float, confidence: float) -> int:
        """Kelly-criterion position sizing (quarter-Kelly for AI signals).

        This is a simplified Kelly formula used for AI-generated signals.
        It computes a PRELIMINARY bet size that will be refined by the full
        Kelly calculation in risk_manager.kelly_size() before any order is placed.

        QUARTER-KELLY (0.25):
        We use an even more conservative fraction than the half-Kelly used elsewhere.
        Why? AI signals are based on qualitative reasoning and web search — they're
        inherently less precise than quantitative model outputs. The extra
        conservatism compensates for that uncertainty.

        Formula here:
          kelly_fraction = edge × confidence
          size = max_bet × kelly_fraction × 0.25

        Example: edge=0.15 (15%), confidence=0.80 (80%), max_bet=$15
          kelly_fraction = 0.15 × 0.80 = 0.12
          size = $15 × 0.12 × 0.25 = $0.45 → rounded to 45 cents
        This is a small preliminary size; the risk manager will scale it up or
        down based on current portfolio conditions.
        """
        if edge <= 0 or confidence < 0.5:
            return 0
        kelly_fraction = edge * confidence
        size = int(config.max_bet_amount_cents * kelly_fraction * 0.25)
        return max(0, min(size, config.max_bet_amount_cents))

    def _pre_filter_markets(self, events: list[Event]) -> list[tuple[Event, Market, float]]:
        """Pre-filter to top ~100 markets with highest edge potential.

        With hundreds of markets available on Kalshi, we can't afford to run
        Claude's full web-research analysis on all of them (it would take hours
        and cost a lot in API fees). So we use a two-pass filter to quickly
        identify the most promising candidates.

        PASS 1 — MICROSTRUCTURE SCORING (runs on ALL markets, very fast):
        Each market gets a score from 0 to 1 based on four factors:

          spread_score (30% weight): Tighter spread → better liquidity → easier to profit
            Score = 1 - (spread / 15). A 0-cent spread scores 1.0; 15+ cents scores 0.
            WHY? In illiquid markets with wide spreads, transaction costs eat your profit.

          price_score (25% weight): Mid price close to 50 cents → more room for edge
            Score = 1 - |price - 50| / 50
            WHY? Markets priced at 95/5 cents have very little room to move.
            The biggest potential mispricings tend to be in the 30-70 cent range.

          volume_score (25% weight): Higher volume → more active market
            Score = min(volume / 5000, 1.0). Caps at 5000 contracts.
            WHY? Very illiquid (low volume) markets can be mispriced but also hard to trade.

          time_score (20% weight): Markets closing 1-30 days out are ideal
            Score = 1.0 for 1-30 days, 0.6 for 30+ days, 0.3 for <1 day
            WHY? Markets expiring very soon are highly volatile. Expiring far out
            might not have enough information yet to improve on the market price.

        PASS 2 — ML MODEL SCORING (runs on top 200 from Pass 1):
        For each market, run the Random Forest model to get its probability estimate.
        If the model significantly disagrees with the market price, that's a signal.

          model_edge = |model_probability - market_price|
          composite_score = 0.3 × microstructure_score + 0.7 × model_edge_score

        The heavy weighting on model edge (70%) means we prioritize markets where
        our quantitative model thinks the market is most wrong.

        Two-pass scoring:
          1. Fast microstructure pass on all open markets (spread, price, volume, time)
          2. RF model prediction on top 200 → boost score with model edge
        Returns top 100 by composite score for deep AI analysis.
        """
        # Pass 1: microstructure scoring on all markets
        raw_candidates = []
        for event in events:
            for market in event.markets:
                if market.status not in ("open", "active"):
                    continue
                if market.volume < 10:
                    continue

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

                micro_score = (
                    spread_score * 0.30
                    + price_score * 0.25
                    + volume_score * 0.25
                    + time_score * 0.20
                )

                raw_candidates.append((event, market, micro_score))

        raw_candidates.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"AI pre-filter pass 1: {len(raw_candidates)} open markets with volume > 10")

        # Pass 2: run RF model on top 200 to find where model sees edge
        top_for_model = raw_candidates[:200]
        scored = []
        if self.rf_generator:
            try:
                from bot.rf_model import extract_features
                for event, market, micro_score in top_for_model:
                    try:
                        history = self.rf_generator.history_cache.get(market.ticker)
                        features = extract_features(market, event, history)
                        prediction = self.rf_generator.model.predict(features)
                        model_prob = prediction["probability"]
                        market_prob = market.mid_price_yes / 100
                        model_edge = abs(model_prob - market_prob)
                        # Composite: microstructure + model edge (heavily weighted)
                        composite = micro_score * 0.3 + min(model_edge / 0.15, 1.0) * 0.7
                        scored.append((event, market, composite))
                    except Exception:
                        scored.append((event, market, micro_score * 0.3))
            except Exception as e:
                logger.warning(f"RF model scoring failed: {e}")
                scored = [(ev, mk, sc * 0.3) for ev, mk, sc in top_for_model]
        else:
            scored = [(ev, mk, sc) for ev, mk, sc in top_for_model]

        scored.sort(key=lambda x: x[2], reverse=True)

        # Take top 100 for deep AI analysis
        result = scored[:100]
        if result:
            logger.info(
                f"AI pre-filter pass 2: top {len(result)} candidates selected "
                f"(best score: {result[0][2]:.3f}, worst: {result[-1][2]:.3f})"
            )
        return result

    def get_all_signals(self) -> list[TradingSignal]:
        """Return all current signals from the knowledge base.

        Filters the knowledge base to only return actionable signals —
        those with at least 3% edge AND 45% confidence minimum.

        The minimum thresholds (3% edge, 45% confidence) are conservative
        floors. The risk_manager.check_signal() will apply stricter filters
        (typically 8% minimum edge) before any actual trades are placed.

        Results are sorted by edge × confidence (highest first), which is
        a combined "opportunity quality" score. A signal with 20% edge and
        90% confidence scores higher than one with 30% edge and 50% confidence.
        """
        signals = []
        for ticker, entry in self._knowledge.items():
            sig = entry.get("signal")
            # Only include signals that have meaningful edge AND confidence
            if sig and sig.edge >= 0.03 and sig.confidence >= 0.45:
                signals.append(sig)
        # Sort best opportunities first: higher edge × confidence = better signal
        signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        return signals

    def analyze_events(self, events: list[Event], max_analyses: int = 0) -> list[TradingSignal]:
        """Analyze top market candidates with fresh AI research.

        This is the main entry point called by the bot's scan loop. Each time
        the bot "scans," it calls this function to get an updated set of signals.

        THE SCANNING CYCLE:
          1. Pre-filter all available markets down to top 100 candidates
          2. Sort candidates by staleness (never-analyzed first)
          3. For each candidate, call analyze_market() (Claude + web search)
          4. Return ALL signals from the full knowledge base

        STALENESS PRIORITY:
        A market that has never been analyzed gets priority over one that was
        analyzed 30 minutes ago. This ensures every market gets at least one
        deep look before any market gets a second look.

        RESCANS:
        After all candidates have been analyzed at least once, subsequent cycles
        re-analyze in order of age: the oldest analysis gets updated first.
        Markets can evolve — a political event's probability changes as news breaks,
        so yesterday's analysis may no longer be valid today.

        Every call does real Claude research with web search — no skipping.
        Markets are prioritized by staleness: never-analyzed first, then
        oldest analyses. The knowledge base grows and refreshes over time.

        Args:
            events: List of market events to analyze (from Kalshi API).
            max_analyses: Cap on markets to research this cycle (0 = default 100).
                          Setting this lower speeds up each scan cycle but
                          means some markets may not be analyzed every cycle.
        """
        import time as _time

        candidates = self._pre_filter_markets(events)
        now = datetime.now(timezone.utc).timestamp()

        # Sort candidates by staleness — never analyzed first, then oldest
        def staleness_key(item):
            _event, market, score = item
            prior = self._knowledge.get(market.ticker)
            if not prior:
                return (0, -score)  # Never analyzed → highest priority, break ties by score
            age = now - prior["timestamp"]
            return (1, -age)  # Analyzed before → priority by age (oldest first)

        candidates.sort(key=staleness_key)

        # Take top N for this cycle
        limit = max_analyses if max_analyses > 0 else 100
        top_candidates = candidates[:limit]

        if not top_candidates:
            logger.warning("AI analysis: 0 candidates passed pre-filter — nothing to analyze")
            return self.get_all_signals()

        # Stats
        never_analyzed = sum(1 for _, m, _ in top_candidates if m.ticker not in self._knowledge)
        rescans = len(top_candidates) - never_analyzed
        cat_counts = {}
        for event, market, _ in top_candidates:
            cat = event.category or market.category or "Other"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        est_time = len(top_candidates) * 120  # ~2 min per market
        logger.info(
            f"AI analysis starting: {len(top_candidates)} markets "
            f"({never_analyzed} new + {rescans} rescans) from {sum(len(e.markets) for e in events)} total, "
            f"est. {est_time // 60}m {est_time % 60}s, "
            f"{len(cat_counts)} categories, knowledge base: {len(self._knowledge)} markets"
        )
        self._progress = {"total": len(top_candidates), "done": 0, "signals": 0, "started": _time.time()}

        analyzed = 0
        new_signals_this_cycle = 0
        batch_start = _time.time()
        for event, market, opp_score in top_candidates:
            market_start = _time.time()
            signal = self.analyze_market(event, market)
            market_elapsed = _time.time() - market_start
            analyzed += 1

            if signal and signal.edge >= 0.03 and signal.confidence >= 0.45:
                new_signals_this_cycle += 1

            self._progress["done"] = analyzed
            self._progress["signals"] = len(self.get_all_signals())

            # Log progress every 3 markets or at the end
            if analyzed % 3 == 0 or analyzed == len(top_candidates):
                elapsed = _time.time() - batch_start
                avg_per_market = elapsed / analyzed
                remaining = (len(top_candidates) - analyzed) * avg_per_market
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                total_signals = len(self.get_all_signals())
                logger.info(
                    f"AI research: {analyzed}/{len(top_candidates)} | "
                    f"{market_elapsed:.0f}s last | {new_signals_this_cycle} new this cycle | "
                    f"{total_signals} total in knowledge base | "
                    f"ETA: {eta_min}m {eta_sec}s"
                )

        # Return ALL signals from knowledge base (not just this cycle)
        all_signals = self.get_all_signals()

        total_elapsed = _time.time() - batch_start
        logger.info(
            f"AI cycle complete: {analyzed} markets researched in {total_elapsed:.0f}s | "
            f"{new_signals_this_cycle} new signals this cycle | "
            f"{len(all_signals)} total signals in knowledge base | "
            f"{len(self._knowledge)} markets tracked"
        )
        return all_signals
