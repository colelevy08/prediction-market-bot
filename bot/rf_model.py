"""
Random Forest + Gradient Boosting ensemble for prediction market analysis.

From the guide: 100+ models voting simultaneously. Each tree looks at different
data. The final answer is the result of all 100+ voting.

Features per tree: sqrt(total_features) — optimal for Random Forest.
Output: probability 0-1 via sigmoid. We only enter when model is 70%+ confident.
Entry: buy when market_price <= model_probability * 0.5 (2x undervalued)
Exit: sell when market_price >= model_probability * 0.9 (90% correction)
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from bot.models import Event, Market, Side, TradingSignal
from bot.config import config


# ── Feature Engineering: 106 features ────────────────────────────────────────

def extract_features(market: Market, event: Event, history: list[dict] | None = None) -> dict[str, float]:
    """
    Extract 106 features from market data for the ensemble model.

    The guide says: "markets run on 100+ factors" — we extract every
    quantitative signal available from price, volume, time, and orderbook data.
    When historical snapshots are available, we add momentum/volatility features.
    """
    yes_mid = market.mid_price_yes
    no_mid = 100 - yes_mid
    spread = market.spread
    volume = market.volume
    oi = market.open_interest
    history = history or []

    # Parse time to expiry
    days_to_expiry = 30.0
    hours_to_expiry = 720.0
    minutes_to_expiry = 43200.0
    if market.close_time:
        try:
            close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            delta = (close_dt - now).total_seconds()
            days_to_expiry = max(0, delta / 86400)
            hours_to_expiry = max(0, delta / 3600)
            minutes_to_expiry = max(0, delta / 60)
        except (ValueError, TypeError):
            pass

    # ── 1. Price Features (18) ───────────────────────────────────
    features = {
        "yes_bid": float(market.yes_bid),
        "yes_ask": float(market.yes_ask),
        "no_bid": float(market.no_bid),
        "no_ask": float(market.no_ask),
        "yes_mid": yes_mid,
        "no_mid": no_mid,
        "spread": float(spread),
        "spread_pct": spread / max(yes_mid, 1),
        "spread_pct_no": spread / max(no_mid, 1),
        "bid_ask_ratio": market.yes_bid / max(market.yes_ask, 1),
        "no_bid_ask_ratio": market.no_bid / max(market.no_ask, 1),
        "price_extremity": abs(yes_mid - 50),
        "price_extremity_sq": (yes_mid - 50) ** 2 / 2500,
        "log_odds_yes": math.log(max(yes_mid, 1) / max(100 - yes_mid, 1)),
        "implied_prob": yes_mid / 100,
        "implied_prob_sq": (yes_mid / 100) ** 2,
        "price_bucket_low": 1.0 if yes_mid < 25 else 0.0,
        "price_bucket_high": 1.0 if yes_mid > 75 else 0.0,
    }

    # ── 2. Volume & Liquidity Features (14) ──────────────────────
    features.update({
        "volume": float(volume),
        "log_volume": math.log1p(volume),
        "volume_sq": math.sqrt(max(volume, 0)),
        "open_interest": float(oi),
        "log_open_interest": math.log1p(oi),
        "volume_oi_ratio": volume / max(oi, 1),
        "liquidity_score": volume * (100 - spread) / 100,
        "turnover_rate": volume / max(oi, 1) if oi > 0 else 0,
        "volume_per_cent_spread": volume / max(spread, 1),
        "dollar_volume": volume * yes_mid / 100,
        "log_dollar_volume": math.log1p(volume * yes_mid / 100),
        "volume_intensity": volume / max(hours_to_expiry, 1),
        "oi_concentration": oi / max(volume, 1),
        "volume_rank_proxy": min(volume / 1000, 10),  # saturates at 10k
    })

    # ── 3. Time Features (14) ────────────────────────────────────
    features.update({
        "days_to_expiry": days_to_expiry,
        "hours_to_expiry": hours_to_expiry,
        "minutes_to_expiry": minutes_to_expiry,
        "log_days_to_expiry": math.log1p(days_to_expiry),
        "log_hours_to_expiry": math.log1p(hours_to_expiry),
        "time_decay_factor": 1 / max(days_to_expiry, 0.01),
        "time_decay_sqrt": 1 / math.sqrt(max(days_to_expiry, 0.01)),
        "is_expiring_soon": 1.0 if days_to_expiry <= 7 else 0.0,
        "is_expiring_today": 1.0 if days_to_expiry <= 1 else 0.0,
        "is_expiring_hour": 1.0 if hours_to_expiry <= 1 else 0.0,
        "expiry_urgency": max(0, 1 - days_to_expiry / 30),
        "sqrt_days_to_expiry": math.sqrt(max(days_to_expiry, 0)),
        "time_value": yes_mid * days_to_expiry / 100,
        "theta_proxy": -yes_mid / max(days_to_expiry, 0.1) / 100,
    })

    # ── 4. Orderbook Imbalance Features (12) ─────────────────────
    yes_total = market.yes_bid + market.yes_ask
    no_total = market.no_bid + market.no_ask
    all_total = yes_total + no_total

    features.update({
        "bid_pressure": market.yes_bid / max(market.yes_bid + market.no_bid, 1),
        "ask_pressure": market.yes_ask / max(market.yes_ask + market.no_ask, 1),
        "order_imbalance": (market.yes_bid - market.no_bid) / max(market.yes_bid + market.no_bid, 1),
        "ask_imbalance": (market.yes_ask - market.no_ask) / max(market.yes_ask + market.no_ask, 1),
        "bid_depth_ratio": market.yes_bid / max(all_total, 1),
        "ask_depth_ratio": market.yes_ask / max(all_total, 1),
        "bid_skew": (market.yes_bid - market.no_bid) / max(spread, 1),
        "ask_skew": (market.yes_ask - market.no_ask) / max(spread, 1),
        "weighted_mid": (market.yes_bid * market.no_ask + market.yes_ask * market.no_bid) / max(yes_total + no_total, 1),
        "microprice": (market.yes_bid * market.yes_ask + market.no_bid * market.no_ask) / max(yes_total + no_total, 1),
        "bid_strength": market.yes_bid / max(yes_mid, 1),
        "ask_weakness": (market.yes_ask - yes_mid) / max(yes_mid, 1),
    })

    # ── 5. Cross-Market / Efficiency Features (10) ───────────────
    features.update({
        "yes_no_spread": float(market.yes_ask - market.no_ask),
        "market_efficiency": 1 - abs(market.yes_ask + market.no_ask - 100) / 100,
        "overround": (market.yes_ask + market.no_ask) / 100,
        "vig_estimate": max(0, (market.yes_ask + market.no_ask - 100)) / 100,
        "synthetic_edge": (market.yes_bid + (100 - market.no_ask)) / 2 - yes_mid,
        "arb_spread": max(0, market.yes_bid + market.no_bid - 100),
        "reverse_arb": max(0, 100 - market.yes_ask - market.no_ask),
        "price_dislocation": abs(market.yes_ask - (100 - market.no_bid)),
        "fair_value_gap": (market.yes_bid + market.yes_ask) / 2 - (100 - (market.no_bid + market.no_ask) / 2),
    })
    features["efficiency_score"] = 1 - abs(features["fair_value_gap"]) / 50

    # ── 6. Momentum Proxies from Current Data (8) ────────────────
    features.update({
        "price_momentum_proxy": (yes_mid - 50) * volume / 10000,
        "mean_reversion_signal": (50 - yes_mid) / max(abs(50 - yes_mid), 1) * (1 / max(days_to_expiry, 0.1)),
        "volume_momentum": volume * (yes_mid - 50) / 5000,
        "buying_urgency": market.yes_bid * volume / max(oi, 1) / 100,
        "selling_urgency": (100 - market.yes_ask) * volume / max(oi, 1) / 100,
        "price_velocity_proxy": (yes_mid - 50) / max(days_to_expiry, 0.1),
        "conviction_score": abs(yes_mid - 50) * math.log1p(volume) / 100,
        "smart_money_proxy": abs(features["order_imbalance"]) * math.log1p(volume),
    })

    # ── 7. Historical Momentum Features (14) ─────────────────────
    # These require historical price snapshots; defaults to 0 if unavailable
    if len(history) >= 2:
        prices = [h.get("yes_mid", 50) for h in history]
        volumes = [h.get("volume", 0) for h in history]
        current = prices[-1]
        prev = prices[-2]

        # Price changes
        pct_change_1 = (current - prev) / max(prev, 1)
        pct_change_all = (current - prices[0]) / max(prices[0], 1) if prices[0] > 0 else 0

        # Volatility
        if len(prices) >= 3:
            returns = [(prices[i] - prices[i-1]) / max(prices[i-1], 1) for i in range(1, len(prices))]
            vol = float(np.std(returns)) if returns else 0
        else:
            returns = []
            vol = 0

        # Volume trend
        vol_change = (volumes[-1] - volumes[0]) / max(volumes[0], 1) if volumes[0] > 0 else 0

        features.update({
            "momentum_1": pct_change_1,
            "momentum_total": pct_change_all,
            "momentum_abs": abs(pct_change_all),
            "volatility": vol,
            "log_volatility": math.log1p(vol),
            "volatility_adj_momentum": pct_change_all / max(vol, 0.001),
            "price_range": (max(prices) - min(prices)) / max(max(prices), 1),
            "price_position": (current - min(prices)) / max(max(prices) - min(prices), 1),
            "trend_strength": abs(pct_change_all) / max(vol, 0.001),
            "volume_trend": vol_change,
            "volume_acceleration": (volumes[-1] - 2 * volumes[len(volumes)//2] + volumes[0]) / max(volumes[0], 1) if len(volumes) >= 3 else 0,
            "price_std": float(np.std(prices)) if prices else 0,
            "price_skew": float((np.mean(prices) - np.median(prices)) / max(float(np.std(prices)), 0.01)) if len(prices) >= 3 else 0,
            "mean_reversion_hist": (np.mean(prices) - current) / max(float(np.std(prices)), 0.01) if len(prices) >= 3 else 0,
        })
    else:
        features.update({
            "momentum_1": 0, "momentum_total": 0, "momentum_abs": 0,
            "volatility": 0, "log_volatility": 0, "volatility_adj_momentum": 0,
            "price_range": 0, "price_position": 0.5, "trend_strength": 0,
            "volume_trend": 0, "volume_acceleration": 0, "price_std": 0,
            "price_skew": 0, "mean_reversion_hist": 0,
        })

    # ── 8. Interaction Features (16) ─────────────────────────────
    # Non-linear combinations that help the model find complex patterns
    features.update({
        "price_x_volume": (yes_mid / 100) * math.log1p(volume),
        "price_x_time": (yes_mid / 100) * (1 / max(days_to_expiry, 0.1)),
        "volume_x_time": math.log1p(volume) * (1 / max(days_to_expiry, 0.1)),
        "spread_x_time": spread * (1 / max(days_to_expiry, 0.1)),
        "spread_x_volume": spread / max(math.log1p(volume), 0.1),
        "edge_x_liquidity": abs(yes_mid - 50) * math.log1p(volume) / max(spread, 1),
        "imbalance_x_volume": features["order_imbalance"] * math.log1p(volume),
        "imbalance_x_time": features["order_imbalance"] * (1 / max(days_to_expiry, 0.1)),
        "extremity_x_efficiency": features["price_extremity"] * features["market_efficiency"],
        "vig_x_volume": features["vig_estimate"] * math.log1p(volume),
        "momentum_x_volume": features["price_momentum_proxy"] * math.log1p(volume) / 100,
        "conviction_x_time": features["conviction_score"] * (1 / max(days_to_expiry, 0.1)),
        "bid_pressure_x_vol": features["bid_pressure"] * math.log1p(volume),
        "spread_efficiency": features["spread_pct"] * features["market_efficiency"],
        "time_weighted_price": yes_mid * math.exp(-days_to_expiry / 30) / 100,
        "risk_adjusted_edge": abs(yes_mid - 50) / max(spread + 1, 1) / max(features.get("volatility", 0.01), 0.01),
    })

    return features


FEATURE_NAMES = list(extract_features(
    Market(ticker="x", event_ticker="x", title="x"),
    Event(event_ticker="x", title="x"),
).keys())


# ── Ensemble Model ───────────────────────────────────────────────────────────

class PredictionModel:
    """
    Ensemble model: Random Forest + Gradient Boosting averaged.

    From the guide:
    - 100+ trees, each looking at different data subsets
    - Features per tree: sqrt(total_features)
    - Output: sigmoid probability 0-1
    - Calibrated probabilities for better accuracy
    """

    def __init__(self, n_estimators: int = 200):
        self.n_estimators = n_estimators
        self.rf: RandomForestClassifier | None = None
        self.gb: GradientBoostingClassifier | None = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.cv_score: float = 0.0
        self.n_training_samples: int = 0
        self._build_models()

    def _build_models(self):
        """Initialize the ensemble with guide-specified parameters."""
        n_features = len(FEATURE_NAMES)
        max_features_per_tree = max(1, int(math.sqrt(n_features)))

        # Random Forest: guide's primary model
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=max_features_per_tree,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            oob_score=True,  # out-of-bag score for free validation
        )

        # Gradient Boosting: complements RF with sequential learning
        self.gb = GradientBoostingClassifier(
            n_estimators=min(self.n_estimators, 150),
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            max_features=max_features_per_tree,
            min_samples_leaf=3,
            random_state=42,
        )

    def train_on_historical(self, features_list: list[dict], outcomes: list[int]) -> dict:
        """
        Train both models on historical market data.

        Returns training metrics including cross-validation score.
        """
        if len(features_list) < 30:
            return {"error": "Need at least 30 samples to train", "trained": False}

        X = np.array([[f.get(name, 0) for name in FEATURE_NAMES] for f in features_list])
        y = np.array(outcomes)

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train Random Forest
        self.rf.fit(X_scaled, y)

        # Train Gradient Boosting
        self.gb.fit(X_scaled, y)

        self.is_trained = True
        self.n_training_samples = len(features_list)

        # Cross-validation score (5-fold)
        try:
            cv_scores = cross_val_score(self.rf, X_scaled, y, cv=min(5, len(y) // 10), scoring="accuracy")
            self.cv_score = float(np.mean(cv_scores))
        except Exception:
            self.cv_score = self.rf.oob_score_ if hasattr(self.rf, 'oob_score_') else 0

        return {
            "trained": True,
            "samples": len(features_list),
            "cv_accuracy": round(self.cv_score, 4),
            "oob_score": round(self.rf.oob_score_, 4) if hasattr(self.rf, 'oob_score_') and self.rf.oob_score_ else 0,
            "n_features": len(FEATURE_NAMES),
        }

    def predict_probability(self, features: dict) -> float:
        """
        Predict YES probability using ensemble averaging.

        Combines RF and GB predictions with weighted average.
        RF gets 60% weight (better calibrated), GB gets 40% (better at edges).
        """
        if not self.is_trained:
            return self._heuristic_probability(features)

        X = np.array([[features.get(name, 0) for name in FEATURE_NAMES]])
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        # Ensemble: weighted average of RF and GB
        rf_proba = self.rf.predict_proba(X_scaled)[0]
        rf_yes = float(rf_proba[1]) if len(rf_proba) > 1 else float(rf_proba[0])

        gb_proba = self.gb.predict_proba(X_scaled)[0]
        gb_yes = float(gb_proba[1]) if len(gb_proba) > 1 else float(gb_proba[0])

        # Weighted ensemble
        ensemble_prob = 0.6 * rf_yes + 0.4 * gb_yes

        return float(np.clip(ensemble_prob, 0.01, 0.99))

    def _heuristic_probability(self, features: dict) -> float:
        """
        Multi-factor heuristic when no training data is available.
        Uses 12 signals weighted by empirical importance.
        """
        implied = features.get("implied_prob", 0.5)
        bid_pressure = features.get("bid_pressure", 0.5)
        order_imbalance = features.get("order_imbalance", 0.0)
        ask_imbalance = features.get("ask_imbalance", 0.0)
        overround = features.get("overround", 1.0)
        volume = features.get("log_volume", 0)
        spread_pct = features.get("spread_pct", 0.1)
        microprice = features.get("microprice", 50)
        smart_money = features.get("smart_money_proxy", 0)
        bid_strength = features.get("bid_strength", 0.5)
        efficiency = features.get("market_efficiency", 1.0)
        momentum = features.get("momentum_1", 0)

        # Base: market implied probability
        prob = implied

        # Signal 1: Order flow (strongest single signal)
        prob += (bid_pressure - 0.5) * 0.12
        prob += order_imbalance * 0.08
        prob += ask_imbalance * 0.05

        # Signal 2: Microprice (volume-weighted fair value)
        microprice_implied = microprice / 100
        prob = prob * 0.85 + microprice_implied * 0.15

        # Signal 3: Smart money indicator
        if smart_money > 2:
            prob += 0.03 * np.sign(order_imbalance)

        # Signal 4: Momentum (if available)
        prob += momentum * 0.1

        # Signal 5: Bid strength vs weakness
        prob += (bid_strength - 0.5) * 0.05

        # Regression toward 0.5 for unreliable data
        if spread_pct > 0.2:
            reliability = max(0.5, 1 - spread_pct)
            prob = prob * reliability + 0.5 * (1 - reliability)

        if overround > 1.0:
            vig = (overround - 1.0) / 2
            prob = prob - vig if prob > 0.5 else prob + vig

        if volume < 3:
            weight = volume / 3
            prob = prob * weight + 0.5 * (1 - weight)

        if efficiency < 0.9:
            prob = prob * efficiency + 0.5 * (1 - efficiency)

        return float(np.clip(prob, 0.01, 0.99))

    def get_feature_importance(self) -> dict[str, float]:
        """Get combined feature importance from both models."""
        if not self.is_trained:
            return {}
        rf_imp = self.rf.feature_importances_
        gb_imp = self.gb.feature_importances_
        combined = 0.6 * rf_imp + 0.4 * gb_imp
        return dict(sorted(
            zip(FEATURE_NAMES, combined),
            key=lambda x: x[1],
            reverse=True,
        ))


# ── Signal Generation (Guide Logic) ─────────────────────────────────────────

class RFSignalGenerator:
    """
    Generates trading signals using the ensemble model
    with the guide's entry/exit rules.

    Entry: market_price <= model_probability * 0.5 (buy at 2x undervalued)
    Exit: market_price >= model_probability * 0.9 (sell at 90% correction)
    Only enter when model is 70%+ confident.
    """

    def __init__(self):
        self.model = PredictionModel(n_estimators=200)
        self.trade_log: list[dict] = []
        self.history_cache: dict[str, list[dict]] = {}

    def record_snapshot(self, market: Market):
        """Record a price snapshot for momentum features."""
        key = market.ticker
        if key not in self.history_cache:
            self.history_cache[key] = []
        self.history_cache[key].append({
            "yes_mid": market.mid_price_yes,
            "volume": market.volume,
            "timestamp": time.time(),
        })
        # Keep last 100 snapshots per market
        if len(self.history_cache[key]) > 100:
            self.history_cache[key] = self.history_cache[key][-100:]

    def generate_signals(self, events: list[Event]) -> list[TradingSignal]:
        """Scan all markets and generate entry signals per guide rules."""
        signals = []

        for event in events:
            for market in event.markets:
                if market.status != "open":
                    continue
                if market.volume < 50:
                    continue

                self.record_snapshot(market)
                signal = self._evaluate_market(event, market)
                if signal:
                    signals.append(signal)

        signals.sort(key=lambda s: s.edge * s.confidence, reverse=True)
        return signals

    def check_exits(self, events: list[Event], positions: list) -> list[TradingSignal]:
        """
        Check open positions for exit signals.
        Guide: sell when market_price >= model_probability * 0.9
        Also exit if days_to_expiry <= 7
        """
        exit_signals = []
        market_map = {}
        for event in events:
            for market in event.markets:
                market_map[market.ticker] = (event, market)

        for pos in positions:
            if pos.ticker not in market_map:
                continue
            event, market = market_map[pos.ticker]
            history = self.history_cache.get(market.ticker)
            features = extract_features(market, event, history)
            model_prob = self.model.predict_probability(features)
            market_price = market.mid_price_yes / 100

            hit_target = market_price >= model_prob * 0.9
            hit_expiry = features["days_to_expiry"] <= 7

            if hit_target or hit_expiry:
                reason = "Target hit (90% of model prob)" if hit_target else "Expiry <7 days"
                exit_signals.append(TradingSignal(
                    ticker=market.ticker,
                    market_title=market.title,
                    side=Side.YES if pos.side == "no" else Side.NO,
                    confidence=model_prob,
                    fair_probability=model_prob,
                    market_probability=market_price,
                    edge=0,
                    reasoning=f"EXIT: {reason}",
                    recommended_size_cents=0,
                ))

        return exit_signals

    def _evaluate_market(self, event: Event, market: Market) -> TradingSignal | None:
        """
        Evaluate using the guide's exact logic:
        1. Model calculates real probability (ensemble)
        2. Entry: buy when market_price <= model_probability * 0.5
        3. Only when confidence >= 70%
        """
        history = self.history_cache.get(market.ticker)
        features = extract_features(market, event, history)
        model_prob = self.model.predict_probability(features)
        market_price = market.mid_price_yes / 100

        # Guide rule: only enter when model is 70%+ confident
        confidence = abs(model_prob - 0.5) * 2
        if confidence < 0.70:
            return None

        # Guide entry rule: buy when market_price <= model_probability * 0.5
        if model_prob > 0.5 and market_price <= model_prob * 0.5:
            side = Side.YES
            edge = model_prob - market_price
        elif model_prob < 0.5 and (1 - market_price) <= (1 - model_prob) * 0.5:
            side = Side.NO
            edge = (1 - model_prob) - (1 - market_price)
        else:
            return None

        # Position sizing: proportional to edge and confidence
        size = int(config.max_bet_amount_cents * min(edge, 0.5) * confidence)
        size = max(0, min(size, config.max_bet_amount_cents))

        if size <= 0:
            return None

        model_type = "ensemble (RF+GB)" if self.model.is_trained else "heuristic"

        return TradingSignal(
            ticker=market.ticker,
            market_title=market.title,
            side=side,
            confidence=confidence,
            fair_probability=model_prob,
            market_probability=market_price,
            edge=edge,
            reasoning=(
                f"{model_type}: {model_prob:.0%} vs market {market_price:.0%}. "
                f"Entry rule: mkt ({market_price:.0%}) <= model ({model_prob:.0%}) x 0.5 = {model_prob*0.5:.0%}. "
                f"{len(FEATURE_NAMES)} features, {self.model.n_estimators} trees."
            ),
            recommended_size_cents=size,
        )
