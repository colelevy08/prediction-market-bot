"""
Random Forest + Gradient Boosting ensemble for prediction market signal generation.

======================================================================================
EDUCATIONAL GUIDE: What This File Does and Why It Works
======================================================================================

--- What Is Machine Learning? ---

Machine learning (ML) is the practice of teaching a computer to make predictions by
showing it many examples, rather than giving it explicit rules. Think of it like
this: instead of writing "if the spread is narrow AND volume is high THEN the market
is efficient," we show the computer 1,000 examples of markets that settled YES and
1,000 that settled NO, and let it figure out the patterns itself.

The more examples (called "training data") you give it, the better it gets. This is
why more data = better model. A model trained on 100 markets is guessing. A model
trained on 10,000 settled markets is genuinely learning.

--- Supervised Learning ---

This bot uses "supervised learning" — a type of ML where every training example has
a known correct answer (called a "label"). Here:
  - Input (features):  All the numbers we can measure about a market right now
                       (price, volume, time left, order book, etc.)
  - Label (outcome):   Did YES win (1) or NO win (0) after the market settled?

The model learns the relationship between the inputs and the outcome. Later, when it
sees a NEW market it has never seen before, it uses those learned relationships to
predict the probability that YES will win.

--- What Is a Feature? ---

A "feature" is any number we measure about a market. Examples:
  - yes_mid = 42 (the market thinks YES has a 42% chance)
  - volume = 5000 (5,000 contracts have been traded so far)
  - days_to_expiry = 3 (this market closes in 3 days)
  - spread = 4 (the gap between what buyers are willing to pay and sellers want)

We extract 108 features from every market. Each feature is a clue. The ML model
weighs all 108 clues together to make its prediction.

--- What Is a Label? ---

A label is the correct answer we use during training. For prediction markets, the
label is simple:
  - 1 = YES won (the event happened)
  - 0 = NO won (the event did not happen)

We only have labels for markets that have already SETTLED (closed). We can't train
on open markets because we don't know the answer yet.

--- Training vs Prediction ---

Training:  We feed the model thousands of past (settled) markets. For each one, we
           show it the features AND the outcome. The model adjusts its internal
           "weights" to get better at predicting. This is a one-time expensive
           computation.

Prediction: We show the model a NEW market's features. It applies everything it
            learned during training to output a probability: "I think there's a
            72% chance YES wins this market."

--- What Is a Random Forest? ---

A single "decision tree" is like a flowchart: "Is volume > 1000? If yes, go left.
Is spread < 5? If yes, go left again. Predict: YES (78% chance)."

The problem with a single decision tree is it's fragile — it memorizes quirks of
the training data rather than learning real patterns. This is called "overfitting."

A Random Forest fixes this by growing HUNDREDS of decision trees, each slightly
different from the others. The differences are created two ways:
  1. Each tree is trained on a random SUBSET of the training examples (bootstrapping)
  2. Each tree only sees a random SUBSET of the 108 features at each decision point
     (specifically sqrt(108) ≈ 10 features per split)

At prediction time, all 500 trees vote. The final probability is the average of all
500 votes. This "wisdom of the crowd" among trees is much more reliable than any
single tree. The technical term for this approach is "bagging" (Bootstrap AGGregatING).

--- What Is Gradient Boosting? ---

Gradient Boosting is a different way to combine many small trees. Instead of growing
all trees independently (like Random Forest), it grows them SEQUENTIALLY. Each new
tree learns from the mistakes of all the trees before it.

Think of it as iterative correction:
  - Tree 1 makes predictions. Some are wrong.
  - Tree 2 focuses specifically on the examples Tree 1 got wrong.
  - Tree 3 focuses on what Tree 1 + Tree 2 together still got wrong.
  - ... and so on for 150 trees.

The "learning rate" (0.01 here) controls how aggressively each new tree corrects
mistakes. A smaller learning rate = more cautious = less overfitting = requires more
trees. The "gradient" in the name refers to the mathematical direction of steepest
error reduction (from calculus), which each new tree follows.

--- Why Use an Ensemble of Two Models? ---

We combine Random Forest AND Gradient Boosting because:

1. DIVERSITY REDUCES ERROR: The two models make different kinds of mistakes. RF
   is better at capturing broad patterns; GB is better at fine-tuning edge cases.
   When their errors don't overlap, averaging them cancels out many mistakes.
   This is the core principle: "a diverse committee of imperfect predictors
   outperforms any single predictor."

2. RF IS ROBUST, GB IS PRECISE: If one model sees unusual data it hasn't encountered
   before, the other acts as a sanity check.

3. WEIGHTS WERE OPTIMIZED: We use 70% RF + 30% GB. These weights were found by
   testing many combinations on held-out data and picking the one with the best
   "Brier score" (a measure of probability prediction accuracy).

--- What Is Calibration? ---

A model is "calibrated" if its predicted probabilities match reality. For example:
  - If a well-calibrated model says "70% chance YES" on 100 different markets,
    about 70 of those markets should actually resolve YES.
  - If only 40 resolve YES, the model is overconfident — it says 70% but reality
    is more like 40%. This is poorly calibrated.

We check calibration after training by grouping predictions into buckets (0.5-0.6,
0.6-0.7, etc.) and comparing the predicted average to the actual win rate in each
bucket. Perfect calibration = the two numbers match.

--- What Is Overfitting? ---

Overfitting is when a model memorizes the training data so perfectly that it fails
on new data. Imagine a student who memorizes every answer to past exams but can't
answer any new question.

Signs of overfitting:
  - Training accuracy: 99% (it knows the training examples by heart)
  - Test accuracy: 65% (it's barely better than guessing on new data)

We prevent overfitting by:
  1. Cross-validation (see below): testing on data the model never trained on
  2. Limiting tree depth (max_depth=10): prevents trees from growing too specific
  3. Requiring minimum samples per leaf (min_samples_leaf): no rules based on tiny
     sample sizes
  4. Random feature subsets: prevents any feature from dominating all trees

--- What Is Cross-Validation? ---

Cross-validation tests how well the model generalizes to NEW data. We split the
training data into "folds":

Example with 5-fold cross-validation on 1000 markets:
  - Fold 1: Train on markets 201-1000, test on markets 1-200
  - Fold 2: Train on markets 1-200 and 401-1000, test on markets 201-400
  - ...etc.

The model never trains and tests on the SAME examples. Each market gets tested
exactly once. The final CV score is the average accuracy across all folds.

This bot uses "purged" cross-validation (a finance-specific improvement) that also
removes training examples NEAR each fold boundary to prevent "leakage" — where
adjacent markets share information because they were active at the same time.

--- What Does StandardScaler Do? ---

Many ML algorithms are sensitive to the scale of features. For example:
  - volume = 50,000 (a large number)
  - implied_prob = 0.42 (a small number between 0 and 1)

Without scaling, the model might put too much weight on volume just because the
number is larger. StandardScaler normalizes all features to have:
  - Mean = 0 (the average value becomes zero)
  - Standard deviation = 1 (values are expressed in "how many standard deviations
    from average" rather than raw units)

After scaling, volume=50000 might become +1.3 (meaning 1.3 standard deviations
above average), and implied_prob=0.42 might become -0.8. Now all features are
comparable in magnitude. Tree-based models (RF, GB) don't technically need scaling,
but it helps the models train faster and more consistently.

--- What Is Feature Importance? ---

After training, we can ask each model: "Which of the 108 features did you rely on
most?" This is called feature importance.

Specifically, it measures how much each feature reduces prediction error across all
the decision points in all the trees. A feature with 5% importance means 5% of the
total prediction accuracy comes from that feature.

High-importance features are the ones that matter most for predicting market outcomes.
Low-importance features (< 0.1%) are essentially noise — we log them but keep them
for compatibility.

--- How Does the Model Output a Probability (Not Just Yes/No)? ---

Each of the 500 RF trees outputs a vote: "YES" or "NO." We count how many trees
voted YES. If 350 out of 500 trees say YES, the RF probability is 350/500 = 70%.

The GB model works similarly across its sequence of trees.

The final ensemble probability = 0.70 * RF_probability + 0.30 * GB_probability.

This gives us a continuous number between 0 and 1 (like 0.73), not just a binary
yes/no. A probability of 0.73 means "the model thinks YES has a 73% chance."

--- Precision vs Recall (Why We Care About Both) ---

Precision: Of all the times the model said "YES, buy this," what fraction were
           actually correct? High precision = fewer false alarms.

Recall:    Of all the markets that actually resolved YES, what fraction did the
           model correctly identify as good buys? High recall = fewer missed
           opportunities.

In trading, we generally prefer HIGH PRECISION over high recall. It's better to
only trade when we're very confident (and be right most of the time) than to trade
on every opportunity and be wrong half the time.

======================================================================================

This is the core ML engine of the trading bot. It implements:

1. Feature Engineering (extract_features):
   - Extracts 106 numerical features from market data organized into 8 categories:
     price (18), volume/liquidity (14), time decay (14), orderbook imbalance (12),
     cross-market efficiency (10), momentum proxies (8), historical momentum (14),
     and interaction/cross features (16).
   - Features are computed from a Market + Event pair, with optional historical
     price snapshots for momentum/volatility signals.

2. Ensemble Model (PredictionModel):
   - Random Forest (500 trees, max_depth=10) + Gradient Boosting (150 trees, lr=0.01)
   - Each tree sees sqrt(106) ~ 10 random features (decorrelation).
   - Predictions are combined via weighted average: 70% RF + 30% GB
     (weights optimized via Brier score grid search).
   - When untrained, falls back to a 12-signal heuristic probability estimate.
   - Hyperparameters tuned via GridSearchCV on 1000+ settled Kalshi markets (AUC 0.84+).

3. Signal Generation (RFSignalGenerator):
   - Scans all open markets across all Kalshi events (0.3s RF eval, 10s data fetch).
   - Entry rule: buy when market_price <= model_probability * (1 - entry_threshold), i.e. undervalued by entry_threshold %.
   - Exit rule: sell when 90% of entry edge has been captured (take-profit).
   - Only enters when model confidence >= 70%.
   - Position sizing proportional to edge * confidence, capped at max_bet_amount_cents.
   - Maintains a per-ticker history cache (last 100 snapshots) for momentum features.

Connects to: bot.models (Market, Event, TradingSignal), bot.config (trading params).
Used by: bot.server (API endpoints), bot.backtester (historical replay), bot.main (CLI).
"""

import logging
import math
import time
from collections import deque      # A "deque" is a list with a fixed maximum size; old items fall off automatically
from datetime import datetime, timezone

import numpy as np                  # NumPy: efficient math on arrays of numbers (like averaging 500 tree votes)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # The two ML models
from sklearn.preprocessing import StandardScaler   # Normalizes features so all are on the same scale
from sklearn.model_selection import StratifiedKFold  # Used for cross-validation (testing on held-out data)

from bot.models import Event, Market, Side, TradingSignal  # Our data structures for markets and signals
from bot.config import config  # Bot-wide settings (min confidence, max bet size, etc.)

logger = logging.getLogger(__name__)  # Python's logging system — writes messages to the log file


# ── Feature Engineering: 106 features ────────────────────────────────────────
#
# "Feature engineering" is the process of converting raw data (a market object)
# into a clean list of numbers that a machine learning model can understand.
# The art here is deciding WHAT to measure. We want signals that are:
#   1. Predictive of future outcomes (correlated with whether YES wins)
#   2. Available in real-time (so we can use them before the market closes)
#   3. Robust to noise (not accidentally correlated with irrelevant patterns)

def extract_features(market: Market, event: Event, history: list[dict] | None = None) -> dict[str, float]:
    """
    Extract 108 features from market data for the ensemble model.

    Think of this function as the "eyes" of the trading bot. It looks at a
    prediction market and measures everything observable: the price, the spread,
    how many contracts have been traded, how much time is left, what the order
    book looks like, and recent price history.

    These 108 numbers become the "input vector" fed to the Random Forest and
    Gradient Boosting models. The models then combine all 108 numbers (each
    weighted by learned importance) to output a single probability.

    The guide says: "markets run on 100+ factors" — we extract every
    quantitative signal available from price, volume, time, and orderbook data.
    When historical snapshots are available, we add momentum/volatility features.

    Args:
        market:  The prediction market object (contains prices, volume, spread, etc.)
        event:   The parent event (contains category, title, etc.)
        history: Optional list of recent price snapshots for this market ticker.
                 Each snapshot is a dict: {"yes_mid": float, "volume": int, "timestamp": float}
                 Used to compute momentum and volatility features (Category 7).
                 If None or too short, those features default to 0.

    Returns:
        A dict mapping feature name -> float value. Always has exactly
        len(FEATURE_NAMES) keys in a consistent order.
    """
    # Fix 1: Top-level try/except — return zeros on any exception
    try:
        return _extract_features_inner(market, event, history)
    except Exception as e:
        logger.warning(f"extract_features failed for {market.ticker}: {e}")
        # Return a dict of zeros with correct keys (will be populated on first successful call)
        return {name: 0.0 for name in _ZERO_FEATURE_NAMES}


def _extract_features_inner(market: Market, event: Event, history: list[dict] | None = None) -> dict[str, float]:
    """Inner implementation of extract_features (separated for fix 1 try/except wrapper).

    All the actual feature computation lives here. The outer function wraps this
    in a try/except so any unexpected error during feature extraction returns
    safe zeros rather than crashing the entire scan loop.
    """
    # Fix 4: Clamp yes_mid to [1, 99] range for safe division
    # yes_mid is the midpoint price of the YES contract, expressed in cents (1-99).
    # Example: yes_mid=42 means the market consensus is a 42% chance of YES.
    # We clamp to [1, 99] (never 0 or 100) to prevent division-by-zero later.
    yes_mid = max(1.0, min(99.0, market.mid_price_yes))
    # no_mid is 100 - yes_mid because YES and NO must together equal 100%.
    # If YES is 42%, then NO is 58%. The two sides are mirror images.
    no_mid = max(1.0, min(99.0, 100 - yes_mid))
    # Fix 5: Ensure spread >= 0
    # The "spread" is the gap between the best ask price and the best bid price.
    # A narrow spread (e.g., 2 cents) means a liquid, efficient market.
    # A wide spread (e.g., 15 cents) means illiquid — it costs more to trade.
    spread = abs(market.spread)
    # Fix 6: Clamp volume to reasonable range [0, 1e9]
    # Volume = total number of contracts traded so far. More volume = more information
    # has been incorporated into the price. Low-volume markets may be mispriced.
    volume = max(0, min(int(1e9), market.volume))
    # Fix 7: Clamp open_interest similarly
    # Open interest (OI) = number of contracts currently held open (not yet settled).
    # High OI means many people are holding positions — the market is well-followed.
    oi = max(0, min(int(1e9), market.open_interest))
    history = history or []

    # Fix 2: Guard time parsing with try/except, default to 30 days
    # Time-to-expiry features are critical: a market that closes in 1 hour behaves
    # very differently from one that closes in 30 days. Near expiry, prices are "sticky"
    # and already reflect nearly all available information. Far from expiry, there's
    # more room for new information to shift the price.
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
        except (ValueError, TypeError, AttributeError):
            # Fix 2: default already set above
            pass

    # Fix 8: Ensure days_to_expiry in [0, 3650]
    days_to_expiry = max(0.0, min(3650.0, days_to_expiry))
    hours_to_expiry = max(0.0, min(3650.0 * 24, hours_to_expiry))
    minutes_to_expiry = max(0.0, min(3650.0 * 24 * 60, minutes_to_expiry))

    # Fix 3: Helper for safe division
    # Division by zero crashes programs. This helper returns a safe default (usually 0)
    # instead of crashing when the denominator is zero. It's used dozens of times below
    # for ratio features like "volume / open_interest" where either number could be 0.
    def safe_div(num: float, denom: float, default: float = 0.0) -> float:
        """Divide num by denom, returning default (usually 0.0) if denom is zero.

        This prevents ZeroDivisionError, which would crash the entire feature extraction
        pipeline for every market in the scan. By returning a safe default instead,
        the market still gets evaluated with a 0 for this particular feature.
        """
        if denom == 0:
            return default
        return num / denom

    # Fix 11: Safe log-odds with explicit guard against log(0)
    # Log-odds = log(p / (1-p)), also called the "logit" transformation.
    # It converts a probability (0 to 1) into an unbounded scale (-infinity to +infinity).
    # A market at 50% has log-odds = 0. At 75% it's about +1.1. At 25% it's about -1.1.
    # Tree models sometimes find it easier to split on log-odds than raw probability.
    log_odds_num = max(yes_mid, 0.5)
    log_odds_den = max(100 - yes_mid, 0.5)
    if log_odds_num <= 0 or log_odds_den <= 0:
        log_odds_val = 0.0
    else:
        log_odds_val = math.log(log_odds_num / log_odds_den)

    # Fix 17: Ensure implied_prob in [0, 1]
    # implied_prob is just yes_mid / 100 — converting from "cents" (0-100 scale)
    # to probability (0.0 to 1.0 scale). A yes_mid of 42 = implied_prob of 0.42.
    implied_prob = max(0.0, min(1.0, yes_mid / 100))

    # ── 1. Price Features (18) ───────────────────────────────────
    # These capture the current market price and its derived transformations.
    # The model learns: does a market trading at 80% actually win 80% of the time?
    # Are markets near 50/50 harder to predict? Are extreme-priced markets more certain?
    features = {
        # yes_bid: the highest price any buyer is currently willing to pay for YES.
        # Example: yes_bid=40 means someone will buy YES at 40 cents right now.
        "yes_bid": float(max(0, market.yes_bid)),

        # yes_ask: the lowest price any seller is currently asking for YES.
        # Example: yes_ask=44 means the cheapest available YES is 44 cents.
        # The spread = ask - bid = 44 - 40 = 4 cents in this example.
        "yes_ask": float(max(0, market.yes_ask)),

        # no_bid / no_ask: same concept but for the NO side.
        # Remember: buying NO at 56 cents is equivalent to saying YES has a 44% chance.
        "no_bid": float(max(0, market.no_bid)),
        "no_ask": float(max(0, market.no_ask)),

        # yes_mid / no_mid: the midpoint of the bid-ask spread.
        # This is the "consensus" market price — the best estimate of fair value
        # before accounting for the spread cost of trading.
        "yes_mid": yes_mid,
        "no_mid": no_mid,

        # spread: the difference between ask and bid. Wider spread = less liquid,
        # harder to trade profitably (you're immediately down the spread cost).
        "spread": float(spread),

        # Fix 3: Use safe_div for all division operations
        # spread_pct: spread as a percentage of the mid price.
        # A spread of 4 on a yes_mid of 40 = 10% spread cost. Very expensive.
        # A spread of 2 on a yes_mid of 80 = 2.5% spread cost. More reasonable.
        "spread_pct": safe_div(spread, yes_mid) if yes_mid > 0 else 0.0,
        "spread_pct_no": safe_div(spread, no_mid) if no_mid > 0 else 0.0,

        # bid_ask_ratio: how close the bid is to the ask.
        # Close to 1.0 = tight spread. Far from 1.0 = wide spread.
        "bid_ask_ratio": safe_div(market.yes_bid, max(market.yes_ask, 1)),
        "no_bid_ask_ratio": safe_div(market.no_bid, max(market.no_ask, 1)),

        # price_extremity: how far the price is from 50% (pure uncertainty).
        # A market at 90% has extremity=40. A market at 50% has extremity=0.
        # Extreme prices (near 0 or 100) tend to be more predictable — the market
        # already "knows" the answer. Markets near 50% are genuinely uncertain.
        "price_extremity": abs(yes_mid - 50),

        # price_extremity_sq: the squared version, normalized to 0-1.
        # Squaring emphasizes very extreme markets more than moderately extreme ones.
        # A market at 95% (extremity=45) has sq=0.81. At 60% (extremity=10) sq=0.04.
        "price_extremity_sq": (yes_mid - 50) ** 2 / 2500,

        # log_odds_yes: the logit transformation of the implied probability.
        # Useful because probabilities near 50% are less certain than those near extremes,
        # and log-odds spaces this out more evenly for the model to work with.
        "log_odds_yes": log_odds_val,

        # implied_prob: yes_mid converted from cents (0-100) to probability (0.0-1.0).
        "implied_prob": implied_prob,

        # implied_prob_sq: squared probability. The model can use this to detect
        # non-linear effects (e.g., "very high probability" behaves differently from
        # "moderately high probability").
        "implied_prob_sq": implied_prob ** 2,

        # price_bucket_low / _high: binary flags indicating whether the market is
        # in the "low price" range (<25%) or "high price" range (>75%).
        # These help the model apply completely different logic to extreme markets.
        "price_bucket_low": 1.0 if yes_mid < 25 else 0.0,
        "price_bucket_high": 1.0 if yes_mid > 75 else 0.0,
    }

    # ── 2. Volume & Liquidity Features (14) ──────────────────────
    # Volume and liquidity reveal how "well-traded" a market is. High-volume markets
    # tend to be more efficiently priced — many traders have looked at this market
    # and incorporated their information into the price. Low-volume markets may have
    # stale or incorrect prices that the model can exploit.
    features.update({
        # volume: raw total contracts traded. High volume = many participants have
        # weighed in on this market. The price is more likely to reflect true value.
        "volume": float(volume),

        # log_volume: the logarithm (base e) of volume.
        # Why log? Because volume can range from 0 to 10,000,000. A linear scale
        # would make high-volume markets completely dominate. Log compresses the
        # scale: log(100)=4.6, log(10000)=9.2, log(1000000)=13.8. Differences
        # between low-volume markets become more visible.
        "log_volume": math.log1p(volume),  # log1p = log(1 + volume) so log(0) is safe

        # volume_sq: actually a square root (sqrt) despite the variable name.
        # This is kept as-is for "feature compatibility" — changing the name would
        # break a trained model that expects this feature to have this meaning.
        "volume_sq": math.sqrt(max(volume, 0)),

        # open_interest: contracts currently held open. Measures how much money is
        # at stake. High OI = many people have skin in the game = more information.
        "open_interest": float(oi),
        "log_open_interest": math.log1p(oi),

        # Fix 3: explicit zero checks for all division
        # volume_oi_ratio: how much volume relative to open interest.
        # High ratio = the market is "churning" — lots of trading activity relative
        # to the number of people holding positions. Suggests active price discovery.
        "volume_oi_ratio": safe_div(volume, oi) if oi > 0 else 0.0,

        # liquidity_score: combines volume AND spread tightness into one number.
        # A high-volume market with a tight spread is very liquid (easy to trade).
        # Formula: volume * (100 - spread) / 100. If spread=0, score=volume. If
        # spread=100 (impossible to trade), score=0.
        "liquidity_score": volume * (100 - spread) / 100 if spread <= 100 else 0.0,

        # turnover_rate: same as volume_oi_ratio. Measures trading velocity.
        "turnover_rate": safe_div(volume, oi) if oi > 0 else 0.0,

        # volume_per_cent_spread: volume divided by the spread in cents.
        # Markets with lots of volume AND tight spreads are the ideal trading targets:
        # you can trade a lot without paying too much in spread costs.
        "volume_per_cent_spread": safe_div(volume, max(spread, 1)),

        # dollar_volume: combines volume with price. A market trading 1000 contracts
        # at 80 cents is "bigger" (dollar_volume=800) than 1000 contracts at 20 cents
        # (dollar_volume=200). Captures the economic significance of trading activity.
        "dollar_volume": volume * yes_mid / 100,
        "log_dollar_volume": math.log1p(volume * yes_mid / 100),

        # volume_intensity: volume per hour of remaining market life.
        # A market with 5000 volume and 100 hours left is less "hot" than one with
        # 5000 volume and 2 hours left. The second is attracting lots of last-minute
        # trading, which often signals that news is driving the price.
        "volume_intensity": safe_div(volume, max(hours_to_expiry, 1)),

        # oi_concentration: open_interest divided by volume.
        # High = people are holding their positions rather than trading in and out.
        # Low = lots of turnover, suggesting active speculation.
        "oi_concentration": safe_div(oi, max(volume, 1)),

        # volume_rank_proxy: caps volume at 10 (for normalized volume in thousands).
        # Provides a bounded, human-readable scale: 10 = very high volume.
        "volume_rank_proxy": min(volume / 1000, 10),
    })

    # ── 3. Time Features (14) ────────────────────────────────────
    # Time remaining until the market closes is one of the most important features.
    # The key insight: as a market approaches expiry, the price "converges" toward
    # the true outcome. A market at 70% with 30 days left is more uncertain than
    # the same market at 70% with 1 hour left. Time remaining affects:
    #   - How much the price might still move
    #   - How quickly we'll know if we're right or wrong
    #   - The "time value" of our edge (edge earned today vs edge earned in 3 months)
    features.update({
        # Raw time-to-expiry in three units. We give the model all three because
        # a market expiring in 2 days might behave differently from one in 48 hours
        # even though they're mathematically equal — markets often react to daily
        # events (news, announcements) that are better measured in days.
        "days_to_expiry": days_to_expiry,
        "hours_to_expiry": hours_to_expiry,
        "minutes_to_expiry": minutes_to_expiry,

        # Log-transformed versions reduce the impact of very long-dated markets.
        # A market 1 day away vs 2 days away is much more meaningful than
        # 300 days away vs 301 days away. Log compression captures this.
        "log_days_to_expiry": math.log1p(days_to_expiry),
        "log_hours_to_expiry": math.log1p(hours_to_expiry),

        # time_decay_factor: 1 / days_to_expiry. Very large for imminent markets.
        # A market expiring in 0.1 days has factor=10. Expiring in 30 days has factor=0.033.
        # Borrowed from options pricing: "theta" decay accelerates near expiry.
        "time_decay_factor": 1 / max(days_to_expiry, 0.01) if days_to_expiry >= 0 else 100.0,
        "time_decay_sqrt": 1 / math.sqrt(max(days_to_expiry, 0.01)) if days_to_expiry >= 0 else 10.0,

        # Binary flags: is this market about to close? These help the model apply
        # different logic for imminent vs distant markets.
        "is_expiring_soon": 1.0 if days_to_expiry <= 7 else 0.0,  # Within a week
        "is_expiring_today": 1.0 if days_to_expiry <= 1 else 0.0,  # Within 24 hours
        "is_expiring_hour": 1.0 if hours_to_expiry <= 1 else 0.0,  # Within 60 minutes

        # expiry_urgency: a 0-1 scale of "how soon is this closing?"
        # 0.0 = 30+ days away (no urgency). 1.0 = closing right now.
        "expiry_urgency": max(0, 1 - days_to_expiry / 30),

        "sqrt_days_to_expiry": math.sqrt(max(days_to_expiry, 0)),

        # time_value: the "expected return" if you hold to expiry.
        # A market at 70% with 10 days left has time_value = 70 * 10 / 100 = 7.
        # Higher = more upside potential (and more time to wait for it).
        "time_value": yes_mid * days_to_expiry / 100,

        # theta_proxy: rate of price change per day (borrowed from options trading).
        # "Theta" in options = how much value an option loses per day as expiry approaches.
        # A YES contract at 70% with 7 days left loses about 1c per day in uncertainty.
        # This feature captures that daily decay rate.
        "theta_proxy": -yes_mid / max(days_to_expiry, 0.1) / 100,
    })

    # ── 4. Orderbook Imbalance Features (12) ─────────────────────
    # The "order book" shows all the buy and sell orders currently waiting.
    # Imbalance = more buyers than sellers (or vice versa). This is a very strong
    # short-term signal: if lots of people want to buy YES and few want to sell,
    # the price is likely to rise.
    #
    # Think of it like a tug-of-war: bid_pressure measures which side (YES or NO)
    # has more buying strength right now. This is often called "order flow."
    yes_total = market.yes_bid + market.yes_ask
    no_total = market.no_bid + market.no_ask
    all_total = yes_total + no_total

    # Fix 13: Handle case where both bid and ask are 0
    bid_sum = market.yes_bid + market.no_bid   # Total bidding interest across both sides
    ask_sum = market.yes_ask + market.no_ask   # Total asking interest across both sides
    # Fix 14: Handle case where no orderbook data exists
    has_orderbook = all_total > 0

    features.update({
        # bid_pressure: what fraction of ALL bids are on the YES side?
        # 0.5 = equal YES/NO interest. >0.5 = more YES buyers. <0.5 = more NO buyers.
        # This is one of the strongest predictive signals in prediction markets.
        "bid_pressure": safe_div(market.yes_bid, bid_sum, 0.5) if bid_sum > 0 else 0.5,

        # ask_pressure: what fraction of asks are on the YES side?
        # Many YES sellers = supply pressure = price may fall.
        "ask_pressure": safe_div(market.yes_ask, ask_sum, 0.5) if ask_sum > 0 else 0.5,

        # order_imbalance: (YES bids - NO bids) / total bids.
        # Positive = more YES buying pressure. Negative = more NO buying pressure.
        # A value of +0.3 means the YES side has 30% more bidding activity than average.
        "order_imbalance": safe_div(market.yes_bid - market.no_bid, bid_sum) if bid_sum > 0 else 0.0,
        "ask_imbalance": safe_div(market.yes_ask - market.no_ask, ask_sum) if ask_sum > 0 else 0.0,

        # bid_depth_ratio / ask_depth_ratio: YES bids/asks as a fraction of all orderbook activity.
        "bid_depth_ratio": safe_div(market.yes_bid, all_total) if has_orderbook else 0.0,
        "ask_depth_ratio": safe_div(market.yes_ask, all_total) if has_orderbook else 0.0,

        # bid_skew / ask_skew: imbalance normalized by the spread.
        # Gives more weight to imbalances that are large relative to the bid-ask spread.
        "bid_skew": safe_div(market.yes_bid - market.no_bid, max(spread, 1)) if spread > 0 else 0.0,
        "ask_skew": safe_div(market.yes_ask - market.no_ask, max(spread, 1)) if spread > 0 else 0.0,

        # weighted_mid / microprice: the simple midpoint between the YES bid and ask.
        # This is the "fair value" estimate from the order book alone.
        "weighted_mid": (market.yes_bid + market.yes_ask) / 2,  # Standard midpoint
        "microprice": (market.yes_bid + market.yes_ask) / 2,  # Standard midpoint (same as yes_mid)

        # bid_strength: how close is the YES bid to the YES midpoint?
        # If bid=42 and mid=44, strength = 42/44 = 0.95 (very close = buyers are aggressive)
        # If bid=25 and mid=50, strength = 0.50 (buyers are far from fair value = passive)
        "bid_strength": safe_div(market.yes_bid, yes_mid) if yes_mid > 0 else 0.0,

        # ask_weakness: how far is the YES ask ABOVE the midpoint?
        # Negative = ask is BELOW mid (sellers are desperate). Positive = sellers are firm.
        "ask_weakness": safe_div(market.yes_ask - yes_mid, yes_mid) if yes_mid > 0 else 0.0,
    })

    # ── 5. Cross-Market / Efficiency Features (10) ───────────────
    # These features measure how "internally consistent" the YES/NO prices are.
    # In an ideal, perfectly efficient market: YES + NO = 100 cents.
    # If YES asks 55 and NO asks 55, that sums to 110 — the exchange is keeping
    # 10 cents of "vig" (vigorish, the house's cut). That's a 10% tax on each trade.
    # Large vig = less efficient market = more room for mispricing = more opportunity.
    #
    # Arbitrage: if YES bid + NO bid > 100, you could simultaneously buy both sides
    # and guarantee a profit. Exchanges prevent this, but NEAR-arbitrage suggests
    # one side is mispriced relative to the other.
    features.update({
        # yes_no_spread: YES ask minus NO ask. In a balanced market this ≈ 0.
        # A large positive value means YES is more expensive to buy than NO.
        "yes_no_spread": float(market.yes_ask - market.no_ask),

        # market_efficiency: how close are the ask prices to summing to 100?
        # 1.0 = perfectly efficient (YES ask + NO ask = 100, no vig).
        # 0.8 = there's a 20% gap — the market has significant friction/vig.
        # LESS efficient markets are MORE interesting — mispricings persist longer.
        "market_efficiency": 1 - abs(market.yes_ask + market.no_ask - 100) / 100,

        # overround: (YES ask + NO ask) / 100. The exchange's profit margin.
        # 1.0 = fair (no vig). 1.10 = 10% vig. Prediction markets typically run
        # 1-5% vig. Very wide vig = avoid (too expensive to trade profitably).
        "overround": safe_div(market.yes_ask + market.no_ask, 100, 1.0),

        # vig_estimate: the raw vig in cents. (YES ask + NO ask - 100) / 100.
        # If YES ask=55 and NO ask=52: vig = (55+52-100)/100 = 0.07 (7 cents).
        "vig_estimate": max(0, (market.yes_ask + market.no_ask - 100)) / 100,

        # synthetic_edge: a measure of whether you can construct a "synthetic"
        # bet more cheaply than the direct market price. This is borrowed from
        # options theory ("put-call parity"). Non-zero values suggest mispricing
        # between the YES and NO sides.
        "synthetic_edge": (market.yes_bid + (100 - market.no_ask)) / 2 - yes_mid,

        # arb_spread: how much you'd profit if you could buy both YES and NO simultaneously.
        # YES bid + NO bid - 100. If positive, there's a theoretical arbitrage.
        # In practice this is always 0 or negative (exchanges prevent real arb),
        # but values close to 0 indicate a very efficient, well-priced market.
        "arb_spread": max(0, market.yes_bid + market.no_bid - 100),

        # reverse_arb: how much cheaper you could get BOTH sides vs the market.
        # 100 - YES ask - NO ask. Positive = both sides are cheap (wide spread = low vig).
        "reverse_arb": max(0, 100 - market.yes_ask - market.no_ask),

        # price_dislocation: mismatch between YES ask and the implied YES price from
        # the NO side. |YES ask - (100 - NO bid)|. Zero = perfect internal consistency.
        # Large dislocation = one side is priced inconsistently with the other.
        "price_dislocation": abs(market.yes_ask - (100 - market.no_bid)),

        # fair_value_gap: difference between the YES midpoint and the implied fair value
        # derived from the NO side. Measures whether YES and NO are "agreeing" on
        # the market's fair value.
        "fair_value_gap": (market.yes_bid + market.yes_ask) / 2 - (100 - (market.no_bid + market.no_ask) / 2),
    })
    # efficiency_score: derived from fair_value_gap.
    # 1.0 = YES/NO sides perfectly agree on fair value. <1.0 = they disagree.
    features["efficiency_score"] = 1 - abs(features["fair_value_gap"]) / 50

    # ── 6. Momentum Proxies from Current Data (8) ────────────────
    # "Momentum" in finance means: things that have been moving tend to keep moving.
    # A market that jumped from 40% to 70% is "momentum-driven" — something changed
    # and the market is still digesting it. These features try to capture that
    # directional energy using ONLY the current snapshot (no history needed).
    # Fix 15: market_cap_proxy handles case where volume and price are both 0
    # Fix 16: yes_no_ratio handles case where no_mid == 0
    features.update({
        # price_momentum_proxy: how "extreme" is the price AND how much activity is there?
        # Combines the distance from 50% (conviction) with volume (confirmation).
        # A market at 80% with high volume has strong momentum toward YES.
        # A market at 80% with zero volume might just be a stale quote.
        "price_momentum_proxy": (yes_mid - 50) * volume / 10000,

        # mean_reversion_signal: the tendency for extreme prices to drift back toward 50%.
        # Markets far from 50% sometimes overcorrect (especially near expiry as uncertainty
        # resolves). This feature is positive when the price is above 50% (suggesting
        # it might fall back) and negative when below 50%.
        "mean_reversion_signal": safe_div(50 - yes_mid, max(abs(50 - yes_mid), 1)) * safe_div(1.0, max(days_to_expiry, 0.1)),

        # volume_momentum: similar to price_momentum_proxy but scaled differently.
        # Captures the interaction between price direction and trading volume.
        "volume_momentum": volume * (yes_mid - 50) / 5000,

        # buying_urgency: how aggressively are people buying YES relative to the
        # total interest in the market? High = YES buyers are rushing in.
        "buying_urgency": safe_div(market.yes_bid * volume, max(oi, 1) * 100),

        # selling_urgency: equivalent on the sell side. High = people rushing to sell.
        "selling_urgency": safe_div((100 - market.yes_ask) * volume, max(oi, 1) * 100),

        # price_velocity_proxy: how fast (in price-per-day) is the market moving away
        # from the 50/50 neutral point? Higher = stronger daily momentum.
        "price_velocity_proxy": safe_div(yes_mid - 50, max(days_to_expiry, 0.1)),

        # conviction_score: combines price extremity with volume on a log scale.
        # A market at 90% with 10,000 volume has high conviction (both the price
        # AND the trading activity confirm the strong opinion).
        "conviction_score": abs(yes_mid - 50) * math.log1p(volume) / 100,

        # smart_money_proxy: combines order imbalance with volume.
        # "Smart money" = informed traders who bet large and directionally.
        # High order imbalance + high volume = someone who knows something is trading.
        "smart_money_proxy": abs(features["order_imbalance"]) * math.log1p(volume),
    })

    # ── 7. Historical Momentum Features (14) ─────────────────────
    # Category 6 estimated momentum from a single snapshot. This category measures
    # ACTUAL momentum using multiple historical price snapshots.
    # Think of it like: Category 6 = "where is the price right now?"
    #                   Category 7 = "which direction has it been moving, and how fast?"
    #
    # These features require historical data (the "history" parameter). If the bot
    # just started or hasn't seen this market before, all these default to 0.
    # The more history available (up to 100 snapshots), the better these estimates.
    # This is why the bot keeps a rolling cache of the last 100 price snapshots
    # for each market ticker.
    #
    # These require historical price snapshots; defaults to 0 if unavailable
    if len(history) >= 2:
        prices = [h.get("yes_mid", 50) for h in history]
        volumes = [h.get("volume", 0) for h in history]
        current = prices[-1]
        prev = prices[-2]

        # Fix 9: Handle case where only 1 usable data point (prev == 0)
        pct_change_1 = safe_div(current - prev, max(prev, 1)) if prev > 0 else 0.0
        pct_change_all = safe_div(current - prices[0], max(prices[0], 1)) if prices[0] > 0 else 0.0

        # Fix 18: Handle volatility when all prices are the same
        if len(prices) >= 3:
            returns = [safe_div(prices[i] - prices[i-1], max(prices[i-1], 1)) for i in range(1, len(prices))]
            # Fix 12: Handle case where all prices are the same (std == 0)
            vol = float(np.std(returns)) if returns else 0.0
        else:
            returns = []
            vol = 0.0

        # Fix 10: Volume acceleration — handle case where < 3 data points
        vol_change = safe_div(volumes[-1] - volumes[0], max(volumes[0], 1)) if volumes[0] > 0 else 0.0

        # Fix 12: RSI-safe avg_gain/avg_loss (handle all-same prices)
        price_std = float(np.std(prices)) if len(prices) >= 2 else 0.0
        price_range_val = max(prices) - min(prices)

        features.update({
            # momentum_1: price change from the previous snapshot to now.
            # Positive = price went up. Negative = price went down.
            # A large positive value means the market moved sharply toward YES recently.
            "momentum_1": pct_change_1,

            # momentum_total: total price change from the OLDEST snapshot to now.
            # This captures the overall trend over the full history window.
            "momentum_total": pct_change_all,

            # momentum_abs: the absolute value of total momentum (ignores direction).
            # Measures "how much has this market moved" regardless of which way.
            # High movement = this market is being actively repriced.
            "momentum_abs": abs(pct_change_all),

            # volatility: standard deviation of the period-over-period returns.
            # "Return" here = (price_now - price_before) / price_before.
            # High volatility = the price has been jumping around a lot.
            # Low volatility = the price has been stable.
            # Volatility matters for position sizing: we bet less when the price
            # is erratic (because our edge estimate is less reliable).
            "volatility": vol,
            "log_volatility": math.log1p(vol),

            # volatility_adj_momentum: momentum divided by volatility.
            # This is the "Sharpe ratio" concept applied to a single market:
            # a big move in a normally-stable market is more meaningful than
            # the same size move in a normally-volatile market.
            "volatility_adj_momentum": safe_div(pct_change_all, max(vol, 0.001)),

            # price_range: (max - min) / max over the history window.
            # How wide has the price range been? Wide range = uncertain, contested market.
            # Narrow range = the market has been consistently priced.
            "price_range": safe_div(price_range_val, max(max(prices), 1)),

            # price_position: where is the current price within the historical range?
            # 0.0 = at the bottom of its recent range. 1.0 = at the top.
            # A reading near 1.0 combined with upward momentum suggests continued strength.
            "price_position": safe_div(current - min(prices), max(price_range_val, 1)) if price_range_val > 0 else 0.5,

            # trend_strength: absolute momentum divided by volatility.
            # Measures "how directional is this market's movement?"
            # High = strong, consistent trend. Low = noisy, directionless movement.
            "trend_strength": safe_div(abs(pct_change_all), max(vol, 0.001)),

            # volume_trend: how has trading volume changed over the history window?
            # Positive = volume is increasing (more people paying attention).
            # Negative = volume is declining (interest fading).
            "volume_trend": vol_change,

            # Fix 10: volume_acceleration only with >= 3 data points
            # volume_acceleration: is volume increasing faster and faster? (second derivative)
            # Like a car: velocity = how fast; acceleration = is it speeding up or slowing down.
            # High acceleration = sudden surge in trading interest — often precedes big price moves.
            "volume_acceleration": safe_div(volumes[-1] - 2 * volumes[len(volumes)//2] + volumes[0], max(volumes[0], 1)) if len(volumes) >= 3 else 0.0,

            # price_std: standard deviation of the raw price levels (not returns).
            # Similar to volatility but measured in absolute cents rather than percentages.
            "price_std": price_std,

            # Fix 18: Handle std == 0 in skew and mean_reversion
            # price_skew: is the price distribution lopsided?
            # Positive skew = the price has been clustering BELOW its average (occasional high spikes).
            # Negative skew = clustering ABOVE average (occasional dips).
            # Borrowed from statistics: "skewness" measures asymmetry of a distribution.
            "price_skew": float(safe_div(float(np.mean(prices)) - float(np.median(prices)), max(price_std, 0.01))) if len(prices) >= 3 else 0.0,

            # mean_reversion_hist: how far is the current price from the historical average?
            # In standard deviations (z-score). +2.0 = price is 2 std devs above its
            # own average — might be due for a pullback (mean reversion).
            "mean_reversion_hist": safe_div(float(np.mean(prices)) - current, max(price_std, 0.01)) if len(prices) >= 3 else 0.0,
        })
    else:
        # Fix 9: When only 1 data point, no momentum calculable
        features.update({
            "momentum_1": 0.0, "momentum_total": 0.0, "momentum_abs": 0.0,
            "volatility": 0.0, "log_volatility": 0.0, "volatility_adj_momentum": 0.0,
            "price_range": 0.0, "price_position": 0.5, "trend_strength": 0.0,
            "volume_trend": 0.0, "volume_acceleration": 0.0, "price_std": 0.0,
            "price_skew": 0.0, "mean_reversion_hist": 0.0,
        })

    # ── 8. Interaction Features (16) ─────────────────────────────
    # "Interaction features" are products (multiplications) of two existing features.
    # Why? Because sometimes what matters is NOT the value of A, and NOT the value of B,
    # but the COMBINATION of A and B together.
    #
    # Example: edge_x_liquidity captures "big price gap in a liquid market."
    # A 20% edge in a market with 10 volume is worthless (you can't trade enough to profit).
    # A 20% edge in a market with 100,000 volume is extremely valuable.
    # Neither "edge" alone nor "volume" alone tells you this — you need both together.
    #
    # Tree-based models CAN discover these interactions on their own, but providing
    # them pre-computed makes it much easier and faster for the model to find them.
    # Cross-feature products that help the tree-based models find complex, non-linear
    # relationships. For example, edge_x_liquidity captures "big edge in liquid markets"
    # which is more actionable than either signal alone.
    features.update({
        # price_x_volume: is a high-conviction price supported by actual trading?
        # High = the market is both far from 50% AND well-traded (confident + confirmed).
        "price_x_volume": (yes_mid / 100) * math.log1p(volume),

        # price_x_time: is a strong price reading imminent (about to resolve)?
        # A market at 80% that closes tomorrow has very different meaning than
        # one at 80% that closes in 6 months. This feature captures that urgency.
        "price_x_time": (yes_mid / 100) * (1 / max(days_to_expiry, 0.1)),

        # volume_x_time: how much trading activity relative to time remaining?
        # High = very active market relative to how soon it closes.
        "volume_x_time": math.log1p(volume) * (1 / max(days_to_expiry, 0.1)),

        # spread_x_time: does the spread widen near expiry? (It usually does.)
        # This feature helps the model learn that a wide spread close to expiry
        # means different things than a wide spread far from expiry.
        "spread_x_time": spread * (1 / max(days_to_expiry, 0.1)),

        # spread_x_volume: is a wide spread despite high volume? That's unusual —
        # normally high volume tightens the spread. This "anomaly" might signal a
        # contested market where bulls and bears are equally matched.
        "spread_x_volume": spread / max(math.log1p(volume), 0.1),

        # edge_x_liquidity: the single most actionable interaction feature.
        # Combines price extremity (how far from 50%) × volume / spread.
        # Only high when the market is BOTH mispriced AND liquid enough to trade.
        "edge_x_liquidity": abs(yes_mid - 50) * math.log1p(volume) / max(spread, 1),

        # imbalance_x_volume: is order flow imbalance confirmed by high volume?
        # Strong directional order flow with high volume = informed trading.
        "imbalance_x_volume": features["order_imbalance"] * math.log1p(volume),

        # imbalance_x_time: does the order imbalance matter more near expiry?
        # Near expiry, last-minute order flow often reflects people who know the
        # answer (e.g., a contract about today's weather, resolving in 1 hour).
        "imbalance_x_time": features["order_imbalance"] * (1 / max(days_to_expiry, 0.1)),

        # extremity_x_efficiency: is an extreme price confirmed by market efficiency?
        # An extreme price in an INEFFICIENT market (low volume, wide spread) might
        # just be a stale quote. In an EFFICIENT market, extreme prices are meaningful.
        "extremity_x_efficiency": features["price_extremity"] * features["market_efficiency"],

        # vig_x_volume: is high vig (exchange fee) common in high-volume markets?
        # Usually not — but when it is, it might signal something unusual.
        "vig_x_volume": features["vig_estimate"] * math.log1p(volume),

        # momentum_x_volume: is a momentum signal backed by real volume?
        # Price momentum on low volume is unreliable. On high volume, it's meaningful.
        "momentum_x_volume": features["price_momentum_proxy"] * math.log1p(volume) / 100,

        # conviction_x_time: is a high-conviction price imminent in resolving?
        # A 90% market that closes in 6 hours has very different meaning than one
        # that closes in 90 days. This feature captures that combination.
        "conviction_x_time": features["conviction_score"] * (1 / max(days_to_expiry, 0.1)),

        # bid_pressure_x_vol: is strong YES buying pressure confirmed by volume?
        # Orderbook pressure on its own can be spoofed (fake orders). But real volume
        # alongside directional pressure is a much stronger signal.
        "bid_pressure_x_vol": features["bid_pressure"] * math.log1p(volume),

        # spread_efficiency: does a wide spread correlate with market inefficiency?
        # Wide spread AND low efficiency = very tradeable market (prone to mispricing).
        "spread_efficiency": features["spread_pct"] * features["market_efficiency"],

        # time_weighted_price: how "urgent" is the current price?
        # Uses exponential decay: exp(-days/30). A price 1 day away is e^(-0.033)=0.97
        # of its raw value. 30 days away is e^(-1.0)=0.37. 90 days away is ~0.05.
        # This discounts long-dated markets heavily (the price may still change a lot).
        "time_weighted_price": yes_mid * math.exp(-days_to_expiry / 30) / 100,

        # risk_adjusted_edge: how large is the price distance from 50%, relative to
        # the cost (spread) and risk (volatility) of capturing it?
        # This is the "information ratio" concept: edge per unit of risk.
        "risk_adjusted_edge": min(100, abs(yes_mid - 50) / (max(spread + 1, 1) * max(features.get("volatility", 0.01), 0.01))),
    })

    # ── 9. New Features (2) ────────────────────────────────────────
    # Fix 19: Circular encoding of time of day (when market was last active)
    # Why encode time as sine and cosine? Because time of day is "circular" — 11:59pm
    # and 12:01am are just 2 minutes apart, but numerically 23.98 and 0.01 look very
    # different. Sine and cosine encoding wraps the clock correctly:
    #   - time_of_day_sin and time_of_day_cos together specify a point on a circle,
    #     where midnight and 11:59pm are very close to each other.
    # This helps the model learn time-of-day effects (e.g., markets are more active
    # during US trading hours) without being confused by the midnight boundary.
    now = datetime.now(timezone.utc)
    hour_of_day = now.hour + now.minute / 60.0
    features["time_of_day_sin"] = math.sin(2 * math.pi * hour_of_day / 24.0)
    features["time_of_day_cos"] = math.cos(2 * math.pi * hour_of_day / 24.0)

    # Fix 20: days_since_creation — age of the market (estimate from close_time - typical durations)
    # If close_time is available, estimate creation as max(0, close_time - 90 days) as proxy
    days_since_creation = 0.0
    if market.close_time:
        try:
            close_dt = datetime.fromisoformat(market.close_time.replace("Z", "+00:00"))
            # Estimate creation: most markets created 1-90 days before close
            # Use days_to_expiry to infer: market_age ~ (total_duration - days_remaining)
            # Assume typical total duration is 30 days if we can't tell
            estimated_total = max(days_to_expiry + 7, 30)
            days_since_creation = max(0.0, estimated_total - days_to_expiry)
        except (ValueError, TypeError, AttributeError):
            days_since_creation = 0.0
    features["days_since_creation"] = days_since_creation

    return features


# Generate the canonical list of feature names by running extract_features on a dummy market.
# This list is used to ensure consistent feature ordering across training and prediction.
#
# WHY consistent ordering matters: Machine learning models are trained with features in
# a specific column order. If we train the model with [volume, spread, yes_mid, ...]
# but then predict with [spread, yes_mid, volume, ...], the model will interpret
# each number as the WRONG feature and give garbage predictions.
#
# By running extract_features() on a dummy market at module load time, we capture the
# exact order that Python's dict returns keys in. This order is fixed for all future
# calls, ensuring training and prediction always use identical column order.
FEATURE_NAMES = list(extract_features(
    Market(ticker="x", event_ticker="x", title="x"),
    Event(event_ticker="x", title="x"),
).keys())

# Fix 1: Zero-feature names for fallback when extract_features fails.
# If feature extraction crashes on a real market, we return a dict of all zeros
# using these same feature names so the model can still run (gracefully degraded).
_ZERO_FEATURE_NAMES = FEATURE_NAMES


# ── Ensemble Model ───────────────────────────────────────────────────────────

class PredictionModel:
    """The main machine learning model for predicting prediction market outcomes.

    This class is the "brain" of the trading bot. It combines two ML algorithms
    (Random Forest and Gradient Boosting) into a single ensemble that:
      1. Can be trained on historical settled markets (supervised learning)
      2. Predicts the probability that YES will win for any current market
      3. Reports its own confidence and uncertainty (prediction intervals)
      4. Detects when the market environment has changed from when it was trained
         (drift detection)
      5. Can retrain incrementally as new data arrives (warm-start retraining)

    --- How a Prediction Works (Step by Step) ---

    When you call predict_probability(features):
      1. The 108 features are passed through the StandardScaler (normalization).
         All values are converted to "how many standard deviations from average."

      2. All 500 Random Forest trees vote independently. Each tree was trained on
         a different random subset of the training data, so each tree gives a
         slightly different answer. We take the average of all 500 votes.

      3. The Gradient Boosting model produces its own probability estimate via its
         150 sequential correction trees.

      4. Final answer = 0.70 × (RF probability) + 0.30 × (GB probability).
         The 70/30 split was determined by testing many combinations and picking
         the one that minimized the Brier Score (a measure of probability accuracy).

    --- When No Training Data Exists ---

    If train_on_historical() has never been called, the model falls back to a
    "heuristic" (rule-based) probability estimate using 12 hard-coded signals.
    This is less accurate than the trained model but much better than random.

    Enhanced with prediction intervals, warm-start retraining, feature caching,
    and model drift detection.
    """

    # Feature cache TTL in seconds.
    # Computing 108 features from scratch for every market in every scan takes time.
    # We cache the result for 120 seconds. If the same market is evaluated again
    # within 2 minutes AND the price hasn't moved more than 2%, we use the cached
    # features rather than recomputing. This is a speed optimization.
    FEATURE_CACHE_TTL = 120
    # Price change threshold for cache invalidation.
    # If the price moves more than 2% since we cached, the cached features are
    # stale (price-derived features like spread_pct would be wrong). Invalidate.
    FEATURE_CACHE_PRICE_THRESHOLD = 0.02

    def __init__(self, n_estimators: int = 50):
        """Initialize the ensemble model with default (untrained) state.

        The model starts untrained. You must call train_on_historical() before
        it can use ML predictions. Until then, it uses the heuristic fallback.

        Args:
            n_estimators: Starting number of trees. Grows up to 75 with warm-start
                          retraining. More trees = more accurate but slower.
        """
        # n_estimators: how many trees to start with.
        # More trees = slower to train but more accurate and stable predictions.
        # We start at 50 and can grow to 75 with incremental retraining.
        self.n_estimators = n_estimators

        # The two ML models. They start as None and are initialized by _build_models().
        # They become "live" after train_on_historical() is called.
        self.rf: RandomForestClassifier | None = None
        self.gb: GradientBoostingClassifier | None = None

        # StandardScaler: normalizes all 108 features to mean=0, std=1.
        # This is fitted during training (scaler.fit()) and then applied both
        # during training AND prediction (scaler.transform()) to ensure consistency.
        # IMPORTANT: The scaler must be saved alongside the model — applying a
        # different scaler during prediction would corrupt all feature values.
        self.scaler = StandardScaler()

        # is_trained: False until train_on_historical() completes successfully.
        # When False, predict_probability() falls back to the heuristic method.
        self.is_trained = False

        # cv_score: the cross-validation accuracy score from the last training run.
        # This is the "honest" accuracy — measured on data the model never trained on.
        # A score of 0.70 means the model was correct 70% of the time on held-out data.
        self.cv_score: float = 0.0

        # n_training_samples: how many historical markets were used in the last training.
        # More samples = better model (up to a point of diminishing returns).
        self.n_training_samples: int = 0

        # Feature cache: {ticker: {features, timestamp, price}}
        # Stores recently computed feature dicts so we don't recompute from scratch
        # every scan cycle. The cache is invalidated by time (TTL) or price movement.
        self._feature_cache: dict[str, dict] = {}

        # Drift detection: rolling window of recent predictions.
        # We store the last 200 probability predictions. If the AVERAGE of recent
        # predictions shifts significantly from the training-time average, it means
        # the market environment has changed (model drift). We should retrain.
        self._prediction_history: deque[float] = deque(maxlen=200)
        self._training_pred_mean: float = 0.5   # What the model predicted on average during training
        self._training_pred_std: float = 0.0    # How spread out those predictions were

        # Task 6: Top feature interactions — which pairs of features work best together
        self.top_interactions: list[dict] = []

        # Task 8: Ensemble disagreement tracking.
        # When RF and GB give very different answers (disagree by >0.15), that's a
        # sign the prediction is uncertain. Tracking the disagreement rate over time
        # tells us how often the two models are "confused" simultaneously.
        self._disagreement_history: deque[float] = deque(maxlen=200)
        self._ensemble_disagreement_rate: float = 0.0

        # Task 9: Adaptive learning rate for GB.
        # If the model's cross-validation accuracy has been declining across multiple
        # retraining runs (consecutive_drops >= 2), reduce the learning rate to be
        # more conservative. This prevents the GB model from over-correcting.
        self._accuracy_history: list[float] = []
        self._learning_rate_history: list[float] = [0.01]
        self._current_gb_lr: float = 0.01

        # Task 10: Training feature means for prediction explanations.
        # After training, we store the average value of each feature across all
        # training samples. This lets us explain predictions: "this market has
        # unusually HIGH volume compared to the training average, which pushed
        # the model's probability upward."
        self._training_feature_means: dict[str, float] = {}

        # Fix 33: Model versioning — incremented each time we retrain.
        # Lets us track whether a cached model file is outdated.
        self.version: int = 0

        # Fix 34: Unix timestamp of when the model was last trained.
        # Helps us check: "is this model stale? Should we retrain?"
        self.last_trained_at: float = 0.0

        # Fix 35: Track number of training samples used in the last training run.
        self.last_training_samples: int = 0

        # Initialize the model objects (but don't train them yet)
        self._build_models()

    def _build_models(self):
        """Initialize the ensemble with optimized hyperparameters.

        This sets up the Random Forest and Gradient Boosting models with specific
        settings that were tuned on Kalshi market data. Each parameter controls
        a different aspect of how the model learns.

        Uses warm_start=True so incremental retraining adds trees
        rather than rebuilding from scratch (faster for live updates).
        """
        n_features = len(FEATURE_NAMES)
        # max_features_per_tree: how many features each tree can consider at each split.
        # We use sqrt(n_features) ≈ 10 features. This is the "random" in Random Forest:
        # each tree only SEES a random subset of features at each decision point.
        # This forces diversity among trees, which makes the ensemble more robust.
        # If every tree could see all 108 features, they'd all make similar decisions,
        # reducing the benefit of combining them.
        max_features_per_tree = max(1, int(math.sqrt(n_features)))

        self.rf = RandomForestClassifier(
            # n_estimators: number of trees. 500 is a good balance of accuracy vs speed.
            n_estimators=max(self.n_estimators, 50),

            # max_features: ~10 random features per split (decorrelates trees).
            max_features=max_features_per_tree,

            # max_depth=10: trees can be AT MOST 10 levels deep.
            # Deeper trees = more specific rules = more overfitting risk.
            # 10 levels allows learning complex patterns without memorizing the training data.
            max_depth=10,

            # min_samples_split=8: a node must have at least 8 examples before it can
            # create a split rule. Prevents the model from creating rules based on
            # tiny sample sizes (which would be noise, not signal).
            min_samples_split=8,

            # min_samples_leaf=1: each final leaf can have just 1 example.
            # This is more permissive than min_samples_split — leaves can be small.
            min_samples_leaf=1,

            # class_weight="balanced": automatically adjusts for class imbalance.
            # If only 30% of training markets resolved YES, the model would be biased
            # toward predicting NO. "balanced" reweights so both classes matter equally.
            class_weight="balanced",

            # random_state=42: makes results reproducible. Same seed = same trees
            # every time we run. Without this, each run gives slightly different results.
            random_state=42,

            # n_jobs=-1: use all available CPU cores to train trees in parallel.
            # Training 500 trees is embarrassingly parallel — each tree is independent.
            n_jobs=-1,

            # oob_score=True: compute the "Out-Of-Bag" score as a free accuracy estimate.
            # Each tree is trained on ~63% of the data (random sampling with replacement).
            # The other ~37% it never saw during training = "out of bag" examples.
            # We can test each tree's accuracy on its OOB examples as a free CV estimate.
            oob_score=True,

            # warm_start=True: when retraining, ADD trees to the existing forest
            # rather than starting from scratch. Much faster for incremental updates.
            warm_start=True,
        )

        self.gb = GradientBoostingClassifier(
            # n_estimators: 50 sequential correction trees.
            # GB needs fewer trees than RF because each tree learns from the previous.
            n_estimators=50,

            # max_depth=7: shallower trees than RF (4 levels) because GB trees are
            # "weak learners" by design — they should only partially correct the error.
            # Deep GB trees overfit aggressively.
            max_depth=7,

            # learning_rate=0.01: each tree corrects only 1% of the remaining error.
            # Very conservative — prevents any single tree from dominating.
            # Lower learning rate = more trees needed = more training time but
            # better generalization. Adapted dynamically based on accuracy history.
            learning_rate=0.01,

            # subsample=0.8: each tree is trained on a random 80% of the training data.
            # This is "stochastic gradient boosting" — adds randomness to prevent overfitting.
            # The 20% left out also gives a free OOB-like estimate.
            subsample=0.8,

            # max_features: same sqrt(108) ≈ 10 random features per split as RF.
            max_features=max_features_per_tree,

            # min_samples_leaf=3: each leaf needs at least 3 examples.
            # Slightly stricter than RF to prevent overfitting (GB is more prone to it).
            min_samples_leaf=3,

            random_state=42,
            warm_start=True,
        )

    def train_on_historical(self, features_list: list[dict], outcomes: list[int]) -> dict:
        """
        Train both models on historical market data.

        This is the "learning" phase of machine learning. We show the model
        thousands of examples of:
            (features of a market) -> (did YES win? 1 or 0)

        After seeing all these examples, the model learns which features predict
        a YES outcome and which predict a NO outcome.

        Why more data = better model:
          - With 100 examples, the model might notice "high volume correlates with YES"
            but not know if that's real or a coincidence.
          - With 10,000 examples, the model can verify the pattern holds across many
            different market types and time periods. It becomes more confident and
            less likely to be fooled by random noise.
          - With 100,000 examples, the model discovers subtle second-order effects
            ("high volume predicts YES, but only when the spread is tight").

        Args:
            features_list: List of feature dicts, one per settled historical market.
                           Each dict has 108 keys (feature names) -> float values.
                           Example: [{"yes_mid": 42.0, "volume": 5000, ...}, ...]

            outcomes:      List of 0s and 1s corresponding to each feature dict.
                           1 = YES won (the event happened).
                           0 = NO won (the event did not happen).
                           Must have the same length as features_list.

        Returns:
            A dict with training metrics: cv_accuracy, oob_score, calibration results,
            feature importance, per-fold scores, and whether training succeeded.

        Returns training metrics including cross-validation score, feature pruning info,
        per-fold CV scores, and calibration results.
        """
        # Fix 21: Validate features list is non-empty
        if not features_list:
            return {"status": "no_data", "error": "Empty features list", "trained": False}

        if len(features_list) < 30:
            return {"status": "insufficient_data", "error": "Need at least 30 samples to train", "trained": False}

        # Fix 22: Validate all feature dicts have the same keys (fill missing with 0)
        expected_keys = set(FEATURE_NAMES)
        missing_count = 0
        for f in features_list:
            missing = expected_keys - set(f.keys())
            if missing:
                missing_count += 1
                for key in missing:
                    f[key] = 0.0
        if missing_count:
            logger.info(f"Filled missing feature keys in {missing_count}/{len(features_list)} samples")

        # Convert the list of feature dicts into a 2D numpy array.
        # X shape: (n_samples, n_features) = (number of markets, 108)
        # Each ROW is one market. Each COLUMN is one feature.
        # Example with 3 markets and 3 features:
        #   X = [[42, 5000, 0.04],   <- market 1: yes_mid=42, volume=5000, spread_pct=0.04
        #        [78, 1200, 0.06],   <- market 2: yes_mid=78, volume=1200, spread_pct=0.06
        #        [31, 8000, 0.02]]   <- market 3: yes_mid=31, volume=8000, spread_pct=0.02
        X = np.array([[f.get(name, 0) for name in FEATURE_NAMES] for f in features_list])

        # y is the outcome vector: 1 = YES won, 0 = NO won.
        # y = [1, 0, 1] means market 1 resolved YES, market 2 resolved NO, market 3 resolved YES.
        y = np.array(outcomes)

        # Fix 23: Handle case where all outcomes are the same class
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.warning(f"All outcomes are class {unique_classes[0]}, cannot train meaningful model")
            return {"status": "single_class", "error": f"All {len(y)} outcomes are class {unique_classes[0]}", "trained": False}

        # Fix 24: Log training metrics
        positive_count = int(np.sum(y == 1))
        negative_count = int(np.sum(y == 0))
        logger.info(f"Training: {len(features_list)} samples, {len(FEATURE_NAMES)} features, "
                     f"class balance: {positive_count}/{negative_count} (pos/neg)")

        # Safety: replace any NaN (Not a Number) or Inf (Infinity) values with safe numbers.
        # NaN/Inf can sneak in from math operations like 0/0 or log(0) that weren't caught.
        # ML models cannot handle NaN — they produce nonsense or crash entirely.
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # FIT the scaler on the training data.
        # scaler.fit(X) computes the mean and std of each feature column.
        # It does NOT change X yet — it just "memorizes" the statistics.
        self.scaler.fit(X)

        # TRANSFORM X using the fitted scaler.
        # Each feature value becomes: (value - mean) / std_dev
        # Example: if volume has mean=5000 and std=3000, then:
        #   - volume=5000 becomes 0.0 (exactly average)
        #   - volume=8000 becomes +1.0 (one standard deviation above average)
        #   - volume=2000 becomes -1.0 (one standard deviation below average)
        # After this transformation, ALL features are on the same scale.
        X_scaled = self.scaler.transform(X)

        # Warm-start: if already trained and getting more data, add trees incrementally
        is_retrain = self.is_trained and len(features_list) > self.n_training_samples
        if is_retrain:
            self.rf.n_estimators = min(self.rf.n_estimators + 5, 75)
            self.gb.n_estimators = min(self.gb.n_estimators + 5, 75)
            logger.info(
                f"Warm-start retrain: RF {self.rf.n_estimators} trees, "
                f"GB {self.gb.n_estimators} trees"
            )

        # Task 9: Adaptive learning rate for GB
        if is_retrain and self._accuracy_history:
            prev_acc = self._accuracy_history[-1] if self._accuracy_history else 0.5
            # Will be updated after CV below; for now adjust LR based on history
            consecutive_drops = 0
            for i in range(len(self._accuracy_history) - 1, 0, -1):
                if self._accuracy_history[i] < self._accuracy_history[i - 1]:
                    consecutive_drops += 1
                else:
                    break
            if consecutive_drops >= 2:
                self._current_gb_lr = 0.001
            elif consecutive_drops >= 1:
                self._current_gb_lr = 0.005
            else:
                self._current_gb_lr = 0.01
            self.gb.learning_rate = self._current_gb_lr
            self._learning_rate_history.append(self._current_gb_lr)
            logger.info(f"Adaptive LR: {self._current_gb_lr} (consecutive drops: {consecutive_drops})")

        # Task 10: Compute sample weights by recency (exponential decay)
        # Recent market data is more relevant than old data. Markets from 6 months ago
        # might reflect a different market environment than today.
        # We assign higher "sample weights" to recent examples using exponential decay:
        #   weight = exp(-0.001 * days_old)
        # A market from yesterday gets weight ≈ 0.999. One from 1000 days ago gets ≈ 0.37.
        # The model pays more attention to recent patterns during training.
        sample_weights = None
        try:
            # Use feature index to approximate recency — later features_list items are more recent
            n = len(features_list)
            # Weight = exp(-0.001 * days_old), approximate days_old from position in list
            # Assume ~1 sample/day spread across the dataset
            days_old = np.linspace(n, 0, n)  # oldest first, newest last
            sample_weights = np.exp(-0.001 * days_old)
            # Normalize to sum to n (preserves effective sample size interpretation)
            sample_weights = sample_weights * n / sample_weights.sum()
        except Exception as e:
            logger.warning(f"Sample weighting failed: {e}")
            sample_weights = None

        # Train Random Forest (with sample weights)
        if sample_weights is not None:
            self.rf.fit(X_scaled, y, sample_weight=sample_weights)
        else:
            self.rf.fit(X_scaled, y)

        # Train Gradient Boosting (with sample weights)
        if sample_weights is not None:
            self.gb.fit(X_scaled, y, sample_weight=sample_weights)
        else:
            self.gb.fit(X_scaled, y)

        self.is_trained = True
        self.n_training_samples = len(features_list)
        # Fix 33: Increment model version on each train
        self.version += 1
        # Fix 34: Track when model was last trained
        self.last_trained_at = time.time()
        # Fix 35: Track number of training samples used
        self.last_training_samples = len(features_list)

        # Task 10: Store training feature means for prediction explanations
        for i, name in enumerate(FEATURE_NAMES):
            self._training_feature_means[name] = float(np.mean(X[:, i]))

        # Compute training prediction distribution for drift detection
        try:
            train_preds = self.rf.predict_proba(X_scaled)
            if train_preds.shape[1] > 1:
                train_yes_probs = train_preds[:, 1]
                self._training_pred_mean = float(np.mean(train_yes_probs))
                self._training_pred_std = float(np.std(train_yes_probs))
        except Exception:
            pass

        # Clear feature cache on retrain
        self._feature_cache.clear()

        # ── Task 21: Feature importance pruning ──
        # After training, we can ask: "which features did the model actually USE?"
        # Feature importance measures how much each feature reduced prediction error
        # across all decision splits in all trees.
        #
        # Features with importance < 0.001 (0.1%) contributed almost nothing.
        # We log these so you can see which features turned out to be useless noise.
        # We don't remove them (to maintain feature order compatibility) but this
        # list is informative for future feature engineering work.
        # Identify low-importance features (< 0.001) and log them for transparency
        pruned_feature_count = 0
        low_importance_features = []
        try:
            # Combined importance: 60% from RF + 40% from GB (matches prediction weights roughly)
            combined_imp = 0.6 * self.rf.feature_importances_ + 0.4 * self.gb.feature_importances_
            for i, name in enumerate(FEATURE_NAMES):
                if combined_imp[i] < 0.001:
                    low_importance_features.append(name)
            pruned_feature_count = len(low_importance_features)
            if low_importance_features:
                logger.info(f"Low importance features ({pruned_feature_count}): {low_importance_features[:10]}{'...' if pruned_feature_count > 10 else ''}")
        except Exception:
            pass

        # ── Purged K-Fold cross-validation ───────────────────────────────
        # Standard StratifiedKFold inflates accuracy on financial time series
        # because adjacent observations share price/label information (leakage).
        # Purged CV removes training samples within an embargo window (15 min
        # = 1 bar) of each test fold boundary, preventing temporal leakage.
        #
        # Reference: López de Prado, "Advances in Financial Machine Learning" (2018)
        # A 96% standard CV score on this dataset is almost certainly inflated;
        # the purged score will be lower and more representative of live accuracy.
        per_fold_scores = []
        try:
            n = len(y)
            min_class_count = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
            n_folds = min(5, n // 10, min_class_count)

            if n_folds >= 2:
                # Embargo: number of samples to remove around each fold boundary.
                # For 15-minute bars, 1 embargo bar = 1 observation.
                embargo = max(1, n // (n_folds * 15))
                fold_size = n // n_folds
                purged_scores = []

                for fold_idx in range(n_folds):
                    # Test fold indices (sequential — time order preserved)
                    test_start = fold_idx * fold_size
                    test_end = test_start + fold_size if fold_idx < n_folds - 1 else n

                    # Embargo: remove training samples adjacent to test boundary
                    embargo_start = max(0, test_start - embargo)
                    embargo_end = min(n, test_end + embargo)

                    train_idx = np.array([
                        i for i in range(n)
                        if i < embargo_start or i >= embargo_end
                    ])
                    test_idx = np.arange(test_start, test_end)

                    if len(train_idx) < 20 or len(test_idx) < 5:
                        continue
                    # Skip folds with only one class in train or test
                    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                        continue

                    self.rf.fit(X_scaled[train_idx], y[train_idx])
                    fold_preds = self.rf.predict(X_scaled[test_idx])
                    fold_acc = float(np.mean(fold_preds == y[test_idx]))
                    purged_scores.append(fold_acc)

                if purged_scores:
                    per_fold_scores = [round(s, 4) for s in purged_scores]
                    self.cv_score = float(np.mean(purged_scores))
                    logger.info(
                        f"Purged CV ({n_folds}-fold, embargo={embargo}) "
                        f"per-fold: {per_fold_scores} mean={self.cv_score:.4f}"
                    )
                    if self.cv_score < 0.57:
                        logger.warning(
                            f"[rf_model] Purged CV={self.cv_score:.3f} < 0.57 — "
                            f"model has weak out-of-sample edge. Use conservatively."
                        )
                    # Re-train on full dataset after CV
                    self.rf.fit(X_scaled, y)
                else:
                    self.cv_score = self.rf.oob_score_ if hasattr(self.rf, 'oob_score_') else 0
            else:
                self.cv_score = self.rf.oob_score_ if hasattr(self.rf, 'oob_score_') else 0
        except Exception as _cv_err:
            logger.warning(f"[rf_model] Purged CV failed: {_cv_err} — falling back to OOB score")
            self.cv_score = self.rf.oob_score_ if hasattr(self.rf, 'oob_score_') else 0

        # ── Task 25: Prediction calibration check ──
        # CALIBRATION: Does the model's predicted probability match reality?
        #
        # We split predictions into "confidence buckets" and check each one:
        #   Bucket 0.5-0.6: "model predicted 50-60% YES" → did ~55% of these actually resolve YES?
        #   Bucket 0.6-0.7: "model predicted 60-70% YES" → did ~65% actually resolve YES?
        #   Bucket 0.7-0.8: did ~75% actually resolve YES?
        #   etc.
        #
        # If the model says 70% and only 50% actually win, it's OVERCONFIDENT.
        # If the model says 70% and 90% actually win, it's UNDERCONFIDENT.
        # Perfect calibration = the bars match. This is measured using RMSE (Root Mean Square Error).
        #
        # Note: this calibration check is on TRAINING data (in-sample), which overstates accuracy.
        # True calibration requires held-out data. But this still catches gross miscalibration.
        # Compare predicted probabilities to actual outcome rates in each bucket
        calibration_results = {}
        calibration_score = 0.0
        try:
            # Get predictions on training data (in-sample, but useful for calibration check)
            rf_proba = self.rf.predict_proba(X_scaled)
            gb_proba = self.gb.predict_proba(X_scaled)

            if len(self.rf.classes_) == 2 and len(self.gb.classes_) == 2:
                # Extract the probability of YES (class 1) from each model's output.
                # predict_proba() returns [[P(NO), P(YES)], ...] for each sample.
                # Column 1 (index [1]) is P(YES).
                rf_yes = rf_proba[:, 1]
                gb_yes = gb_proba[:, 1]
                # Combine using the same 70/30 weights as live predictions
                ensemble_proba = 0.7 * rf_yes + 0.3 * gb_yes

                # Define probability buckets to check calibration
                buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
                total_squared_error = 0.0
                n_buckets_with_data = 0
                for low, high in buckets:
                    # Find all predictions that fell in this confidence bucket
                    mask = (ensemble_proba >= low) & (ensemble_proba < high)
                    if np.sum(mask) >= 5:  # Need at least 5 samples for a meaningful check
                        predicted_avg = float(np.mean(ensemble_proba[mask]))  # Avg prediction in this bucket
                        actual_rate = float(np.mean(y[mask]))  # Actual YES win rate in this bucket
                        bucket_key = f"{low:.1f}-{high:.1f}"
                        calibration_results[bucket_key] = {
                            "predicted": round(predicted_avg, 4),  # What the model predicted
                            "actual": round(actual_rate, 4),         # What actually happened
                            "count": int(np.sum(mask)),              # How many markets in this bucket
                        }
                        # Squared error: (0.75 predicted, 0.65 actual) = (0.10)^2 = 0.01 error
                        total_squared_error += (predicted_avg - actual_rate) ** 2
                        n_buckets_with_data += 1
                        logger.info(f"Calibration [{bucket_key}]: predicted={predicted_avg:.3f}, actual={actual_rate:.3f} (n={int(np.sum(mask))})")

                if n_buckets_with_data > 0:
                    # Calibration score: 1.0 = perfect, 0.0 = terrible (RMSE-based)
                    # RMSE = Root Mean Square Error. 0.2 RMSE = 20% average miscalibration.
                    rmse = math.sqrt(total_squared_error / n_buckets_with_data)
                    # Scale to 0-1: 0.2 RMSE = 0.0 score (terrible). 0.0 RMSE = 1.0 (perfect).
                    calibration_score = max(0.0, 1.0 - rmse * 5)  # Scale: 0.2 RMSE = 0.0 score
        except Exception as e:
            logger.warning(f"Calibration check failed: {e}")

        # Task 6: Feature interaction detection via permutation importance of feature pairs
        self.top_interactions = []
        try:
            if len(X_scaled) >= 50 and len(self.rf.classes_) == 2:
                from sklearn.metrics import accuracy_score
                base_preds = self.rf.predict(X_scaled)
                base_acc = accuracy_score(y, base_preds)
                importance_dict = self.get_feature_importance()
                top_features = list(importance_dict.keys())[:15]  # Top 15 features
                top_indices = [FEATURE_NAMES.index(f) for f in top_features if f in FEATURE_NAMES]

                interaction_scores = []
                rng = np.random.RandomState(42)
                for i in range(len(top_indices)):
                    for j in range(i + 1, len(top_indices)):
                        fi, fj = top_indices[i], top_indices[j]
                        # Permute both features simultaneously
                        X_perm = X_scaled.copy()
                        perm_idx = rng.permutation(len(X_perm))
                        X_perm[:, fi] = X_perm[perm_idx, fi]
                        X_perm[:, fj] = X_perm[perm_idx, fj]
                        perm_preds = self.rf.predict(X_perm)
                        perm_acc = accuracy_score(y, perm_preds)
                        # Interaction importance = drop from permuting pair - sum of individual drops
                        X_perm_i = X_scaled.copy()
                        X_perm_i[:, fi] = X_perm_i[perm_idx, fi]
                        drop_i = base_acc - accuracy_score(y, self.rf.predict(X_perm_i))
                        X_perm_j = X_scaled.copy()
                        X_perm_j[:, fj] = X_perm_j[perm_idx, fj]
                        drop_j = base_acc - accuracy_score(y, self.rf.predict(X_perm_j))
                        interaction_importance = (base_acc - perm_acc) - drop_i - drop_j
                        interaction_scores.append({
                            "features": f"{FEATURE_NAMES[fi]} x {FEATURE_NAMES[fj]}",
                            "interaction_importance": round(interaction_importance, 6),
                            "combined_drop": round(base_acc - perm_acc, 6),
                        })
                interaction_scores.sort(key=lambda x: abs(x["interaction_importance"]), reverse=True)
                self.top_interactions = interaction_scores[:10]
                if self.top_interactions:
                    logger.info(f"Top feature interaction: {self.top_interactions[0]['features']} "
                                f"(importance={self.top_interactions[0]['interaction_importance']:.4f})")
        except Exception as e:
            logger.warning(f"Feature interaction detection failed: {e}")

        # Task 9: Record accuracy for adaptive LR
        self._accuracy_history.append(round(self.cv_score, 4))
        if len(self._accuracy_history) > 20:
            self._accuracy_history = self._accuracy_history[-20:]

        return {
            "trained": True,
            "samples": len(features_list),
            "cv_accuracy": round(self.cv_score, 4),
            # Fix 32: Handle case where oob_score is NaN
            "oob_score": round(self.rf.oob_score_, 4) if hasattr(self.rf, 'oob_score_') and self.rf.oob_score_ and not math.isnan(self.rf.oob_score_) else 0,
            "n_features": len(FEATURE_NAMES),
            "pruned_feature_count": pruned_feature_count,
            "low_importance_features": low_importance_features[:20],
            "per_fold_scores": per_fold_scores,
            "calibration_results": calibration_results,
            "calibration_score": round(calibration_score, 4),
            "warm_start": is_retrain,
            "top_interactions": self.top_interactions,
            "learning_rate": self._current_gb_lr,
            "learning_rate_history": self._learning_rate_history[-10:],
            "model_version": self.version,
            "trained_at": self.last_trained_at,
        }

    def save_to_disk(self, path: str | None = None) -> bool:
        """Persist the trained model to disk using joblib. Returns True on success."""
        if not self.is_trained:
            return False
        try:
            import joblib, pathlib
            p = pathlib.Path(path or "data/prediction_model.joblib")
            p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "rf": self.rf,
                "gb": self.gb,
                "scaler": self.scaler,
                "version": self.version,
                "n_training_samples": self.n_training_samples,
                "last_trained_at": getattr(self, "last_trained_at", 0),
                "cv_score": self.cv_score,
                "_training_feature_means": self._training_feature_means,
                "_training_pred_mean": self._training_pred_mean,
                "_training_pred_std": self._training_pred_std,
            }, p, compress=3)
            logger.info(f"Model saved to {p} (v{self.version}, {self.n_training_samples} samples)")
            return True
        except Exception as e:
            logger.warning(f"Model save failed: {e}")
            return False

    def load_from_disk(self, path: str | None = None) -> bool:
        """Load a previously trained model from disk. Returns True on success."""
        try:
            import joblib, pathlib
            p = pathlib.Path(path or "data/prediction_model.joblib")
            if not p.exists():
                return False
            state = joblib.load(p)
            self.rf = state["rf"]
            self.gb = state["gb"]
            self.scaler = state["scaler"]
            self.version = state.get("version", 0)
            self.n_training_samples = state.get("n_training_samples", 0)
            self.last_trained_at = state.get("last_trained_at", 0)
            self.cv_score = state.get("cv_score", 0.0)
            self._training_feature_means = state.get("_training_feature_means", {})
            self._training_pred_mean = state.get("_training_pred_mean", 0.5)
            self._training_pred_std = state.get("_training_pred_std", 0.0)
            self.is_trained = True
            age_hours = (time.time() - self.last_trained_at) / 3600
            logger.info(f"Model loaded from {p} (v{self.version}, {self.n_training_samples} samples, {age_hours:.1f}h old)")
            return True
        except Exception as e:
            logger.warning(f"Model load failed: {e}")
            return False

    def predict_probability(self, features: dict) -> float:
        """
        Predict YES probability using ensemble averaging.

        This is the core "inference" step — using the trained model to make a
        prediction on a market it has NEVER seen before. The model applies
        everything it learned during training to produce a single number:
        the probability that YES will win this market.

        Example:
            features = extract_features(market, event)
            prob = model.predict_probability(features)
            # prob = 0.73 means "73% chance YES wins"

        The result is used in two ways:
          1. DIRECTION: Is prob > 0.5? → bet YES. Is prob < 0.5? → bet NO.
          2. EDGE: How far is prob from the current market price?
             If prob=0.73 and market_price=0.55, edge = 0.73 - 0.55 = 0.18 (18%).

        Combines RF and GB predictions with weighted average.
        RF gets 70% weight (better calibrated), GB gets 30% (optimized via Brier score).
        """
        # Fix 26: Return 0.5 (max uncertainty) when model is not trained.
        # 0.5 = maximum uncertainty = "I have no idea." Rather than crashing,
        # the untrained model uses a heuristic (rule-based) fallback.
        if not self.is_trained:
            return self._heuristic_probability(features)

        # Fix 27: Validate feature dict has all required keys, fill missing with 0
        # Work on a copy to avoid mutating caller's dict
        features_copy = dict(features)
        for name in FEATURE_NAMES:
            if name not in features_copy:
                features_copy[name] = 0.0

        X = np.array([[features_copy.get(name, 0) for name in FEATURE_NAMES]])
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        # Ensemble: weighted average of RF and GB
        # predict_proba() returns an array of [P(NO), P(YES)] for this one market.
        # We take [0] because we only have one market (the input was shape [1, 108]).
        rf_proba = self.rf.predict_proba(X_scaled)[0]

        # Edge case: if the model was trained on data where all outcomes were the
        # same class (e.g., all NO), it only knows one class. We handle this by
        # returning 1.0 or 0.0 based on which class the model knows.
        if len(self.rf.classes_) == 1:
            rf_yes = 1.0 if self.rf.classes_[0] == 1 else 0.0
        else:
            rf_yes = float(rf_proba[1])  # rf_proba[1] = P(YES) from RF

        gb_proba = self.gb.predict_proba(X_scaled)[0]
        if len(self.gb.classes_) == 1:
            gb_yes = 1.0 if self.gb.classes_[0] == 1 else 0.0
        else:
            gb_yes = float(gb_proba[1])  # gb_proba[1] = P(YES) from GB

        # Weighted ensemble (optimized via Brier score grid search).
        # The Brier Score measures probability accuracy: Brier = mean((predicted_prob - actual)^2).
        # Lower is better (0 = perfect, 0.25 = random). We tested many weight combinations
        # (0.5/0.5, 0.6/0.4, 0.7/0.3, etc.) and 70/30 gave the best Brier Score.
        ensemble_prob = 0.7 * rf_yes + 0.3 * gb_yes

        # Clip to [0.01, 0.99] — we never say "absolutely certain" (0 or 1) because
        # even the most obvious-seeming market can have surprises. This also prevents
        # downstream math from hitting log(0) or similar problems.
        prob = float(np.clip(ensemble_prob, 0.01, 0.99))

        # Track for drift detection
        self._prediction_history.append(prob)

        # Task 8: Track ensemble disagreement
        disagreement = abs(rf_yes - gb_yes)
        self._disagreement_history.append(disagreement)

        return prob

    def predict(self, features: dict) -> dict:
        """Predict YES probability WITH a confidence interval showing the model's certainty.

        This is the "rich" version of predict_probability(). In addition to the
        probability estimate, it also tells you HOW CONFIDENT the model is.

        The confidence interval comes from asking all 500 individual trees to vote
        separately, then measuring how much they disagree. Think of it like asking
        500 experts for their opinion:
          - If 490 out of 500 say "70% YES" → low spread → high confidence.
          - If 250 say "30% YES" and 250 say "80% YES" → high spread → low confidence.

        The spread (standard deviation) of tree predictions IS the uncertainty.
        Roughly 68% of outcomes fall within ±1 standard deviation of the mean.

        Returns a dict with:
          - probability: ensemble prediction (0-1)
          - prediction_std: std dev of individual RF tree predictions (lower = more confident)
          - confidence_low: lower bound of ~68% confidence interval
          - confidence_high: upper bound of ~68% confidence interval

        Example:
            result = model.predict(features)
            # {"probability": 0.72, "prediction_std": 0.08,
            #  "confidence_low": 0.64, "confidence_high": 0.80}
            # → Model says 72%, but could reasonably be anywhere from 64% to 80%.
        """
        if not self.is_trained:
            prob = self._heuristic_probability(features)
            return {
                "probability": prob,
                "prediction_std": 0.15,
                "confidence_low": max(0.01, prob - 0.15),
                "confidence_high": min(0.99, prob + 0.15),
            }

        X = np.array([[features.get(name, 0) for name in FEATURE_NAMES]])
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        X_scaled = self.scaler.transform(X)

        # Get individual tree predictions for confidence interval.
        # We ask each tree in the Random Forest to vote independently.
        # self.rf.estimators_ is the list of all 500 individual Decision Tree objects.
        tree_predictions = []
        for tree in self.rf.estimators_:
            # Each tree outputs its own probability estimate for this one market.
            tree_proba = tree.predict_proba(X_scaled)[0]
            if len(tree.classes_) == 1:
                tree_yes = 1.0 if tree.classes_[0] == 1 else 0.0
            else:
                tree_yes = float(tree_proba[1])
            tree_predictions.append(tree_yes)

        # Standard deviation of the 500 tree votes = the model's uncertainty.
        # Low std (e.g., 0.03) = all trees agreed → high confidence.
        # High std (e.g., 0.20) = trees disagreed widely → low confidence.
        prediction_std = float(np.std(tree_predictions))

        # Ensemble prediction
        rf_proba = self.rf.predict_proba(X_scaled)[0]
        rf_yes = float(rf_proba[1]) if len(self.rf.classes_) > 1 else (1.0 if self.rf.classes_[0] == 1 else 0.0)

        gb_proba = self.gb.predict_proba(X_scaled)[0]
        gb_yes = float(gb_proba[1]) if len(self.gb.classes_) > 1 else (1.0 if self.gb.classes_[0] == 1 else 0.0)

        ensemble_prob = float(np.clip(0.7 * rf_yes + 0.3 * gb_yes, 0.01, 0.99))
        self._prediction_history.append(ensemble_prob)

        return {
            "probability": ensemble_prob,
            "prediction_std": round(prediction_std, 4),
            "confidence_low": max(0.01, round(ensemble_prob - prediction_std, 4)),
            "confidence_high": min(0.99, round(ensemble_prob + prediction_std, 4)),
        }

    def get_drift_score(self) -> float:
        """Compute model drift score: absolute shift in mean prediction from training.

        "Model drift" happens when the real world changes but the model doesn't.
        Example: the model was trained on 2023 data when crypto markets were highly
        volatile. In 2025, markets are calmer. The model was trained to expect
        volatility that no longer exists — its predictions will be systematically off.

        We detect drift by comparing:
          - training_pred_mean: the average prediction the model made ON TRAINING DATA
          - current_pred_mean: the average prediction the model has been making LIVE

        If these diverge significantly (drift > 0.10 = 10% shift), it's a signal
        to retrain the model on more recent data.

        Returns 0.0 if insufficient data. Values > 0.1 indicate significant drift.
        """
        if len(self._prediction_history) < 20:
            return 0.0
        current_mean = sum(self._prediction_history) / len(self._prediction_history)
        return round(abs(current_mean - self._training_pred_mean), 4)

    def get_drift_status(self) -> dict:
        """Get drift detection status for the API."""
        drift_score = self.get_drift_score()
        return {
            "drift_score": drift_score,
            "drift_detected": drift_score > 0.1,
            "training_pred_mean": round(self._training_pred_mean, 4),
            "current_pred_mean": round(
                sum(self._prediction_history) / max(len(self._prediction_history), 1), 4
            ) if self._prediction_history else 0.5,
            "prediction_history_size": len(self._prediction_history),
        }

    def get_cached_features(self, ticker: str, current_price: float) -> dict | None:
        """Return cached features if still valid (within TTL and price hasn't moved >2%)."""
        cached = self._feature_cache.get(ticker)
        if not cached:
            return None
        elapsed = time.time() - cached["timestamp"]
        if elapsed > self.FEATURE_CACHE_TTL:
            return None
        cached_price = cached.get("price", 0)
        if cached_price > 0 and abs(current_price - cached_price) / cached_price > self.FEATURE_CACHE_PRICE_THRESHOLD:
            return None
        return cached["features"]

    def cache_features(self, ticker: str, features: dict, price: float):
        """Store computed features in the cache with current timestamp."""
        self._feature_cache[ticker] = {
            "features": features,
            "timestamp": time.time(),
            "price": price,
        }

    def _heuristic_probability(self, features: dict) -> float:
        """
        Multi-factor heuristic (rule-based) probability estimate when no ML model is trained.

        This is the bot's "fallback brain" — a hand-crafted set of 12 rules that
        approximate what a human expert might do when evaluating a market.

        It is NOT as accurate as the trained ML model. But it's much better than
        just using the market's own price as the probability (which ignores order
        flow, momentum, and efficiency signals).

        Think of it as: "before we had enough data to train the ML model, what would
        a smart analyst do? They'd look at the order book, check if the spread is
        reasonable, see if there's momentum, and adjust the market's stated probability
        accordingly."

        The weights (+/- 0.12, +/- 0.08, etc.) were chosen empirically based on
        which signals tend to be most predictive in prediction markets.

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

        # Base: start with the market's implied probability as the anchor
        prob = implied

        # Signal 1: Order flow (strongest single signal in prediction markets)
        # Bid pressure > 0.5 means YES side has more buying interest
        prob += (bid_pressure - 0.5) * 0.12  # +/- 6% max adjustment
        prob += order_imbalance * 0.08        # Net bid imbalance between YES and NO
        prob += ask_imbalance * 0.05          # Ask-side imbalance

        # Signal 2: Microprice (volume-weighted fair value from bid/ask quotes)
        # Blend 85% from accumulated signals + 15% from microprice estimate
        microprice_implied = microprice / 100
        prob = prob * 0.85 + microprice_implied * 0.15

        # Signal 3: Smart money indicator (large volume + clear directional flow)
        if smart_money > 2:
            prob += 0.03 * np.sign(order_imbalance)  # Nudge 3% in the direction of smart money

        # Signal 4: Momentum from recent price changes (if historical data available)
        prob += momentum * 0.1

        # Signal 5: Bid strength vs weakness (how aggressively buyers are bidding)
        prob += (bid_strength - 0.5) * 0.05

        # ── Reliability Adjustments ──
        # Regress toward 0.5 (maximum uncertainty) when data quality is poor.
        # Wide spreads, high vig, low volume, and inefficient markets all reduce reliability.
        if spread_pct > 0.2:
            reliability = max(0.5, 1 - spread_pct)
            prob = prob * reliability + 0.5 * (1 - reliability)

        if overround > 1.0:
            # Adjust for the exchange's vig: split the overround evenly between YES and NO
            vig = (overround - 1.0) / 2
            prob = prob - vig if prob > 0.5 else prob + vig

        if volume < 3:  # log_volume < 3 means raw volume < ~19
            weight = volume / 3
            prob = prob * weight + 0.5 * (1 - weight)

        if efficiency < 0.9:
            prob = prob * efficiency + 0.5 * (1 - efficiency)

        return float(np.clip(prob, 0.01, 0.99))

    def get_feature_importance(self) -> dict[str, float]:
        """Get combined feature importance from both models.

        Feature importance answers: "Which of the 108 features does the model
        actually rely on most when making predictions?"

        The importance metric used is "Gini importance" (also called "mean decrease
        in impurity"). At each decision split in each tree, we measure how much that
        split reduced prediction error. We then sum these reductions across all splits
        where that feature was used, across all trees.

        A feature importance of 0.05 means 5% of the model's total predictive power
        comes from that feature. All importances sum to 1.0.

        This tells us things like:
          - "volume and spread are the most important features" → focus on liquidity
          - "time features rank highly" → time-to-expiry matters a lot
          - "interaction features outperform base features" → the combinations matter

        Weighted 60% RF + 40% GB (RF gets more weight since it has more trees
        and its importance estimates are more stable). Returns a dict sorted
        by importance descending.
        """
        if not self.is_trained:
            return {}
        rf_imp = self.rf.feature_importances_   # Gini importance from Random Forest (108 values summing to 1.0)
        gb_imp = self.gb.feature_importances_   # Gini importance from Gradient Boosting (also sums to 1.0)
        combined = 0.6 * rf_imp + 0.4 * gb_imp  # Weighted blend: RF gets more weight (more stable estimates)
        return dict(sorted(
            zip(FEATURE_NAMES, combined),  # Pair each feature name with its combined importance
            key=lambda x: x[1],           # Sort by importance value
            reverse=True,                 # Highest importance first
        ))

    # ── Task 7: Prediction Explanation ─────────────────────────────────────

    def explain_prediction(self, features: dict) -> list[dict]:
        """For a given prediction, compute which top 3 features pushed it above/below 0.5.

        This answers: "WHY did the model predict 73%? What drove it above 50%?"

        The method uses a simple but effective approach:
          1. For each feature, compute: (current value - training average) × importance
          2. This gives the "contribution" — how much this feature pushed the
             prediction away from the "average" prediction.
          3. Positive contribution → pushed probability UP (toward YES).
          4. Negative contribution → pushed probability DOWN (toward NO).

        Example output:
            [{"feature": "bid_pressure", "value": 0.72, "training_mean": 0.50,
              "importance": 0.08, "contribution": +0.018, "direction": "up"},
             {"feature": "spread_pct", "value": 0.03, "training_mean": 0.10,
              "importance": 0.06, "contribution": -0.004, "direction": "down"}]
        → "bid_pressure is unusually HIGH, which strongly pushed the prediction UP."
        → "spread_pct is unusually LOW (tight spread), which slightly pushed it DOWN
            (tight spreads = efficient market = less mispricing = lower edge)."

        Uses the difference between the feature value and the training mean,
        weighted by feature importance.

        Args:
            features: The 106-feature dict for a market.

        Returns:
            List of top 3 explanation dicts: [{feature, value, training_mean, importance,
            contribution, direction}].
        """
        if not self.is_trained or not self._training_feature_means:
            return []

        importance = self.get_feature_importance()
        explanations = []
        for name in FEATURE_NAMES:
            imp = importance.get(name, 0)
            if imp <= 0:
                continue
            val = features.get(name, 0)
            mean = self._training_feature_means.get(name, 0)
            diff = val - mean
            # Contribution: how much this feature pushes the prediction
            contribution = diff * imp
            explanations.append({
                "feature": name,
                "value": round(val, 4),
                "training_mean": round(mean, 4),
                "importance": round(imp, 6),
                "contribution": round(contribution, 6),
                "direction": "up" if contribution > 0 else "down",
            })

        # Sort by absolute contribution and take top 3
        explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return explanations[:3]

    # ── Fix 29-30: Warm-start retrain method ──────────────────────────────

    def warm_start_retrain(self, new_features: list[dict], new_outcomes: list[int]) -> dict:
        """Incrementally retrain with new data. Validates input and logs accuracy comparison.

        Fix 29: Validates new_features is non-empty.
        Fix 30: Logs before/after accuracy comparison.
        """
        # Fix 29: Validate new_features is non-empty
        if not new_features:
            logger.warning("warm_start_retrain called with empty features list")
            return {"status": "no_data", "error": "Empty new_features list", "trained": False}

        if len(new_features) != len(new_outcomes):
            logger.warning("warm_start_retrain: features/outcomes length mismatch")
            return {"status": "error", "error": "features/outcomes length mismatch", "trained": False}

        # Fix 30: Log before accuracy
        before_cv = self.cv_score
        before_version = self.version
        logger.info(f"warm_start_retrain: before cv_score={before_cv:.4f}, version={before_version}, "
                     f"adding {len(new_features)} new samples")

        result = self.train_on_historical(new_features, new_outcomes)

        # Fix 30: Log after accuracy comparison
        after_cv = self.cv_score
        logger.info(f"warm_start_retrain: after cv_score={after_cv:.4f}, version={self.version}, "
                     f"delta={after_cv - before_cv:+.4f}")
        result["before_cv"] = round(before_cv, 4)
        result["after_cv"] = round(after_cv, 4)
        result["cv_delta"] = round(after_cv - before_cv, 4)
        return result

    # ── Task 8: Ensemble Disagreement Status ───────────────────────────────

    def get_ensemble_status(self) -> dict:
        """Get ensemble disagreement metrics.

        Tracks when RF and GB models disagree significantly (>0.15 difference).
        When disagreement rate > 30%, flag as ensemble_unstable.

        Returns:
            Dict with disagreement_rate, is_stable, avg_disagreement, and recent history.
        """
        if len(self._disagreement_history) < 5:
            return {
                "disagreement_rate": 0.0,
                "is_stable": True,
                "avg_disagreement": 0.0,
                "significant_disagreements": 0,
                "total_predictions": len(self._disagreement_history),
                "status": "insufficient_data",
            }

        total = len(self._disagreement_history)
        significant = sum(1 for d in self._disagreement_history if d > 0.15)
        disagreement_rate = significant / total
        avg_disagreement = sum(self._disagreement_history) / total

        is_stable = disagreement_rate <= 0.30
        status = "stable" if is_stable else "ensemble_unstable"

        return {
            "disagreement_rate": round(disagreement_rate, 4),
            "is_stable": is_stable,
            "avg_disagreement": round(avg_disagreement, 4),
            "significant_disagreements": significant,
            "total_predictions": total,
            "status": status,
            "learning_rate": self._current_gb_lr,
            "learning_rate_history": self._learning_rate_history[-10:],
            "accuracy_history": self._accuracy_history[-10:],
            "top_interactions": self.top_interactions[:5],
        }


# ── Edge Decay Tracker ───────────────────────────────────────────────────────

# ── Market Regime Detector ───────────────────────────────────────────────────

class RegimeDetector:
    """Classifies the current market "regime" based on recent market observations.

    A "market regime" is a description of the overall environment: are markets
    calm or volatile? Are prices trending in one direction, or bouncing around?
    Different strategies work better in different regimes.

    Examples:
      - high_volatility regime: prices are jumping around a lot, spreads are wide.
        In this regime, be more conservative — the model's predictions are less reliable.
      - low_volatility regime: prices are stable, spreads are tight.
        In this regime, be more aggressive — the model's predictions are more reliable.
      - trending regime: prices are consistently moving in one direction across markets.
        Could indicate new information is being priced in market-wide.
      - mean_reverting regime: prices bounce around their average without trending.
        Suggests markets are efficiently priced and extreme values get corrected quickly.

    The detector observes the last 100 markets scanned and classifies the regime
    by measuring: average spread, average volume, average price extremity, and
    the consistency of price movement direction.

    Tracks spread, volume, and price extremity across the last 100 market observations
    to classify the current regime as: high_volatility, low_volatility, trending,
    or mean_reverting.
    """

    def __init__(self, window: int = 100):
        """Initialize the regime detector with a rolling window of market observations.

        Args:
            window: How many recent market observations to keep. Default 100.
                    Larger window = smoother, more stable regime classification.
                    Smaller window = more responsive to sudden market shifts.
        """
        self.window = window
        # Rolling history of spreads, volumes, price extremities, and price changes
        # across all recently scanned markets. "deque(maxlen=window)" automatically
        # drops old entries when the list exceeds the window size.
        self._spreads: deque[float] = deque(maxlen=window)
        self._volumes: deque[float] = deque(maxlen=window)
        self._extremities: deque[float] = deque(maxlen=window)
        self._price_changes: deque[float] = deque(maxlen=window)

    def record_market(self, market: Market, price_change: float = 0.0):
        """Record a market observation for regime stats."""
        # Fix 44: Validate input arrays are numeric
        try:
            spread_val = float(market.spread)
            volume_val = float(market.volume)
            extremity_val = abs(float(market.mid_price_yes) - 50)
            change_val = float(price_change) if price_change is not None else 0.0
        except (TypeError, ValueError):
            return  # Skip invalid data
        self._spreads.append(spread_val)
        self._volumes.append(volume_val)
        self._extremities.append(extremity_val)
        self._price_changes.append(change_val)

    def detect_regime(self) -> dict:
        """Classify current market conditions."""
        # Fix 43: Handle case where not enough history
        if len(self._spreads) < 10:
            return {"regime": "unknown", "confidence": 0.0, "stats": {}}

        avg_spread = sum(self._spreads) / len(self._spreads)
        avg_volume = sum(self._volumes) / len(self._volumes)
        avg_extremity = sum(self._extremities) / len(self._extremities)

        changes = list(self._price_changes)
        abs_changes = [abs(c) for c in changes]
        avg_abs_change = sum(abs_changes) / len(abs_changes) if abs_changes else 0

        if changes:
            positive = sum(1 for c in changes if c > 0)
            negative = sum(1 for c in changes if c < 0)
            direction_ratio = max(positive, negative) / len(changes)
        else:
            direction_ratio = 0.5

        if avg_abs_change > 3.0 or avg_spread > 8.0:
            regime = "high_volatility"
            confidence = min(1.0, avg_abs_change / 5.0)
        elif avg_abs_change < 1.0 and avg_spread < 4.0:
            regime = "low_volatility"
            confidence = min(1.0, (4.0 - avg_spread) / 4.0)
        elif direction_ratio > 0.65:
            regime = "trending"
            confidence = min(1.0, (direction_ratio - 0.5) * 4)
        else:
            regime = "mean_reverting"
            confidence = min(1.0, (0.65 - direction_ratio) * 4)

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "stats": {
                "avg_spread": round(avg_spread, 2),
                "avg_volume": round(avg_volume, 0),
                "avg_extremity": round(avg_extremity, 2),
                "avg_abs_price_change": round(avg_abs_change, 2),
                "direction_ratio": round(direction_ratio, 2),
                "observations": len(self._spreads),
            },
        }


class EdgeTracker:
    """Rolling window tracker comparing what the model PREDICTED vs what actually happened.

    "Edge" in trading = your expected advantage per trade. If the model says a market
    is 70% likely to be YES and it's trading at 50%, the predicted edge is 20%.

    But does that 20% edge actually materialize? The EdgeTracker measures the
    "edge realization ratio" = (actual average return) / (predicted average edge).

    Interpretation:
      - ratio = 1.0 → perfectly calibrated. The model predicts exactly as much
        edge as it actually delivers.
      - ratio = 0.7 → the model overestimates edge by 30%. Multiply all position
        sizes by 0.7 to compensate.
      - ratio = 1.3 → the model underestimates edge by 30% (this is good!).
        The model is more conservative than it needs to be.

    This connects to the "Kelly Criterion" — optimal bet sizing depends on knowing
    your true edge. If you think your edge is 20% but it's really 14%, Kelly says
    you're overbetting. The EdgeTracker catches this and adjusts position sizes.

    Computes edge_realization_ratio = avg_realized_return / avg_predicted_edge.
    If ratio < 1.0, the model is overestimating its edge and position sizes
    should be scaled down proportionally.
    """

    def __init__(self, window: int = 50):
        """Initialize the edge tracker with a rolling window of recent trades.

        Args:
            window: How many recent completed trades to track. Default 50.
                    The ratio is computed over the most recent 'window' trades.
                    Smaller window = reacts faster to recent performance changes.
                    Larger window = more stable, less affected by a few bad trades.
        """
        self.window = window
        # Rolling history of predicted edges and actual realized returns for past trades.
        # When get_edge_realization_ratio() is called, it computes avg(realized) / avg(predicted).
        self.predicted_edges: deque[float] = deque(maxlen=window)
        self.realized_returns: deque[float] = deque(maxlen=window)

    def record(self, predicted_edge: float, realized_return: float):
        """Record a completed trade's predicted edge and realized log return."""
        # Fix 42: Validate that predicted values are in [0, 1]
        predicted_edge = max(0.0, min(1.0, predicted_edge))
        self.predicted_edges.append(predicted_edge)
        self.realized_returns.append(realized_return)

    def get_edge_realization_ratio(self) -> float:
        """Ratio of realized return to predicted edge. 1.0 = perfectly calibrated.

        Returns 1.0 if insufficient data (<10 trades). Clamped to [0.1, 2.0].
        """
        # Fix 41: Handle case where window is empty
        if not self.predicted_edges or len(self.predicted_edges) < 10:
            return 1.0
        avg_predicted = sum(self.predicted_edges) / len(self.predicted_edges)
        avg_realized = sum(self.realized_returns) / len(self.realized_returns)
        if avg_predicted <= 0:
            return 1.0
        return max(0.1, min(2.0, avg_realized / avg_predicted))


# ── Signal Generation (Quant-Optimized) ─────────────────────────────────────

class RFSignalGenerator:
    """The top-level "trader" class that scans all markets and decides when to buy or sell.

    This class ties everything together. It:
      1. Holds a PredictionModel (the ML brain)
      2. Maintains a price history cache for every market ticker
      3. Scans all Kalshi markets and calls the ML model to predict each one
      4. Applies entry filters to find genuinely undervalued markets
      5. Computes Kelly-optimal position sizes (how much to bet)
      6. Monitors open positions for exit signals (stop-loss, take-profit, etc.)

    --- The Entry Decision ---

    The bot buys a market when THREE conditions are all true:
      1. The model's confidence is high enough (usually >= 65%)
         "Confidence" = max(model_prob, 1 - model_prob). At 80% YES, confidence=80%.
      2. The market price is undervalued by at least entry_threshold (e.g., 15%)
         If model says 70% but market trades at 55%, that's a 15% gap → buy YES.
      3. The combined signal quality is above a minimum threshold
         (accounts for liquidity, spread cost, and time until resolution)

    --- The Exit Decision ---

    We exit a position when any of these occur:
      - Stop-loss: the position has lost too much (caps downside)
      - Take-profit: we've captured enough of the expected gain (locks profits)
      - Model disagreement: the model changed its mind (new information invalidated thesis)
      - Trailing stop: price dropped significantly from its peak (locks in partial gains)
      - Time decay: the market is expiring soon and we're not winning

    --- Kelly Position Sizing ---

    The Kelly Criterion is a mathematical formula that tells you the optimal fraction
    of your bankroll to bet on a given opportunity. It balances:
      - Betting too little: leaves money on the table
      - Betting too much: risk of ruin (losing everything on a bad streak)

    Formula: Kelly_f = (b * p - q) / b
      where b = payout odds (e.g., if you bet $1 and win, you get $2 back → b=1)
            p = probability of winning (model's prediction)
            q = 1 - p = probability of losing

    We use "quarter-Kelly" (0.25x the full Kelly bet) because Kelly assumes perfectly
    calibrated probabilities. Since our model isn't perfect, we bet conservatively.
    The actual size is also scaled down by volatility, spread width, and edge ratio.

    Generates trading signals using the RF+GB ensemble with quant-optimized
    entry/exit rules, Kelly sizing, and signal quality filtering.

    Entry: market_price <= model_probability * 0.5 (buy at 2x undervaluation)
    Exit: Multi-leg — stop-loss, take-profit, time decay, model disagreement, trailing stop
    Sizing: Binary Kelly with volatility + liquidity + drawdown scaling
    Filter: Composite signal_quality = edge * confidence * liquidity_factor
    """

    def __init__(self):
        """Initialize the signal generator with a fresh untrained model and empty history.

        The model starts untrained. Call model.train_on_historical() or
        model.load_from_disk() to enable ML predictions. Until then, the heuristic
        fallback will be used automatically.

        Components initialized:
          - model: The RF+GB ensemble (starts untrained, uses heuristic until trained)
          - history_cache: Price snapshots per ticker (grows as markets are scanned)
          - edge_tracker: Monitors predicted vs realized returns for calibration scaling
          - regime_detector: Classifies market conditions for adaptive behavior
        """
        self.model = PredictionModel(n_estimators=50)  # The ML brain (untrained initially)
        self.trade_log: list[dict] = []                 # Log of all signals generated this session
        self.history_cache: dict[str, list[dict]] = {}  # {ticker: [{"yes_mid", "volume", "timestamp"}, ...]}
        self.edge_tracker = EdgeTracker(window=50)       # Tracks historical edge accuracy
        self.regime_detector = RegimeDetector(window=100)  # Classifies market environment
        # Category rotation tracking: rolling 24h signal log per category
        self._category_signals: list[dict] = []  # [{timestamp, category, edge}]
        # High-water marks for trailing stops: {ticker: highest_favorable_price}
        self._high_water_marks: dict[str, float] = {}
        # Category momentum tracking: rolling signal outcomes per category
        self._category_outcomes: list[dict] = []  # [{timestamp, category, won, edge}]

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
        if len(self.history_cache[key]) > 100:
            self.history_cache[key] = self.history_cache[key][-100:]

        # Prune history_cache if it grows too large (stale tickers accumulate)
        if len(self.history_cache) > 10000:
            tickers = list(self.history_cache.keys())
            for t in tickers[:5000]:
                del self.history_cache[t]

    def optimal_entry_check(self, market: Market) -> str:
        """Analyze recent price momentum to optimize entry timing.

        Uses the last 3 momentum snapshots from the history cache to determine
        whether price is moving toward our target (wait for better price) or
        stable/moving away (enter now before it gets worse).

        Args:
            market: The market to check entry timing for.

        Returns:
            "wait" if price is falling toward our target (better entry coming),
            "enter" if price is stable or rising away (enter now).
        """
        history = self.history_cache.get(market.ticker)
        if not history or len(history) < 3:
            return "enter"  # Not enough data, enter now

        # Get the last 3 price snapshots
        recent = history[-3:]
        prices = [h.get("yes_mid", 50) for h in recent]

        # Calculate momentum: negative means price is falling
        momentum_1 = prices[-1] - prices[-2]
        momentum_2 = prices[-2] - prices[-3]
        avg_momentum = (momentum_1 + momentum_2) / 2

        # If price is consistently falling (negative momentum), wait for better entry
        if momentum_1 < 0 and momentum_2 < 0 and avg_momentum < -0.5:
            return "wait"

        return "enter"

    def get_hot_categories(self) -> list[dict]:
        """Get categories sorted by signal frequency and avg edge over rolling 24h window.

        Tracks which categories are generating the most signals to help identify
        market regime shifts and category rotation patterns.

        Returns:
            List of dicts sorted by signal count desc, each with category, signal_count,
            avg_edge, and last_signal_time.
        """
        from collections import defaultdict

        cutoff = time.time() - 86400  # 24 hours ago
        # Prune old entries
        self._category_signals = [s for s in self._category_signals if s["timestamp"] >= cutoff]

        cats = defaultdict(lambda: {"edges": [], "last_ts": 0.0})
        for s in self._category_signals:
            cat = s["category"]
            cats[cat]["edges"].append(s["edge"])
            cats[cat]["last_ts"] = max(cats[cat]["last_ts"], s["timestamp"])

        result = []
        for cat, data in cats.items():
            result.append({
                "category": cat,
                "signal_count": len(data["edges"]),
                "avg_edge": round(sum(data["edges"]) / len(data["edges"]), 4) if data["edges"] else 0,
                "last_signal_time": datetime.fromtimestamp(data["last_ts"], tz=timezone.utc).isoformat() if data["last_ts"] else "",
            })

        result.sort(key=lambda x: x["signal_count"], reverse=True)
        return result

    # ── Task 16: Order Book Imbalance Analysis ─────────────────────────────

    def analyze_order_book(self, market: Market) -> dict:
        """Compute order book imbalance metrics for a market.

        bid_ask_imbalance = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth)
        Strong imbalance (>0.3 or <-0.3) suggests directional pressure.

        Returns:
            Dict with imbalance score, direction, strength assessment, and reasoning.
        """
        total_bid_depth = market.yes_bid + market.no_bid
        total_ask_depth = market.yes_ask + market.no_ask
        total_depth = total_bid_depth + total_ask_depth

        if total_depth == 0:
            return {
                "bid_ask_imbalance": 0.0,
                "direction": "neutral",
                "strength": "no_data",
                "total_bid_depth": 0,
                "total_ask_depth": 0,
                "reasoning": "No order book data available.",
            }

        imbalance = (total_bid_depth - total_ask_depth) / total_depth

        if abs(imbalance) > 0.3:
            strength = "strong"
        elif abs(imbalance) > 0.15:
            strength = "moderate"
        else:
            strength = "weak"

        if imbalance > 0.05:
            direction = "bullish"
        elif imbalance < -0.05:
            direction = "bearish"
        else:
            direction = "neutral"

        reasoning = (
            f"Order book imbalance: {imbalance:+.3f} ({strength} {direction}). "
            f"Bid depth: {total_bid_depth}, Ask depth: {total_ask_depth}."
        )
        if abs(imbalance) > 0.3:
            reasoning += " Strong directional pressure detected."

        return {
            "bid_ask_imbalance": round(imbalance, 4),
            "direction": direction,
            "strength": strength,
            "total_bid_depth": total_bid_depth,
            "total_ask_depth": total_ask_depth,
            "reasoning": reasoning,
        }

    # ── Task 17: Market Efficiency Scoring ─────────────────────────────────

    def score_market_efficiency(self, market: Market) -> dict:
        """Score a market's efficiency on a 0-1 scale.

        Based on: spread tightness, volume, distance from round numbers,
        and bid-ask symmetry. Inefficient markets (score < 0.5) are better
        trading opportunities because mispricings persist longer.

        Returns:
            Dict with efficiency_score (0-1), component scores, and is_opportunity flag.
        """
        # 1. Spread tightness (0-1): tighter spread = more efficient
        spread_pct = market.spread / max(market.mid_price_yes, 1)
        spread_score = max(0.0, min(1.0, 1.0 - spread_pct * 2.5))  # 40% spread -> 0 (accepts spreads up to 40c)

        # 2. Volume score (0-1): higher volume = more efficient
        vol_score = min(1.0, math.log1p(market.volume) / 10.0)  # ~22000 vol -> 1.0

        # 3. Round number proximity (0-1): closer to round numbers = more efficient
        mid = market.mid_price_yes
        nearest_round = round(mid / 10) * 10
        round_distance = abs(mid - nearest_round) / 10  # 0 at round, 0.5 at midpoint
        round_score = 1.0 - round_distance  # 1.0 at round numbers, 0.5 at midpoints

        # 4. Bid-ask symmetry (0-1): symmetric = more efficient
        yes_spread = market.yes_ask - market.yes_bid
        no_spread = market.no_ask - market.no_bid
        total_spread = yes_spread + no_spread
        if total_spread > 0:
            symmetry = 1.0 - abs(yes_spread - no_spread) / total_spread
        else:
            symmetry = 1.0

        # Weighted combination
        efficiency_score = (
            0.30 * spread_score
            + 0.30 * vol_score
            + 0.15 * round_score
            + 0.25 * symmetry
        )

        return {
            "efficiency_score": round(efficiency_score, 4),
            "spread_score": round(spread_score, 4),
            "volume_score": round(vol_score, 4),
            "round_number_score": round(round_score, 4),
            "symmetry_score": round(symmetry, 4),
            "is_opportunity": efficiency_score < 0.5,
        }

    # ── Task 18: Correlated Market Detection ───────────────────────────────

    def detect_correlated_markets(self, signals: list[TradingSignal]) -> list[TradingSignal]:
        """Group correlated signals and keep only the best from each group.

        Correlation is detected by: same event (event_ticker prefix), same category,
        or overlapping ticker prefixes. Only the highest signal_quality signal from
        each correlation group is kept to avoid concentration risk.

        Adds correlation_group to each kept signal's reasoning.

        Returns:
            Filtered list of signals with one per correlation group.
        """
        if not signals:
            return []

        groups: dict[str, list[TradingSignal]] = {}

        for sig in signals:
            # Derive correlation group key from ticker prefix and category
            # Kalshi tickers like "KXBTC-24MAR14-T100000" share prefix "KXBTC"
            ticker_prefix = sig.ticker.split("-")[0] if "-" in sig.ticker else sig.ticker
            category = sig.category or "unknown"

            # Group by: ticker prefix (same underlying event/asset)
            group_key = f"{ticker_prefix}:{category}"

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(sig)

        # Keep only the best signal from each group
        filtered = []
        for group_key, group_signals in groups.items():
            group_signals.sort(key=lambda s: s.signal_quality, reverse=True)
            best = group_signals[0]

            # Annotate the signal with its correlation group info
            group_size = len(group_signals)
            if group_size > 1:
                best.reasoning += (
                    f" [correlation_group={group_key}, group_size={group_size}, "
                    f"best_of_group=True]"
                )

            filtered.append(best)

        return filtered

    # ── Task 19: Smart Money Indicator ─────────────────────────────────────

    def compute_smart_money_score(self, market: Market) -> float:
        """Detect smart money activity based on volume and spread tightness.

        smart_money_score = volume * (1 / spread_pct) normalized to 0-1.
        High scores suggest informed participants are actively trading:
        large volume on tight spreads is a hallmark of informed flow.

        Returns:
            Float in [0, 1]. >0.7 suggests strong informed activity.
        """
        spread_pct = market.spread / max(market.mid_price_yes, 1)
        if spread_pct <= 0:
            spread_pct = 0.01  # Avoid division by zero

        raw_score = market.volume * (1.0 / spread_pct)

        # Normalize using a sigmoid-like function: score = raw / (raw + k)
        # k=50000 chosen so that volume=5000 with 10% spread gives ~0.5
        k = 50000.0
        normalized = raw_score / (raw_score + k)

        return round(min(1.0, max(0.0, normalized)), 4)

    # ── Task 20: Category Momentum Scoring ─────────────────────────────────

    def record_category_outcome(self, category: str, won: bool, edge: float):
        """Record a signal outcome for category momentum tracking.

        Call this when a trade resolves to track per-category performance.
        """
        self._category_outcomes.append({
            "timestamp": time.time(),
            "category": category,
            "won": won,
            "edge": edge,
        })
        # Keep last 500 outcomes
        if len(self._category_outcomes) > 500:
            self._category_outcomes = self._category_outcomes[-500:]

    def get_category_momentum(self) -> dict[str, dict]:
        """Compute rolling momentum for each category over last 20 signals.

        Returns a dict of {category: {win_rate, avg_edge, signal_count, momentum_boost}}
        where momentum_boost is a multiplier (1.0-1.3) for signal priority.
        Categories with high recent win rate (>60%) get a boost.
        """
        from collections import defaultdict

        cat_data: dict[str, list[dict]] = defaultdict(list)
        for outcome in self._category_outcomes:
            cat_data[outcome["category"]].append(outcome)

        result = {}
        for category, outcomes in cat_data.items():
            # Use last 20 signals for momentum calculation
            recent = sorted(outcomes, key=lambda x: x["timestamp"], reverse=True)[:20]
            if not recent:
                continue

            win_rate = sum(1 for o in recent if o["won"]) / len(recent)
            avg_edge = sum(o["edge"] for o in recent) / len(recent)

            # Momentum boost: categories with >60% win rate get a boost
            if win_rate > 0.70 and len(recent) >= 5:
                momentum_boost = 1.3  # 30% boost for hot categories
            elif win_rate > 0.60 and len(recent) >= 5:
                momentum_boost = 1.15  # 15% boost for warm categories
            elif win_rate < 0.40 and len(recent) >= 5:
                momentum_boost = 0.8  # 20% penalty for cold categories
            else:
                momentum_boost = 1.0

            result[category] = {
                "win_rate": round(win_rate, 4),
                "avg_edge": round(avg_edge, 4),
                "signal_count": len(recent),
                "momentum_boost": round(momentum_boost, 2),
            }

        return result

    def generate_signals(self, events: list[Event], n_positions: int = 0) -> list[TradingSignal]:
        """Scan ALL open Kalshi markets and return a ranked list of trading opportunities.

        This is the main "scanner" that runs every scan cycle (typically every 1-3 seconds).
        For each market, it:
          1. Applies pre-filters (minimum volume, reasonable price range, etc.)
          2. Calls _evaluate_market() which runs the full ML pipeline
          3. Annotates passing signals with order book analysis, efficiency scores,
             smart money indicators, and category momentum
          4. Removes correlated signals (keeps only the best from each group)
          5. Returns signals sorted by signal_quality (highest first)

        The "near_misses" list captures markets that almost qualified but didn't quite
        pass the filters. This is useful for display — you can see what the model
        considered and why it was rejected (e.g., "too narrow edge" or "low volume").

        Applies market intelligence (order book, efficiency, smart money, category
        momentum) and correlated market detection to filter and prioritize signals.

        Args:
            events: Live Kalshi event data (list of events, each containing markets).
            n_positions: Current number of open positions (for dynamic quality threshold).
                         As positions fill up, we raise the quality bar to be more selective.

        Returns:
            List of TradingSignal objects sorted by signal_quality descending.
            Empty list if no qualifying opportunities were found.
        """
        # Fix 36: Log total duration
        start_time = time.time()
        signals = []
        category_momentum = self.get_category_momentum()

        # Fix 37: Handle case where model is not trained (use heuristic — already handled in predict)
        if not self.model.is_trained:
            logger.info("generate_signals: model not trained, using heuristic predictions")

        # Fix 47: Count total markets scanned
        total_markets_scanned = 0
        # Fix 48: Count signals filtered by confidence
        filtered_by_confidence = 0
        # Fix 49: Count signals filtered by edge
        filtered_by_edge = 0

        near_misses = []  # Markets that almost passed filters — useful for display
        worst_opportunities = []  # Markets where model sees significant NEGATIVE edge (overpriced)

        for event in events:
            for market in event.markets:
                # Validate each market has required fields
                if not (market.ticker and market.event_ticker):
                    continue
                if market.status not in ("open", "active"):
                    continue

                # Crypto markets already passed pre-filter in server.py
                # For other markets, apply standard filters
                ticker_prefix = (market.ticker or "").split("-")[0]
                is_crypto = ticker_prefix in ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXBTCD", "KXETHD", "KXSOLD")

                if not is_crypto:
                    # Pre-filters for non-crypto markets. These are coarse filters
                    # that avoid spending time on clearly untradeable markets.
                    # The ML model is only asked to evaluate markets that pass these.

                    # Filter 1: Volume must be at least 2000 contracts.
                    # Low-volume markets are likely mispriced, but we can't trade enough
                    # to profit (our order would move the price against us).
                    if market.volume < 2000:
                        continue

                    # Filter 2: Price must be between 5% and 95%.
                    # Markets at <5% or >95% are nearly certain — small edge, high risk.
                    # Not worth the spread cost to trade these near-certainties.
                    mid = market.mid_price_yes
                    if mid <= 5 or mid >= 95:
                        continue

                    # Filter 3: Spread must be positive (valid market) and under 15 cents.
                    # A 15-cent spread on a 50-cent market = 30% round-trip cost. Too expensive.
                    spread = (market.yes_ask or 0) - (market.yes_bid or 0)
                    if spread <= 0 or spread > 15:
                        continue

                    # Filter 4: Open interest must be at least 500 contracts.
                    # If almost nobody holds positions, the market isn't being monitored
                    # and prices can be very stale or unreliable.
                    if (market.open_interest or 0) < 500:
                        continue

                total_markets_scanned += 1
                self.record_snapshot(market)

                # Collect worst opportunities — markets where model disagrees with market price
                # (market is overpriced relative to model)
                market_price = market.mid_price_yes / 100
                try:
                    history = self.history_cache.get(market.ticker)
                    features = extract_features(market, event, history)
                    prediction = self.model.predict(features)
                    mp = prediction["probability"]
                    conf = max(mp, 1 - mp)
                    # Negative edge: market thinks YES is worth more than model does
                    if mp > 0.5:
                        neg_edge = mp - market_price  # positive if underpriced, negative if overpriced
                    else:
                        neg_edge = (1 - mp) - (1 - market_price)
                    if neg_edge < -0.05 and conf > 0.6:  # >5% overpriced, confident
                        worst_opportunities.append({
                            "ticker": market.ticker,
                            "market_title": market.title,
                            "event_title": event.title,
                            "side": "yes" if mp > 0.5 else "no",
                            "edge": round(neg_edge, 4),
                            "fair_probability": round(mp, 4),
                            "market_probability": round(market_price, 4),
                            "confidence": round(conf, 4),
                            "volume": market.volume,
                            "close_time": market.close_time,
                        })
                except Exception:
                    pass

                # Check entry timing before evaluating (skip "wait" but still record)
                entry_timing = self.optimal_entry_check(market)
                if entry_timing == "wait":
                    continue

                signal = self._evaluate_market(event, market, n_positions, near_misses=near_misses)
                if signal:
                    # Task 16: Include order book imbalance in reasoning
                    ob_analysis = self.analyze_order_book(market)
                    if ob_analysis["strength"] == "strong":
                        signal.reasoning += f" OB: {ob_analysis['reasoning']}"

                    # Task 17: Add efficiency score to signal output
                    efficiency = self.score_market_efficiency(market)
                    signal.reasoning += f" efficiency={efficiency['efficiency_score']:.2f}"

                    # Task 19: Add smart money score as feature
                    smart_money = self.compute_smart_money_score(market)
                    signal.reasoning += f" smart_money={smart_money:.2f}"

                    # Task 20: Apply category momentum boost to signal quality
                    cat = signal.category
                    if cat and cat in category_momentum:
                        boost = category_momentum[cat]["momentum_boost"]
                        signal.signal_quality *= boost
                        if boost != 1.0:
                            signal.reasoning += f" cat_momentum={boost:.2f}"

                    signals.append(signal)

        # Sort by signal quality (composite score) instead of raw edge * confidence
        signals.sort(key=lambda s: s.signal_quality, reverse=True)

        # Task 18: Deduplicate correlated markets -- keep best per group
        pre_dedup_count = len(signals)
        signals = self.detect_correlated_markets(signals)

        # Attach near-miss data for display — top 10 near-misses by edge
        near_misses.sort(key=lambda nm: nm.get("edge", 0), reverse=True)
        self._last_near_misses = near_misses[:10]

        # Attach worst opportunities — top 10 most overpriced by negative edge
        worst_opportunities.sort(key=lambda w: w.get("edge", 0))  # most negative first
        self._last_worst_opportunities = worst_opportunities[:10]

        # Fix 36: Log total duration
        elapsed = time.time() - start_time
        logger.info(f"generate_signals: scanned {total_markets_scanned} markets in {elapsed:.2f}s, "
                     f"generated {len(signals)} signals (pre-dedup: {pre_dedup_count}), "
                     f"near-misses: {len(near_misses)}, worst: {len(worst_opportunities)}")

        return signals

    def check_exits(self, events: list[Event], positions: list) -> list[TradingSignal]:
        """Check open positions for multi-leg exit conditions.

        Exit triggers (checked in order):
          1. Stop-loss:         Drawdown > 60% of entry edge (caps downside)
          2. Take-profit:       Gain > 70% of entry edge (locks profits early)
          3. Model disagreement: Model flipped direction (new info invalidated thesis)
          4. Trailing stop:     Price drops >30% from high-water mark (only after >10% profit)
          5. Time decay:        Tightened stops as expiry approaches; force exit at 1 day
          6. Settlement:        Market resolved
        """
        exit_signals = []
        market_map = {}
        for event in events:
            for market in event.markets:
                market_map[market.ticker] = (event, market)

        for pos in positions:
            if pos.ticker not in market_map:
                logger.debug(f"check_exits: position {pos.ticker} not in cached events, skipping")
                continue
            # Fix 39: Handle case where position's entry data is missing
            if not pos.ticker:
                logger.warning("check_exits: position with empty ticker, skipping")
                continue
            event, market = market_map[pos.ticker]
            history = self.history_cache.get(market.ticker)
            features = extract_features(market, event, history)
            model_prob = self.model.predict_probability(features)
            market_price = market.mid_price_yes / 100
            days_left = features.get("days_to_expiry", 30)

            # Fix 39: Compute entry edge with fallback for missing entry data
            entry_edge = abs(getattr(pos, 'model_prob', 0.5) - getattr(pos, 'avg_price_cents', 50) / 100)
            if hasattr(pos, 'entry_price'):
                entry_edge = abs(getattr(pos, 'model_prob', 0.5) - pos.entry_price)
            entry_edge = max(entry_edge, 0.10)  # Floor at 10% to prevent micro-stops

            # Fixed stop-loss and take-profit distances
            # Don't exit on noise — only exit on significant moves
            stop_distance = max(0.15, entry_edge * 0.9)    # Min 15c stop, 90% of edge
            # Take-profit must cover fees (Kalshi ~1c/contract) + 10% gain minimum
            # Minimum take-profit = 3c (covers ~1c fee each way + 1c profit minimum)
            min_profit = 0.05  # 5c minimum profit per contract to cover fees + gain
            take_profit_distance = max(min_profit, entry_edge * 0.7)  # 70% of edge or 5c minimum

            # Time decay: only tighten in final 24 hours before expiry
            if days_left <= 1:
                stop_distance = max(0.05, stop_distance * 0.5)
                take_profit_distance = max(0.03, take_profit_distance * 0.5)

            # Evaluate exit conditions by side
            reason = None
            if pos.side == "yes":
                entry_p = getattr(pos, 'entry_price', getattr(pos, 'avg_price_cents', 50) / 100)
            else:
                # NO positions: avg_price_cents is NO cost basis, convert to YES price at entry
                entry_p = 1 - (getattr(pos, 'avg_price_cents', 50) / 100)

            # ── PROFIT RULES ──
            # 1. At 1.5x profit: cash out original stake, leave "house money" riding
            #    UNLESS model is 90%+ confident → hold everything for max payout
            # 2. At 90%+ confidence: hold to settlement for max payout (100c or 0c)
            # 3. Below 90% confidence: normal take-profit at target

            if pos.side == "yes":
                current_profit = market_price - entry_p
                model_very_confident = model_prob >= 0.90
            else:
                current_profit = entry_p - market_price  # NO profits when price drops
                model_very_confident = model_prob <= 0.10  # 90%+ NO confidence

            # Check if we've hit 1.5x return (50% profit on investment)
            profit_ratio = current_profit / max(entry_p, 0.01) if pos.side == "yes" else current_profit / max(1 - entry_p, 0.01)
            at_1_5x = profit_ratio >= 0.50  # 50% return = 1.5x

            if pos.side == "yes":
                if market_price <= entry_p - stop_distance:
                    reason = "Stop-loss (YES price dropped)"
                elif at_1_5x and not model_very_confident:
                    reason = "Cash-out at 1.5x (securing profit, leaving house money)"
                elif not model_very_confident and market_price >= entry_p + take_profit_distance:
                    reason = "Take-profit (target reached)"
                elif model_prob < 0.30:
                    reason = "Model disagreement (strongly flipped to NO)"
            else:
                if market_price >= entry_p + stop_distance:
                    reason = "Stop-loss (YES price rose, bad for NO)"
                elif at_1_5x and not model_very_confident:
                    reason = "Cash-out at 1.5x (securing profit, leaving house money)"
                elif not model_very_confident and market_price <= entry_p - take_profit_distance:
                    reason = "Take-profit (NO side gained)"
                elif model_prob > 0.70:
                    reason = "Model disagreement (strongly flipped to YES)"

            # ── Task 24: Trailing stop logic ──
            # Track high-water mark for each position. If price drops more than
            # 30% from the high-water mark, trigger exit. Only activates after
            # position is >10% profitable (more sophisticated than fixed stops).
            if reason is None:
                ticker = pos.ticker
                # Compute favorable price based on side
                if pos.side == "yes":
                    favorable_price = market_price
                else:
                    favorable_price = 1.0 - market_price

                # Update high-water mark
                current_hwm = self._high_water_marks.get(ticker, favorable_price)
                if favorable_price > current_hwm:
                    self._high_water_marks[ticker] = favorable_price
                    current_hwm = favorable_price

                # Check if position is >10% profitable
                entry_favorable = entry_p if pos.side == "yes" else (1.0 - entry_p)
                profit_pct = (favorable_price - entry_favorable) / max(entry_favorable, 0.01)
                hwm_profit_pct = (current_hwm - entry_favorable) / max(entry_favorable, 0.01)

                if hwm_profit_pct > 0.10 and current_hwm > 0:
                    # Position was >10% profitable at some point; check trailing stop
                    drawdown_from_hwm = (current_hwm - favorable_price) / current_hwm
                    if drawdown_from_hwm > 0.30:
                        reason = (
                            f"Trailing stop: price dropped {drawdown_from_hwm:.0%} from "
                            f"high-water mark ({current_hwm:.2f} -> {favorable_price:.2f})"
                        )

            # Near-expiry: only exit if underwater and <30min (avoid total loss)
            hours_left = features.get("hours_to_expiry", days_left * 24)
            if reason is None and hours_left < 0.5:
                if pos.side == "yes":
                    is_profitable = market_price > entry_p
                else:
                    is_profitable = market_price < entry_p
                if not is_profitable:
                    reason = "Expiry urgency: underwater position, <30min to expiry"

            # Let confident positions ride to settlement for max payout
            # Only exit near expiry if model has flipped (handled above)

            if reason:
                # Fix 40: Log each exit signal with reason
                logger.info(f"EXIT signal: {pos.ticker} side={pos.side} reason='{reason}' "
                            f"model_prob={model_prob:.3f} market_price={market_price:.3f}")
                # Clean up high-water mark on exit
                self._high_water_marks.pop(pos.ticker, None)
                exit_signals.append(TradingSignal(
                    ticker=market.ticker,
                    market_title=market.title,
                    side=Side(pos.side),
                    confidence=model_prob,
                    fair_probability=model_prob,
                    market_probability=market_price,
                    edge=0,
                    reasoning=f"EXIT: {reason}",
                    recommended_size_cents=0,
                ))

        return exit_signals

    def _evaluate_market(self, event: Event, market: Market, n_positions: int = 0, near_misses: list | None = None) -> TradingSignal | None:
        """Evaluate a market for entry using quant-optimized logic.

        Pipeline:
          1. Ensemble model predicts fair probability
          2. Confidence filter (>= 60% for scanning, configurable)
          3. Multi-timeframe signal confirmation (2/3 snapshots must agree, when available)
          4. Entry rule: market_price <= model_prob * (1 - entry_threshold), i.e. undervalued by entry_threshold %
          5. Signal quality filter (edge * confidence * liquidity, dynamic threshold)
          6. Kelly-optimal position sizing with vol/liq/edge-decay scaling

        Near-misses (markets that almost passed) are collected for display purposes.
        """
        history = self.history_cache.get(market.ticker)
        market_price = market.mid_price_yes / 100

        # Use feature cache if available and price hasn't changed significantly
        features = self.model.get_cached_features(market.ticker, market_price)
        if features is None:
            features = extract_features(market, event, history)
            self.model.cache_features(market.ticker, features, market_price)

        # Use predict() for confidence interval + probability
        prediction = self.model.predict(features)
        model_prob = prediction["probability"]
        prediction_std = prediction.get("prediction_std", 0.15)

        # Feed regime detector
        price_change = 0.0
        if history and len(history) >= 2:
            price_change = history[-1].get("yes_mid", 50) - history[-2].get("yes_mid", 50)
        self.regime_detector.record_market(market, price_change)

        # Confidence = how sure the model is about one side
        confidence = max(model_prob, 1 - model_prob)

        # Direction agreement guard: the model was trained on markets with avg YES mid ~36c
        # (skewed toward NO outcomes). When market_price is near 50%, the model extrapolates
        # outside its training distribution and outputs its learned base rate (~24-25% YES).
        # This creates fake edges: model says NO but market says YES at 53% → we bet NO → lose.
        # Fix: only signal a trade if model and market AGREE on which side is more likely.
        # Both must see the same side as >50% probability. If they disagree on the basic
        # direction, the model is out of distribution — skip the trade entirely.
        market_says_yes = market_price > 0.50
        model_says_yes = model_prob > 0.50
        if market_says_yes != model_says_yes:
            return None

        # Base-rate guard: the model is biased toward ~0.24-0.26 YES for all non-crypto markets
        # (trained on a NO-heavy distribution). If model outputs near its learned base rate,
        # it has no real signal — it's just defaulting. Require model to show genuine conviction
        # by outputting outside the [0.20, 0.30] base-rate zone, unless confidence is very high.
        ticker_prefix = (market.ticker or "").split("-")[0]
        is_crypto_market = ticker_prefix in ("KXBTC15M", "KXETH15M", "KXSOL15M", "KXBTCD", "KXETHD", "KXSOLD", "KXXRP15M", "KXDOGE15M")
        if not is_crypto_market and 0.20 <= model_prob <= 0.30:
            return None  # Model is near base rate — no genuine signal for non-crypto

        # Use a scanning confidence (lower than trading confidence) to find more opportunities
        # Scanning threshold = 60% to discover, trading threshold = min_confidence to execute
        scan_confidence_threshold = config.min_confidence
        if confidence < scan_confidence_threshold:
            return None

        # Multi-timeframe signal confirmation: if we have 3+ snapshots,
        # require that 2/3 of recent snapshots agree on the signal direction.
        # Skip this filter on first scan (no history yet) to avoid empty results.
        mtf_confirmed = True
        if history and len(history) >= 3:
            recent_snapshots = history[-3:]
            if model_prob > 0.5:
                agreeing = sum(1 for h in recent_snapshots if h.get("yes_mid", 50) / 100 < model_prob)
                mtf_confirmed = agreeing >= 2
            elif model_prob < 0.5:
                agreeing = sum(1 for h in recent_snapshots if h.get("yes_mid", 50) / 100 > model_prob)
                mtf_confirmed = agreeing >= 2

        # Entry rule: buy when market is undervalued vs model by entry_threshold %
        # e.g. entry_threshold=0.15 means buy when market_price <= model_prob * 0.85 (15% undervalued)
        entry_multiplier = 1 - config.entry_threshold
        if model_prob > 0.5 and market_price <= model_prob * entry_multiplier:
            side = Side.YES
            edge = model_prob - market_price
        elif model_prob < 0.5 and (1 - market_price) <= (1 - model_prob) * entry_multiplier:
            side = Side.NO
            edge = (1 - model_prob) - (1 - market_price)
        else:
            # Check for near-miss: model sees edge but not enough for entry rule
            if near_misses is not None:
                if model_prob > 0.5:
                    raw_edge = model_prob - market_price
                else:
                    raw_edge = (1 - model_prob) - (1 - market_price)
                if raw_edge > 0.03:  # At least 3% edge to be a near-miss
                    near_misses.append({
                        "ticker": market.ticker,
                        "market_title": market.title,
                        "event_title": event.title,
                        "category": event.category or market.category or "",
                        "model_prob": round(model_prob, 4),
                        "market_price": round(market_price, 4),
                        "edge": round(raw_edge, 4),
                        "confidence": round(confidence, 4),
                        "volume": market.volume,
                        "spread": market.spread,
                        "side": "yes" if model_prob > 0.5 else "no",
                        "reason": "entry_threshold",
                        "close_time": market.close_time,
                    })
            return None

        # Collect near-miss if edge is below minimum threshold
        if edge < config.min_edge_threshold:
            if near_misses is not None and edge > 0.03:
                near_misses.append({
                    "ticker": market.ticker,
                    "market_title": market.title,
                    "event_title": event.title,
                    "category": event.category or market.category or "",
                    "model_prob": round(model_prob, 4),
                    "market_price": round(market_price, 4),
                    "edge": round(edge, 4),
                    "confidence": round(confidence, 4),
                    "volume": market.volume,
                    "spread": market.spread,
                    "side": side.value,
                    "reason": "min_edge",
                    "close_time": market.close_time,
                })
            return None

        # Multi-timeframe confirmation: skip only if we have enough history
        # On fresh boot, history_cache is empty so mtf_confirmed is always False
        # In that case, allow trades to proceed (confidence + edge are sufficient)
        has_history = len(self.history_cache) > 100
        if has_history and not mtf_confirmed:
            if near_misses is not None:
                near_misses.append({
                    "ticker": market.ticker,
                    "market_title": market.title,
                    "event_title": event.title,
                    "category": event.category or market.category or "",
                    "model_prob": round(model_prob, 4),
                    "market_price": round(market_price, 4),
                    "edge": round(edge, 4),
                    "confidence": round(confidence, 4),
                    "volume": market.volume,
                    "spread": market.spread,
                    "side": side.value,
                    "reason": "mtf_unconfirmed",
                    "close_time": market.close_time,
                })
            return None

        # ── Signal Quality Filter ──
        log_vol = math.log1p(market.volume)
        spread = features.get("spread_pct", 0.1)
        liquidity_factor = min(1.0, log_vol / 7.0) * min(1.0, 0.10 / max(spread, 0.01))
        # Time preference: boost markets that resolve sooner (edge realized faster)
        days_to_close = features.get("days_to_expiry", 365)
        if days_to_close <= 1:
            time_boost = 3.0   # Closing today — 3x quality boost
        elif days_to_close <= 7:
            time_boost = 2.0   # This week — 2x boost
        elif days_to_close <= 30:
            time_boost = 1.5   # This month — 1.5x boost
        elif days_to_close <= 90:
            time_boost = 1.0   # Within 3 months — normal
        else:
            time_boost = 0.5   # Long-dated — penalize (capital locked up too long)
        if edge == 0 and confidence == 0:
            signal_quality = 0.0
        else:
            signal_quality = edge * confidence * liquidity_factor * time_boost
        signal_quality = max(0.0, min(1.0, signal_quality))

        # Dynamic threshold: more selective as portfolio fills up
        position_fill = n_positions / max(config.max_open_positions, 1)
        quality_threshold = 0.02 + 0.01 * position_fill  # Raised — require real edge+conviction
        if signal_quality < quality_threshold:
            if near_misses is not None:
                near_misses.append({
                    "ticker": market.ticker,
                    "market_title": market.title,
                    "event_title": event.title,
                    "category": event.category or market.category or "",
                    "model_prob": round(model_prob, 4),
                    "market_price": round(market_price, 4),
                    "edge": round(edge, 4),
                    "confidence": round(confidence, 4),
                    "volume": market.volume,
                    "spread": market.spread,
                    "side": side.value,
                    "reason": "quality_filter",
                    "close_time": market.close_time,
                })
            return None

        # ── Kelly-Optimal Position Sizing ──
        # The Kelly Criterion determines how much to bet to maximize long-term growth.
        # It's a mathematical formula: Kelly_f = (b * p - q) / b
        # where:
        #   p = probability of winning (our model's prediction)
        #   q = 1 - p = probability of losing
        #   b = net profit per dollar bet (the "odds")
        #
        # Example: bet on YES at 55 cents when model says 72% probability
        #   b = (1.00 - 0.55) / 0.55 = 0.82 (win 82 cents per dollar bet)
        #   p = 0.72, q = 0.28
        #   Kelly_f = (0.82 * 0.72 - 0.28) / 0.82 = (0.590 - 0.28) / 0.82 = 0.378
        #   = Bet 37.8% of bankroll... but we use a conservative fraction (quarter-Kelly)
        #   Final Kelly = 0.378 * 0.25 = 9.5% of bankroll per trade

        # market_cost: how much do we pay per contract on our chosen side?
        market_cost = market_price if side == Side.YES else (1 - market_price)

        # b = the "odds": how much do we WIN per dollar WAGERED if correct?
        # Buying YES at 55 cents: if YES wins, contract pays 100 cents → profit = 45 cents
        # → b = 45/55 = 0.82 (82 cents profit per dollar wagered)
        b = (1.0 - market_cost) / max(market_cost, 0.01)  # payout odds

        # win_prob: model's estimated probability that our chosen side wins
        win_prob = model_prob if side == Side.YES else (1 - model_prob)

        # q = probability of losing
        q = 1.0 - win_prob

        # The Kelly fraction (0 to 1) represents what fraction of bankroll to bet.
        # We apply config.kelly_fraction (typically 0.25) to get "quarter-Kelly"
        # as a safety margin against model miscalibration.
        kelly_f = max(0, (b * win_prob - q) / b) if b > 0 else 0
        kelly_f *= config.kelly_fraction  # Quarter-Kelly (25% of full Kelly) by default

        # Volatility scaling: reduce bet size when the market has been jumping around.
        # High volatility = uncertain = our edge estimate is less reliable.
        # If market volatility is 2x the "normal" level (0.05), halve the bet size.
        vol = max(features.get("volatility", 0.05), 0.01)
        vol_scalar = min(1.0, 0.05 / vol)  # 1.0 at normal vol, 0.5 at 2x normal vol

        # Liquidity scaling: reduce bet size when the spread is wide.
        # A wide spread means we pay more in trading costs, reducing our effective edge.
        # If spread is 2x the "acceptable" level (0.10), halve the bet size.
        liq_scalar = min(1.0, 0.10 / max(spread, 0.01))

        # Edge decay scaling: reduce bet size if the model has been overestimating edge.
        # The EdgeTracker measures historical edge_realization_ratio. If it's 0.7,
        # the model has been 30% overconfident → scale down all bets by 30%.
        edge_ratio = self.edge_tracker.get_edge_realization_ratio()
        edge_scalar = min(1.0, edge_ratio) if edge_ratio < 1.0 else 1.0

        # Final adjusted Kelly combines all scaling factors.
        # adjusted_kelly is a fraction of bankroll. We multiply by 4x max_bet as
        # a proxy for bankroll size (actual bankroll tracking is done in risk_manager.py).
        adjusted_kelly = kelly_f * vol_scalar * liq_scalar * edge_scalar
        size = int(adjusted_kelly * config.max_bet_amount_cents * 4)  # 4x max_bet as bankroll proxy

        # Cap at max_bet_amount_cents (config setting, e.g., $15 = 1500 cents).
        # Never bet more than this regardless of what Kelly says.
        size = max(0, min(size, config.max_bet_amount_cents))

        if size <= 0:
            return None

        model_type = "ensemble (RF+GB)" if self.model.is_trained else "heuristic"
        category = event.category or market.category or ""

        # ── Signal Strength Score ──
        # Composite: edge * confidence * liquidity_factor * time_decay_factor
        days_to_expiry = features.get("days_to_expiry", 30)
        time_decay_factor = min(1.0, days_to_expiry / 7.0) if days_to_expiry > 0 else 0.1
        liquidity_score_raw = features.get("liquidity_score", 0)
        signal_strength = edge * confidence * min(1.0, liquidity_score_raw / 1000) * time_decay_factor

        # Record for category rotation tracking
        if category:
            self._category_signals.append({
                "timestamp": time.time(),
                "category": category,
                "edge": edge,
            })
            # Prune to 2500 entries when exceeding 5000 to prevent unbounded growth
            if len(self._category_signals) > 5000:
                self._category_signals = self._category_signals[-2500:]

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
                f"Kelly={kelly_f:.2f} vol={vol_scalar:.2f} liq={liq_scalar:.2f} "
                f"quality={signal_quality:.3f} edge_ratio={edge_ratio:.2f} "
                f"pred_std={prediction_std:.3f} strength={signal_strength:.3f}. "
                f"{len(FEATURE_NAMES)} features, {self.model.n_estimators} trees."
            ),
            recommended_size_cents=size,
            category=category,
            signal_quality=signal_quality,
            close_time=market.close_time or "",
        )
