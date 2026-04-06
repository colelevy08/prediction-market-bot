"""
Risk management module enforcing trading limits and position sizing.

----------------------------------------------------------------------
WHAT IS RISK MANAGEMENT?
----------------------------------------------------------------------
In trading, "risk management" means setting rules that protect your
account from catastrophic losses. Without these rules, a bot could:
  - Bet too much on a single trade and wipe out the account
  - Keep trading during a losing streak, making losses worse
  - Put all money into one type of market (lack of diversification)
  - Place orders when there isn't enough cash to cover them

Think of this module as the responsible adult in the room — it checks
every idea (called a "signal") before any money is spent.

----------------------------------------------------------------------
THE KELLY CRITERION — WHAT IS IT AND WHY DOES IT WORK?
----------------------------------------------------------------------
The Kelly Criterion is a mathematical formula that tells you the IDEAL
fraction of your bankroll to bet on each trade, given the odds.

The formula is:
    f* = (b * p - q) / b

Where:
  - f* = the fraction of bankroll to bet (e.g., 0.10 = bet 10%)
  - b  = the "net odds" — how much you WIN per $1 risked
            Example: if you buy a YES contract at 40 cents and it pays
            $1.00 if correct, you win 60 cents on a 40-cent bet, so
            b = 0.60 / 0.40 = 1.5
  - p  = your estimated probability of WINNING
  - q  = your estimated probability of LOSING = 1 - p

Example: A coin flip that pays 3-to-1 (b=3), but you know it's
actually a 60% chance heads (p=0.60, q=0.40):
    f* = (3 * 0.60 - 0.40) / 3 = (1.80 - 0.40) / 3 = 0.467
So Kelly says bet 46.7% of your bankroll. Over time, this maximizes
the long-run growth rate of your account.

WHY DOES IT WORK? Because it perfectly balances:
  - Betting MORE when you have a bigger edge (you'll grow faster)
  - Betting LESS when your edge is small (you'll preserve capital)
  - Never betting 100% (you can never go broke if you size correctly)

FRACTIONAL KELLY: Most practitioners use "half-Kelly" (kelly_fraction=0.5)
because real-world probability estimates are never perfect. Half-Kelly
cuts the bet size in half, sacrificing some growth rate in exchange for
much less volatility. This is the conservative, real-money approach.

----------------------------------------------------------------------
DRAWDOWN — WHAT IS IT?
----------------------------------------------------------------------
"Drawdown" is the percentage drop from your account's highest point
(called the "peak" or "high-water mark") to the current value.

Example:
  Account grows to $1,000 (peak), then drops to $800.
  Drawdown = ($1,000 - $800) / $1,000 = 20%

The risk manager shrinks bet sizes during drawdowns because:
  1. Statistically, big drawdowns are often followed by more losses
  2. Smaller bets mean you need fewer wins to recover
  3. It's psychological protection against "revenge trading"

----------------------------------------------------------------------
HOW THIS MODULE CONNECTS TO THE REST OF THE BOT
----------------------------------------------------------------------
This module sits between signal generation and order execution as a
safety gate. Every trade signal must pass through
RiskManager.check_signal() before an order is placed. The risk checks
include:

  1. Daily loss limit:    Dynamic — 5% of peak equity or configured max.
  2. Max open positions:  Prevents over-concentration (default 10).
  3. Category limits:     Max 3 positions per market category (correlation control).
  4. Duplicate positions: Rejects signals for markets where a position already exists.
  5. Minimum edge:        Filters low-edge signals below min_edge_threshold (default 8%).
  6. Balance check:       Ensures sufficient account balance.
  7. Bet size cap:        Enforces max_bet_amount_cents per trade.

Position sizing uses the Kelly Criterion with volatility + liquidity + drawdown scaling:
  - Full Kelly: f* = (b*p - q) / b
  - Fractional Kelly (default half-Kelly via kelly_fraction=0.5)
  - Volatility scaling: smaller bets in volatile markets
  - Liquidity scaling: smaller bets in illiquid markets
  - Drawdown scaling: reduces size during equity drawdowns

Connects to: bot.config (all threshold/limit values), bot.models (signal/order types).
Used by: bot.server (signal validation, order building).
"""

from __future__ import annotations

import math
from collections import Counter

import logging

from bot.config import config
from bot.models import OrderRequest, PortfolioSummary, TradingSignal

logger = logging.getLogger("predictionbot")


class RiskManager:
    """Pre-trade risk gate that validates signals and sizes positions.

    Every trading signal must pass through check_signal() before an order is placed.
    The risk manager maintains daily counters and equity tracking for drawdown control.

    Think of RiskManager as the bouncer at a nightclub — it checks every trade
    signal at the door and only lets it through if it passes ALL the rules. Even
    a great signal gets rejected if the account is already down too much today,
    or if too many positions are already open.

    The class tracks:
      - daily_pnl_cents:       How much money we've made or lost today (in cents).
                               When this hits the daily loss limit, trading stops
                               for the rest of the day.
      - peak_equity_cents:     The highest the account balance has ever been.
                               Used to calculate drawdown.
      - current_equity_cents:  The current account balance. If this is less than
                               peak_equity_cents, we're in a drawdown.
      - consecutive_losses:    How many trades in a row have been losers.
                               Used to shrink bet sizes during a losing streak
                               (this is called "anti-tilt" — preventing emotional
                               over-betting to "get back" losses).
    """

    def __init__(self):
        """Initialize with zeroed counters and equity tracking.

        All counters start at zero. Call set_equity() on startup to initialize
        the equity tracking with the real account balance.
        """
        # How much money we've made or lost so far today (can be negative)
        self.daily_pnl_cents: int = 0
        # Total number of trades executed today
        self.trades_today: int = 0
        # The highest the account has ever been (high-water mark)
        self.peak_equity_cents: int = 0
        # Where the account is right now
        self.current_equity_cents: int = 0
        # How many trades in a row have lost money (used to shrink bet sizes)
        self.consecutive_losses: int = 0

        # ── Task 25: Position Scaling Tracking ────────────────────────────────
        # Track how many times we've scaled into each ticker (per-instance, not class-level)
        self._scale_counts: dict[str, int] = {}

        # Fix 70: Validate that max_open_positions > 0
        if config.max_open_positions <= 0:
            logger.error(f"max_open_positions must be > 0, got {config.max_open_positions}. Defaulting to 10.")

        # Fix 56: Log initial risk parameters
        logger.info(
            f"RiskManager initialized: max_daily_loss={config.max_daily_loss_cents}c, "
            f"max_open_positions={config.max_open_positions}, "
            f"max_positions_per_category={config.max_positions_per_category}, "
            f"min_edge={config.min_edge_threshold:.1%}, "
            f"kelly_fraction={config.kelly_fraction}"
        )

    def check_signal(
        self, signal: TradingSignal, portfolio: PortfolioSummary
    ) -> tuple[bool, str]:
        """Validate a trading signal against all risk rules.

        This is the main "gatekeeper" function. It runs every proposed trade
        through a checklist of safety rules. If ANY rule fails, the trade is
        rejected with an explanation of why.

        The checks happen in a specific order — most critical first:
          1. Daily loss limit (are we already too far down today?)
          2. Dynamic position limits (do we have too many open trades?)
          3. Category exposure cap (are we too concentrated in one type of market?)
          4. Duplicate positions (do we already have money in this market?)
          5. Minimum edge (is the expected profit margin big enough to be worth it?)
          6. Sufficient balance (do we literally have enough cash?)
          7. Bet size cap (is the proposed bet size within our max?)

        Args:
            signal: The TradingSignal to validate. A TradingSignal contains the
                    market ticker, which side to bet (YES or NO), the bot's
                    estimated probability, and how much to bet.
            portfolio: Current portfolio state — includes current cash balance
                       and a list of all open positions.

        Returns:
            A tuple: (True, "OK") if the trade is allowed,
                     (False, "reason") if the trade is rejected.
            The reason string is logged and can be shown in the dashboard.
        """
        # ── Check 1: Daily Loss Limit ──────────────────────────────────────
        # "daily_pnl_cents" accumulates all wins and losses today. If it's
        # too negative, trading stops for the rest of the day.
        #
        # The limit is DYNAMIC: it scales with the account size. Larger
        # accounts get a larger absolute loss limit, but the same percentage cap.
        # We take whichever limit is SMALLER (more conservative).
        #
        # Why have a daily loss limit?
        # Without one, a bot can keep trading through a bad day, turning a
        # manageable loss into a catastrophic one. Stopping early is the
        # single most important risk control for automated trading.
        dynamic_limit = max(500, int(self.peak_equity_cents * config.max_drawdown_pct)) if self.peak_equity_cents > 0 else config.max_daily_loss_cents
        effective_limit = min(dynamic_limit, config.max_daily_loss_cents)
        if self.daily_pnl_cents <= -effective_limit:
            reason = f"Daily loss limit reached (${effective_limit / 100:.2f})"
            logger.debug(f"Signal check DENIED for {signal.ticker}: {reason}")
            return False, reason

        # ── Check 2: Dynamic Position Limits ──────────────────────────────
        # Limits how many open trades we can have at once. "Open" means we've
        # bought a contract but haven't yet received the outcome.
        #
        # Having too many positions at once is risky because:
        #   - Each position ties up cash
        #   - Correlated positions (markets that move together) can all lose at once
        #   - It becomes hard to monitor and manage many positions
        #
        # The limit is DYNAMIC: during a drawdown, we reduce it further.
        # During a winning streak, we allow slightly more.
        max_positions = self.get_dynamic_position_limit()
        if len(portfolio.positions) >= max_positions:
            return False, f"Max open positions ({max_positions}, dynamic) reached"

        # ── Check 3: Category Exposure Cap ────────────────────────────────
        # Markets are grouped into categories (politics, crypto, sports, etc.)
        # This check prevents putting more than 40% of all open positions in
        # one single category — a form of DIVERSIFICATION.
        #
        # Why diversify? If all your bets are in one category (e.g., all crypto
        # markets), a single piece of news can affect all your positions at once.
        # Spreading across categories means losses in one category don't
        # automatically cause losses in others.
        #
        # Task 23: Sector rotation — category exposure capped at 40%
        cat_check = self.category_exposure_check(signal, portfolio)
        if not cat_check[0]:
            return cat_check

        # Category concentration limit (original per-category count limit)
        if signal.category:
            category_counts = Counter(
                p.category for p in portfolio.positions if p.category
            )
            if category_counts.get(signal.category, 0) >= config.max_positions_per_category:
                return False, f"Max positions in '{signal.category}' ({config.max_positions_per_category}) reached"

        # ── Check 4: Duplicate Positions ──────────────────────────────────
        # If we already own contracts in this exact market, don't buy more
        # (unless scaling in is specifically approved — see scale_position_check).
        #
        # Owning the same market twice doesn't add diversification, it just
        # doubles concentration in a single outcome.
        #
        # EXCEPTION: "Scaling in" means intentionally adding to an existing
        # position when the bot becomes even MORE confident in the trade.
        # This is allowed up to 3 times total (configurable).
        #
        # Task 25: Allow scaling if confidence is increasing (max 3x)
        for pos in portfolio.positions:
            if pos.ticker == signal.ticker:
                scale_result = self.scale_position_check(signal, pos)
                if scale_result[0]:
                    return True, scale_result[1]
                return False, f"Already have position in {signal.ticker}"

        # ── Check 5: Minimum Edge ──────────────────────────────────────────
        # "Edge" is the difference between what the bot thinks the true
        # probability is and what the market is currently pricing.
        #
        # Example: Market says 40% chance of YES. Bot calculates 55% chance.
        # Edge = 55% - 40% = 15%
        #
        # Why require a minimum edge?
        # Every trade has transaction costs (spread, fees). A tiny edge might
        # look profitable in theory but gets eaten up by those costs. We need
        # a meaningful edge to make trading worthwhile.
        #
        # The default minimum is 8% — any trade where the bot's advantage
        # over the market price is less than 8% is rejected as "not worth it."
        if signal.edge < config.min_edge_threshold:
            return False, f"Edge {signal.edge:.1%} below threshold {config.min_edge_threshold:.1%}"

        # ── Check 6: Sufficient Balance ────────────────────────────────────
        # The simplest check: do we have enough cash to place this trade?
        # "cost" is the recommended bet size in cents (e.g., 500 = $5.00).
        cost = signal.recommended_size_cents
        if cost > portfolio.balance_cents:
            return False, f"Insufficient balance: need {cost}c, have {portfolio.balance_cents}c"

        # ── Check 7: Bet Size Cap ──────────────────────────────────────────
        # Even if Kelly says to bet a lot, we enforce a hard maximum per trade.
        # This prevents any single trade from being catastrophically large.
        # Also rejects "zero-size" bets, which would be a no-op.
        if signal.recommended_size_cents > config.max_bet_amount_cents:
            return False, f"Size {signal.recommended_size_cents}c exceeds max {config.max_bet_amount_cents}c"

        if signal.recommended_size_cents <= 0:
            return False, "Recommended size is zero"

        # Fix 57: Log signal check result
        logger.debug(f"Signal check ALLOWED: {signal.ticker} edge={signal.edge:.1%} size={signal.recommended_size_cents}c")
        return True, "OK"

    def get_drawdown_scalar(self) -> float:
        """Compute position size multiplier based on current drawdown.

        This function answers: "Given how far we are from our peak, how
        much should we scale down our bets?"

        The logic is a LINEAR SCALE DOWN:
          0% drawdown   -> 1.0x (full size — we're at or above all-time high)
          20% drawdown  -> 0.5x (half size — we're 20% below our peak)
          40%+ drawdown -> 0.25x (quarter size — severe drawdown, be very cautious)

        WHY SCALE DOWN DURING DRAWDOWNS?
        When the account is losing, there are two possibilities:
          1. The strategy is broken / market conditions changed (bad luck)
          2. We're just in a temporary rough patch (normal variance)
        In either case, smaller bets protect us. If it's #1, we lose less.
        If it's #2, we recover eventually without having dug a bigger hole.

        CONSECUTIVE LOSS PENALTY (anti-tilt):
        "Tilt" is a poker term for when a losing player starts making
        irrational, larger bets to try to win back losses — which usually
        makes things much worse. This penalty does the OPPOSITE: 3+ losses
        in a row reduces bets further, forcing patience.

        Returns:
            Float multiplier in [0.25, 1.0] to apply to position sizes.
            Always at least 0.25 — never less than quarter-sized bets.
        """
        if self.peak_equity_cents <= 0:
            # No equity history yet — use full size (first-time startup)
            dd_scalar = 1.0
        else:
            # dd_pct = how far below the peak we are, as a fraction
            # Example: peak=$1000, current=$800 → dd_pct = 0.20 (20%)
            dd_pct = max(0, (self.peak_equity_cents - self.current_equity_cents) / self.peak_equity_cents)
            if dd_pct <= 0:
                dd_scalar = 1.0
            elif dd_pct < 0.20:
                dd_scalar = 1.0 - dd_pct * 2.5  # 1.0 -> 0.5 over 0-20% DD
            else:
                dd_scalar = max(0.25, 0.5 - (dd_pct - 0.20) * 1.25)  # 0.5 -> 0.25 over 20-40% DD

        # Consecutive loss penalty (anti-tilt)
        # After 3 losses in a row, start shrinking bets by 25% per additional loss.
        # Loss 3: 0.75x, Loss 4: 0.50x, Loss 5: 0.25x (minimum floor)
        if self.consecutive_losses >= 3:
            loss_scalar = max(0.25, 1.0 - (self.consecutive_losses - 2) * 0.25)
            dd_scalar *= loss_scalar

        return max(0.25, dd_scalar)

    def get_anti_correlation_scalar(
        self, signal: TradingSignal, existing_positions: list | None = None,
    ) -> float:
        """Compute anti-correlation position size multiplier for mean reversion.

        "Mean reversion" is the idea that prices tend to drift back toward
        their average over time. If we're losing in a category, a new strong
        signal in that SAME category might be the market correcting back —
        so we can bet slightly more (up to 20% bigger).

        This only applies when:
          1. There's an existing LOSING position in the same category
          2. The new signal has very high confidence (> 85%)

        Think of it as: "We've been wrong about sports markets lately, but
        THIS new sports market signal is extremely confident — let's bet
        slightly more than usual because the market may be overcorrecting."

        Returns:
            Multiplier in [1.0, 1.2]. Returns 1.0 (no change) if the boost
            doesn't apply. Maximum 1.2 means at most 20% bigger than normal.
        """
        if not signal.category or signal.confidence <= 0.85 or not existing_positions:
            return 1.0

        has_losing_same_category = False
        for pos in existing_positions:
            if getattr(pos, 'category', '') == signal.category:
                unrealized = getattr(pos, 'unrealized_pnl_cents', None)
                if unrealized is not None and unrealized < 0:
                    has_losing_same_category = True
                    break
                if self.consecutive_losses > 0:
                    has_losing_same_category = True
                    break

        if has_losing_same_category:
            confidence_excess = signal.confidence - 0.85
            boost = min(0.20, confidence_excess * 2.0)
            return 1.0 + boost

        return 1.0

    def kelly_size(
        self, signal: TradingSignal, bankroll_cents: int,
        volatility: float = 0.05, spread_pct: float = 0.10,
        existing_positions: list | None = None,
    ) -> int:
        """Calculate position size using Kelly Criterion with vol/liquidity/drawdown/correlation scaling.

        This is the core "how much to bet" function. It starts with the pure
        Kelly formula and then applies several real-world adjustment factors
        that shrink the bet when conditions are less favorable.

        THE KELLY FORMULA STEP BY STEP:
          Full Kelly: f* = (b * p - q) / b

          Where b = (1 - cost) / cost
            "cost" = the price you pay for the contract (as a fraction of $1)
            Example: YES contract at 40 cents → cost = 0.40
                     b = (1 - 0.40) / 0.40 = 1.5
                     Means: for every $1 risked, you win $1.50 if correct

          p = your probability of winning this bet
          q = 1 - p (probability of losing)

          If Kelly comes out negative, the edge doesn't justify a bet → size = 0

        THE ADJUSTMENT MULTIPLIERS (all between 0 and 1, applied in sequence):

          1. Fractional Kelly (kelly_fraction, default 0.5 = "half-Kelly"):
             Cuts the raw Kelly fraction in half. This is standard practice
             because our probability estimates are never perfect — being
             conservative here prevents over-betting on uncertain estimates.

          2. Correlation adjustment (0.50 to 1.0):
             If we already hold 2 or 3 positions in the same category, those
             bets are "correlated" — they'll likely all win or lose together.
             We reduce new bets in that category to avoid over-exposure.

          3. Volatility scaling (vol_scalar):
             "Volatility" means how much the market price is moving around.
             High volatility = high uncertainty = smaller bets to compensate.
             Baseline is 5% volatility. If volatility is 10%, vol_scalar = 0.5
             (half size).

          4. Liquidity scaling (liq_scalar):
             "Liquidity" means how easy it is to buy/sell without moving the
             price. The bid-ask spread (difference between buy and sell price)
             is the main indicator. Wide spread = illiquid market = smaller bets
             because you lose more to transaction costs.
             Baseline spread = 10%. Wider spread → smaller bet.

          5. Drawdown scaling (dd_scalar):
             From get_drawdown_scalar() — shrinks bets during account drawdowns.

          6. Anti-correlation boost (anti_corr_scalar):
             From get_anti_correlation_scalar() — slightly larger bets for
             very confident signals in categories where we've been losing.

        TRANSACTION COST DEDUCTION:
          After computing the Kelly size, we subtract estimated transaction
          costs (slippage + commission). "Slippage" is the difference between
          the price you expected and the price you actually got — it's a
          hidden cost that every real trader faces.

        DYNAMIC CAP:
          The final size is capped at 15% of the current balance (or the
          configured max, whichever is higher). This prevents any single
          trade from being disproportionately large even if Kelly says so.

        Args:
            signal: TradingSignal with fair_probability and market_probability.
            bankroll_cents: Current available balance in cents.
            volatility: Historical price volatility (0-1). Default 0.05.
            spread_pct: Bid-ask spread as fraction. Default 0.10.
            existing_positions: List of current positions (with .category attribute) for
                correlation-adjusted sizing.

        Returns:
            Position size in cents (e.g., 500 = $5.00).
        """
        # p = our estimated "true" probability of this market resolving correctly
        p = signal.fair_probability
        # market_price = what the market is charging for this contract (as a 0-1 fraction)
        market_price = signal.market_probability

        if market_price <= 0 or market_price >= 1 or p <= 0:
            return signal.recommended_size_cents

        # Cost basis depends on side
        # For YES: you pay the YES ask price
        # For NO: you pay (1 - YES ask price) because NO is the complement of YES
        market_cost = market_price if signal.side.value == "yes" else (1 - market_price)

        # Guard: market_cost must be in valid range for Kelly formula
        if market_cost < 0.01 or market_cost > 0.99:
            return signal.recommended_size_cents

        # b = net odds: how much you WIN per $1 of cost
        # Example: cost=0.40 → b = 0.60/0.40 = 1.5 (win $1.50 for every $1 risked)
        b = (1.0 - market_cost) / market_cost

        # Guard: payout ratio must be positive for division
        if b <= 0:
            return 0

        # Win probability for the chosen side
        # For YES bets: p is directly the probability of YES resolving
        # For NO bets: we flip it (1-p) because we need the probability of NO resolving
        win_prob = p if signal.side.value == "yes" else (1 - p)
        q = 1.0 - win_prob  # Loss probability

        # Full Kelly formula: f* = (b*p - q) / b
        # max(0, ...) ensures we never get a negative bet size
        kelly_f = max(0, (b * win_prob - q) / b)

        # Apply fractional Kelly (default: half-Kelly)
        # This is the most important single adjustment — it's the difference
        # between aggressive Kelly betting and responsible Kelly betting
        kelly_f *= config.kelly_fraction

        # Correlation-adjusted sizing: reduce when holding correlated positions
        # "Correlated" = markets in the same category tend to move together
        # Having 3 crypto markets open means a bad day for crypto hurts all 3 at once
        if existing_positions and signal.category:
            same_category_count = sum(
                1 for pos in existing_positions
                if getattr(pos, 'category', '') == signal.category
            )
            if same_category_count >= 3:
                kelly_f *= 0.50  # 50% reduction for 3+ correlated positions
            elif same_category_count >= 2:
                kelly_f *= 0.70  # 30% reduction for 2 correlated positions

        # Volatility scaling: baseline vol=0.05, scale down if higher
        # "vol" here is the standard deviation of recent price movements
        # Higher volatility = more uncertain market = smaller bet
        vol = max(volatility, 0.01)
        vol_scalar = min(1.0, 0.05 / vol)

        # Liquidity scaling: baseline spread=10%, scale down if wider
        # A wide spread means it costs more to enter and exit the trade
        spr = max(spread_pct, 0.01)
        liq_scalar = min(1.0, 0.10 / spr)

        # Drawdown scaling — from get_drawdown_scalar() above
        dd_scalar = self.get_drawdown_scalar()

        # Anti-correlation boost for mean reversion within categories
        anti_corr_scalar = self.get_anti_correlation_scalar(signal, existing_positions)

        # Multiply all adjustments together to get the final Kelly fraction
        # Then multiply by bankroll to get the dollar amount to bet
        adjusted_kelly = kelly_f * vol_scalar * liq_scalar * dd_scalar * anti_corr_scalar
        size = int(adjusted_kelly * bankroll_cents)

        # Fix 59: Subtract estimated transaction cost from Kelly size
        # "Slippage" = the difference between expected and actual execution price
        # This is a real cost that must be accounted for before trading
        transaction_cost = config.slippage_cents + config.commission_cents
        size = max(0, size - transaction_cost)

        # Dynamic cap: 15% of current balance (scales with account), floored at max_bet_amount_cents
        # This prevents any single trade from being disproportionately large
        dynamic_cap = max(config.max_bet_amount_cents, int(bankroll_cents * config.max_bet_pct))
        final_size = max(0, min(size, dynamic_cap))

        # Fix 58: Log Kelly calculation details for debugging
        logger.debug(
            f"Kelly size for {signal.ticker}: kelly_f={kelly_f:.4f} vol_scalar={vol_scalar:.2f} "
            f"liq_scalar={liq_scalar:.2f} dd_scalar={dd_scalar:.2f} anti_corr={anti_corr_scalar:.2f} "
            f"adjusted={adjusted_kelly:.4f} raw_size={size}c final_size={final_size}c"
        )

        return final_size

    def build_order(self, signal: TradingSignal, bankroll_cents: int = 0) -> OrderRequest:
        """Convert a validated TradingSignal into a Kalshi OrderRequest.

        Kalshi orders are denominated in "contracts" — each contract pays $1
        if the market resolves in your favor, and $0 if it doesn't. The price
        is quoted in cents (1–99), representing the probability as a percentage.

        Example: Buying 5 YES contracts at 40 cents each costs $2.00 total.
                 If the market resolves YES, you receive $5.00 (profit: $3.00).
                 If the market resolves NO, you receive $0 (loss: $2.00).

        "count" = how many contracts to buy, calculated as:
            count = total_dollars_to_spend / price_per_contract

        Args:
            signal: Validated TradingSignal (already passed check_signal).
            bankroll_cents: If > 0, recalculates size using Kelly; otherwise
                           uses the size pre-computed in the signal.

        Returns:
            OrderRequest ready for KalshiClient.place_order().
        """
        # Convert probability (0-1) to Kalshi price (1-99 cents)
        # For YES: price is the YES probability × 100
        # For NO:  price is the NO probability × 100 = (1 - YES probability) × 100
        price = int(signal.market_probability * 100) if signal.side.value == "yes" else int((1 - signal.market_probability) * 100)
        price = max(1, min(99, price))  # Clamp to valid Kalshi price range

        if bankroll_cents > 0:
            size = self.kelly_size(signal, bankroll_cents)
        else:
            size = signal.recommended_size_cents

        # Calculate how many contracts we can buy with the given dollar amount
        # Integer division: e.g., 200 cents ÷ 40 cents/contract = 5 contracts
        count = max(1, size // max(price, 1))

        return OrderRequest(
            ticker=signal.ticker,
            side=signal.side,
            price_cents=price,
            count=count,
        )

    def record_trade(self, pnl_cents: int = 0):
        """Update counters after a trade. Tracks daily P&L, equity, and consecutive losses.

        Called after every trade completes (win or loss). Updates:
          - daily_pnl_cents: running today's total profit/loss
          - current_equity_cents: running account balance estimate
          - peak_equity_cents: the all-time high (for drawdown calculation)
          - consecutive_losses: resets to 0 on a win, increments on a loss

        Args:
            pnl_cents: Profit or loss from the trade in cents. Positive = profit,
                       negative = loss. Example: +250 means you made $2.50.
        """
        # Fix 64: Validate pnl_cents is a number
        if not isinstance(pnl_cents, (int, float)):
            logger.warning(f"record_trade: pnl_cents is not a number ({type(pnl_cents)}), defaulting to 0")
            pnl_cents = 0
        pnl_cents = int(pnl_cents)
        self.trades_today += 1
        self.daily_pnl_cents += pnl_cents
        self.current_equity_cents += pnl_cents
        self.peak_equity_cents = max(self.peak_equity_cents, self.current_equity_cents)
        if pnl_cents < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_daily(self):
        """Reset daily P&L and trade count. Call at the start of each trading day."""
        # Fix 65: Log the reset with previous values
        logger.info(
            f"Daily reset: previous daily_pnl={self.daily_pnl_cents}c, "
            f"trades_today={self.trades_today}, consecutive_losses={self.consecutive_losses}"
        )
        self.daily_pnl_cents = 0
        self.trades_today = 0
        self.consecutive_losses = 0

    def set_equity(self, equity_cents: int):
        """Initialize equity tracking (call on startup with current balance)."""
        self.current_equity_cents = equity_cents
        self.peak_equity_cents = max(self.peak_equity_cents, equity_cents)

    def get_portfolio_risk(self, portfolio: PortfolioSummary) -> dict:
        """Calculate portfolio risk metrics for the dashboard.

        Returns several risk measures that help monitor how concentrated and
        risky the portfolio is at a given moment:

        TOTAL EXPOSURE: The total dollars currently "at risk" in open positions.
          If every open position resolved against us simultaneously, this is
          roughly the maximum we could lose.

        CONCENTRATION RISK (HHI — Herfindahl-Hirschman Index):
          A standard economics measure of market concentration, adapted here
          to measure how concentrated the portfolio is by category.
          - HHI = sum of (category_share²) for each category
          - HHI near 0 = perfectly diversified (many categories, each small)
          - HHI near 1 = completely concentrated (all in one category)
          - HHI > 0.25 is considered "highly concentrated" and triggers
            larger risk adjustments in the VaR estimate.

        VAR (Value at Risk):
          An estimate of the maximum loss expected with 95% probability.
          The "1.65" multiplier comes from statistics: in a normal distribution,
          95% of outcomes fall within 1.65 standard deviations of the mean.
          In plain English: "There's a 5% chance we could lose this much or more."

          The correlation adjustment amplifies the VaR when positions are
          concentrated — because concentrated positions are more likely to
          all lose at the same time.

        Returns: total_exposure, concentration risk (HHI), max single position %,
        category breakdown, and correlation-adjusted VaR estimate.
        """
        positions = portfolio.positions
        if not positions:
            return {
                "total_exposure_cents": 0,
                "concentration_risk": 0.0,
                "max_single_position_pct": 0.0,
                "category_breakdown": {},
                "estimated_var_cents": 0,
                "position_count": 0,
                "total_portfolio_value_cents": portfolio.balance_cents,
            }

        # Total exposure: sum of position costs
        # avg_price_cents × quantity = total cost of each position
        total_exposure = sum(p.avg_price_cents * p.quantity for p in positions)

        # Concentration risk: Herfindahl-Hirschman Index across categories
        # First, calculate how much money is in each category
        category_exposure: dict[str, int] = {}
        for p in positions:
            cat = p.category or "uncategorized"
            category_exposure[cat] = category_exposure.get(cat, 0) + p.avg_price_cents * p.quantity

        if total_exposure > 0:
            # "shares" = each category's fraction of total exposure
            # e.g., if 70% is in crypto and 30% in politics: shares = [0.7, 0.3]
            # HHI = 0.7² + 0.3² = 0.49 + 0.09 = 0.58 (fairly concentrated)
            shares = [v / total_exposure for v in category_exposure.values()]
            hhi = sum(s ** 2 for s in shares)
        else:
            hhi = 0.0

        # Max single position as % of portfolio
        # If one position is 50% of the portfolio, that's a huge concentration risk
        max_position_cost = max(
            (p.avg_price_cents * p.quantity for p in positions), default=0
        )
        total_value = portfolio.balance_cents + total_exposure
        max_single_pct = max_position_cost / max(total_value, 1)

        # Correlation-adjusted VaR estimate
        # First, compute the standard deviation of position sizes
        # (how much the sizes vary from each other)
        position_costs = [p.avg_price_cents * p.quantity for p in positions]
        if len(position_costs) >= 2:
            mean_cost = sum(position_costs) / len(position_costs)
            variance = sum((c - mean_cost) ** 2 for c in position_costs) / (len(position_costs) - 1)
            std_cost = math.sqrt(variance)
            # Correlation adjustment: HHI > 0.25 = concentrated, amplify risk
            # Concentrated positions are more likely to all lose together
            corr_factor = 1.0 + max(0, hhi - 0.25) * 2
            # 1.65 = 95th percentile z-score (standard statistics)
            # sqrt(n) accounts for having multiple positions
            var_estimate = int(1.65 * std_cost * math.sqrt(len(position_costs)) * corr_factor)
        else:
            var_estimate = total_exposure

        return {
            "total_exposure_cents": total_exposure,
            "concentration_risk": round(hhi, 4),
            "max_single_position_pct": round(max_single_pct, 4),
            "category_breakdown": {
                cat: {"exposure_cents": exp, "pct": round(exp / max(total_exposure, 1), 4)}
                for cat, exp in category_exposure.items()
            },
            "estimated_var_cents": min(var_estimate, total_exposure),
            "position_count": len(positions),
            "total_portfolio_value_cents": total_value,
        }

    # ── Task 22: Dynamic Position Limits ──────────────────────────────────

    def get_dynamic_position_limit(self) -> int:
        """Dynamically adjust max open positions based on current performance.

        Instead of a fixed limit like "always max 10 positions," this function
        adapts the limit based on how the account is performing RIGHT NOW.

        The logic:
          - Bad times  (drawdown > 20%): Allow 30% FEWER positions.
                       When losing, concentrate on fewer, higher-conviction trades.
          - Good times (winning streak + profitable day): Allow 20% MORE positions.
                       When performing well, can safely take on more opportunities.
          - Normal times: Use the configured default (max_open_positions).

        This is an example of a "regime-aware" trading strategy — it changes
        behavior based on current conditions rather than always acting the same.

        Returns:
            Adjusted max position count (integer).
        """
        base_limit = config.max_open_positions

        # Fix 60: Handle case where no trades exist (return default)
        if self.trades_today == 0 and self.peak_equity_cents == 0:
            return base_limit

        # Check for drawdown
        if self.peak_equity_cents > 0:
            dd_pct = (self.peak_equity_cents - self.current_equity_cents) / self.peak_equity_cents
            if dd_pct > 0.20:
                # Reduce by 30% during significant drawdowns
                reduced = int(base_limit * 0.70)
                return max(1, reduced)

        # Check for winning streak (consecutive_losses == 0 means we're winning)
        # We track consecutive losses, so 0 losses after 5+ trades suggests a streak
        # Use a simple heuristic: if daily_pnl is positive and no consecutive losses
        if self.consecutive_losses == 0 and self.daily_pnl_cents > 0 and self.trades_today >= 5:
            expanded = int(base_limit * 1.20)
            return min(15, expanded)

        return base_limit

    # ── Task 23: Sector Rotation Strategy (Category Exposure Check) ───────

    def category_exposure_check(
        self, signal: TradingSignal, portfolio: PortfolioSummary,
    ) -> tuple[bool, str]:
        """Enforce configurable category exposure cap to force diversification.

        If a category already has >= max_category_exposure_pct of total open positions,
        skip new entries in that category. This prevents over-concentration in any
        single market sector. Fix 68: Cap is configurable via config.

        Args:
            signal: The TradingSignal to validate.
            portfolio: Current portfolio state.

        Returns:
            Tuple of (allowed: bool, reason: str).
        """
        if not signal.category or not portfolio.positions:
            return True, "OK"

        total_positions = len(portfolio.positions)
        if total_positions == 0:
            return True, "OK"

        category_count = sum(
            1 for p in portfolio.positions
            if (p.category or "") == signal.category
        )

        exposure_pct = category_count / total_positions
        # Fix 61: Log category exposure levels
        logger.debug(f"Category '{signal.category}' exposure: {exposure_pct:.0%} ({category_count}/{total_positions})")
        if exposure_pct >= config.max_category_exposure_pct:
            return False, (
                f"Category '{signal.category}' has {exposure_pct:.0%} exposure "
                f"({category_count}/{total_positions} positions), exceeds {config.max_category_exposure_pct:.0%} limit"
            )

        return True, "OK"

    # ── Task 25: Position Scaling ─────────────────────────────────────────

    def scale_position_check(
        self, signal: TradingSignal, existing_position,
    ) -> tuple[bool, str]:
        """Check if we should scale into an existing position.

        "Scaling in" means adding more contracts to a position you already have.
        Professional traders do this when their conviction increases — if the
        evidence gets stronger, buy more.

        RULES for scaling approval (ALL must be true):
          1. Haven't already scaled 3 times (max scale = configurable, default 3)
          2. New signal is for the SAME side (can't flip direction)
          3. New signal has HIGHER confidence than the original entry
             (only scale up if the bot becomes MORE sure, not less)

        Example:
          Original entry: bought YES at 40 cents with 70% confidence
          New signal: bot now estimates 85% confidence → SCALE IN (buy more)
          New signal: bot now estimates 60% confidence → REJECT (lower conviction)

        Args:
            signal: New signal for the same ticker (same market, possibly higher confidence).
            existing_position: The existing position object we might add to.

        Returns:
            Tuple of (allowed: bool, reason: str).
        """
        ticker = signal.ticker
        current_scale = self._scale_counts.get(ticker, 1)

        # Fix 67: Max scaling configurable via config.max_position_scale
        if current_scale >= config.max_position_scale:
            return False, f"Max scale count ({config.max_position_scale}) reached for {ticker}"

        # Must be same side
        existing_side = getattr(existing_position, 'side', '')
        if existing_side and existing_side != signal.side.value:
            return False, f"Cannot scale: signal side ({signal.side.value}) != position side ({existing_side})"

        # Require increasing confidence
        existing_confidence = getattr(existing_position, 'model_prob', None)
        if existing_confidence is None:
            existing_confidence = getattr(existing_position, 'confidence', 0.5)
        # New signal must have higher confidence to justify scaling in
        if signal.confidence <= existing_confidence:
            return False, (
                f"Cannot scale: new confidence ({signal.confidence:.2f}) not higher "
                f"than existing ({existing_confidence:.2f})"
            )

        # Allow scaling
        self._scale_counts[ticker] = current_scale + 1
        msg = (
            f"Scale-in approved for {ticker}: scale {current_scale + 1}/{config.max_position_scale}, "
            f"confidence {existing_confidence:.2f} -> {signal.confidence:.2f}"
        )
        # Fix 62: Log scaling decisions
        logger.info(msg)
        return True, msg

    def get_risk_summary(self, portfolio: PortfolioSummary | None = None) -> dict:
        """Fix 69: Get current risk summary (daily P&L, open positions, category exposure).

        Returns:
            Dict with daily_pnl_cents, trades_today, consecutive_losses, peak_equity,
            current_equity, drawdown_pct, dynamic_position_limit, and category_exposure.
        """
        dd_pct = 0.0
        if self.peak_equity_cents > 0:
            dd_pct = (self.peak_equity_cents - self.current_equity_cents) / self.peak_equity_cents

        summary = {
            "daily_pnl_cents": self.daily_pnl_cents,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "peak_equity_cents": self.peak_equity_cents,
            "current_equity_cents": self.current_equity_cents,
            "drawdown_pct": round(dd_pct, 4),
            "dynamic_position_limit": self.get_dynamic_position_limit(),
            "drawdown_scalar": round(self.get_drawdown_scalar(), 4),
        }

        if portfolio:
            category_counts: dict[str, int] = {}
            for p in portfolio.positions:
                cat = getattr(p, 'category', '') or 'uncategorized'
                category_counts[cat] = category_counts.get(cat, 0) + 1
            summary["open_positions"] = len(portfolio.positions)
            summary["category_exposure"] = category_counts

        return summary

    def reset_scale_count(self, ticker: str):
        """Reset scale tracking for a ticker when position is fully closed."""
        self._scale_counts.pop(ticker, None)
