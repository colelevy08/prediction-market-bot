"""
Performance tracking and trade analytics for the prediction market bot.

----------------------------------------------------------------------
WHY DO WE TRACK PERFORMANCE?
----------------------------------------------------------------------
Trading without performance tracking is like driving blindfolded. You
need numbers to answer questions like:
  - Is the strategy actually profitable, or just getting lucky?
  - Is the bot good at certain types of markets but bad at others?
  - Are we getting worse over time (strategy decay)?
  - How much risk are we taking to earn each dollar of profit?

This module records every completed trade and then computes the
"report card" for the strategy as a whole.

----------------------------------------------------------------------
KEY METRICS EXPLAINED FOR BEGINNERS
----------------------------------------------------------------------

P&L (Profit and Loss):
  The simplest measure. Total money made minus total money lost.
  Example: Won $50, lost $30, P&L = +$20.
  A positive P&L is good, but P&L alone doesn't tell you if you're
  taking smart risks or just getting lucky.

WIN RATE:
  The percentage of trades that made money.
    Win Rate = (number of winning trades) / (total trades)
  A 60% win rate means 6 out of every 10 trades are profitable.
  IMPORTANT: Win rate alone is not enough! A strategy can be profitable
  with a 40% win rate if the wins are much bigger than the losses.

PROFIT FACTOR:
  Gross profit (sum of all wins) divided by gross loss (sum of all losses).
  A profit factor of 2.0 means you earn $2 for every $1 lost.
  A profit factor above 1.0 means the strategy is overall profitable.
  Below 1.0 means it's losing money overall.

EXPECTED VALUE (EV) PER TRADE:
  On average, how much does each trade make?
    EV = (win_rate × avg_win) - (loss_rate × avg_loss)
  A positive EV means the strategy is mathematically profitable in the
  long run. This is the single most important metric for assessing
  whether a strategy "has edge."

SHARPE RATIO:
  The gold standard of risk-adjusted returns in finance.
  It measures: "How much return are you getting per unit of risk taken?"
    Sharpe = (average return - risk-free rate) / standard deviation of returns
  - "Risk-free rate" is what you'd earn doing nothing (e.g., a savings account).
    If you can earn 5% risk-free, any strategy must beat 5% to be worth the risk.
  - "Standard deviation" measures how volatile the returns are — how much
    they jump up and down.
  A Sharpe ratio of:
    < 1.0 = Bad — not worth the risk
    1–2   = Good — beating the market on a risk-adjusted basis
    > 2   = Excellent — exceptional strategy
  Multiply by sqrt(365) to annualize: daily Sharpe × 19.1 = annual Sharpe.

LOG RETURNS:
  Instead of simple percentage returns, we use logarithmic (log) returns:
    log_return = ln(exit_price / entry_price)
  Why? Log returns are "additive" — you can sum them across trades to get
  the total return over time. Simple percentage returns can't be summed
  this way. Also, log returns handle large losses without hitting -100%.

MAX DRAWDOWN:
  The largest peak-to-trough decline in the equity curve.
  Example: Account grows to $1,000 (peak), drops to $700 (trough).
  Max drawdown = $300 = 30%.
  This tells you the worst-case scenario the strategy has actually hit.
  Investors often care more about max drawdown than raw returns because
  it measures "how painful was the worst stretch?"

MAE (Max Adverse Excursion):
  During a trade's lifetime, what was the WORST the unrealized loss got?
  Example: You buy YES at 40 cents. The price drops to 25 cents before
  recovering to 80 cents. MAE = 40 - 25 = 15 cents per contract.
  Used to optimize where to set stop-losses: if your MAE on winning trades
  is typically 10 cents, you might set stops at 15 cents.

MFE (Max Favorable Excursion):
  During a trade's lifetime, what was the BEST the unrealized gain got?
  Example: Price hits 75 cents before settling back to 60 cents at exit.
  MFE = 75 - 40 = 35 cents per contract.
  Used to optimize take-profit targets: if MFE is usually 30 cents, you
  might set a take-profit limit there to capture gains before they erode.

EDGE REALIZATION RATIO:
  "The model predicted a 15% edge. Did we actually realize that edge?"
  Compares the predicted edge (model probability - market probability)
  against the actual log return achieved. A ratio of 1.0 = perfect
  realization. Less than 1.0 = some edge was "leaked" to costs or slippage.

----------------------------------------------------------------------
HOW THIS MODULE CONNECTS TO THE REST OF THE BOT
----------------------------------------------------------------------
This module records every completed trade (paper or live) and computes the
quantitative metrics used to evaluate strategy quality:

Key metrics computed:
  - Sharpe Ratio: SR = (mean(returns) - Rf) / std(returns)
    where Rf = 5% annual / 365 days. Labels: <1 Bad, 1-2 Good, >2 Excellent.
  - Log Returns: ln(P_exit / P_entry) for YES trades, ln(P_entry / P_exit) for NO.
    Log returns are additive over time and handle large price moves correctly.
  - MAE (Max Adverse Excursion): Deepest drawdown during a trade's lifetime.
    Used to optimize stop-loss placement.
  - MFE (Max Favorable Excursion): Best unrealized profit during a trade.
    Used to optimize take-profit targets.
  - Win Rate, Profit Factor, Max Drawdown, Best/Worst trade, Average Edge.

Data structures:
  - TradeRecord: Immutable record of a single completed trade with all fields.
  - PerformanceMetrics: Aggregated statistics computed from all TradeRecords.
  - PerformanceTracker: Stateful tracker that accumulates trades and computes metrics.

The tracker also maintains an equity curve (list of {time, equity_cents, trade_num})
for charting in the frontend, and supports per-category breakdowns and trade journal
notes that persist to Supabase.

Connects to: bot.database (optional Supabase persistence for trades and notes).
Used by: bot.backtester (Backtester and PaperTrader), bot.server (performance endpoints).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("predictionbot")


@dataclass
class TradeRecord:
    """Immutable record of a single completed trade (entry + exit).

    A "trade record" is a snapshot of everything about one completed trade —
    from the moment we bought the contract to the moment we received the
    outcome. Think of it as a receipt.

    All prices are in the 0-1 range (not cents, not percentages).
    Example: 0.45 means 45 cents per contract.

    P&L and log return are COMPUTED fields — they're calculated automatically
    by PerformanceTracker.record_trade() when the record is created.

    The "@dataclass" decorator means Python auto-generates the __init__
    function from the field list below, saving a lot of boilerplate code.
    """
    # The unique identifier for this market on Kalshi (e.g., "BTCUSD-24DEC31")
    ticker: str
    # Which side was bet: "yes" (predicting event happens) or "no" (predicting it doesn't)
    side: str
    # The price paid per contract at entry, as a fraction (0.45 = 45 cents)
    entry_price: float
    # The price per contract when the trade closed (1.0 = full win, 0.0 = full loss)
    exit_price: float
    # When we entered the trade (ISO 8601 format: "2024-12-31T14:30:00Z")
    entry_time: str = ""
    # When the trade closed (ISO 8601 format)
    exit_time: str = ""
    # Number of contracts bought (each contract is worth $1 if it resolves your way)
    contracts: int = 1
    # Net profit or loss in cents (computed automatically). +250 = made $2.50
    pnl_cents: int = 0
    # Logarithmic return: ln(exit/entry) for YES, ln((1-entry)/(1-exit)) for NO.
    # Log returns are additive: sum them up to get total compounded return.
    log_return: float = 0.0
    # Max Adverse Excursion: the worst the unrealized loss got during this trade.
    # Example: entry=0.40, price dipped to 0.25 → MAE = 0.15
    mae: float = 0.0
    # Max Favorable Excursion: the best the unrealized gain got during this trade.
    # Example: entry=0.40, price hit 0.75 → MFE = 0.35
    mfe: float = 0.0
    # The bot's machine-learning model's predicted probability at entry time.
    # Used to calculate "edge" = model_probability - market_probability_at_entry
    model_probability: float = 0.0
    # What the market was pricing this at when we entered (the market consensus)
    market_probability_at_entry: float = 0.0
    # Did we make money? True if pnl_cents > 0 (computed automatically)
    won: bool = False
    # The type of market: "politics", "crypto", "sports", "economics", etc.
    category: str = ""
    # Free-text notes a user can add via the dashboard for journaling purposes
    notes: str = ""
    # What the model thought the fair price was (for slippage measurement)
    expected_price: float = 0.0
    # What price we actually got when the order filled (may differ from expected)
    actual_price: float = 0.0
    # Auto-generated descriptive tags: "high_confidence", "contrarian", "quick_flip", etc.
    tags: list = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """Aggregated performance statistics computed from all completed trades.

    This is the "report card" object — a single snapshot of how the strategy
    is performing across all trades. The frontend dashboard displays these
    numbers in real time.

    All float values are pre-rounded for display. This dataclass is returned by
    PerformanceTracker.get_metrics() and serialized to JSON for the frontend.
    """
    # How many trades have completed in total
    total_trades: int = 0
    # How many trades made money (pnl_cents > 0)
    wins: int = 0
    # How many trades lost money (pnl_cents <= 0)
    losses: int = 0
    # wins / total_trades — what fraction of trades are profitable
    win_rate: float = 0.0
    # Sum of all trade P&Ls in cents. Positive = overall profitable.
    total_pnl_cents: int = 0
    # Sharpe ratio: return per unit of risk. See full explanation above.
    sharpe_ratio: float = 0.0
    # Average log return per trade (additive measure of compounded growth)
    avg_log_return: float = 0.0
    # Average Max Adverse Excursion (how bad did the trade look before resolving?)
    avg_mae: float = 0.0
    # Average Max Favorable Excursion (how good did the trade look at its best?)
    avg_mfe: float = 0.0
    # Average "edge" = model probability - market probability across all trades
    # Positive avg_edge means we're consistently finding mispricings.
    avg_edge: float = 0.0
    # The single best (most profitable) trade's P&L in cents
    best_trade_pnl: int = 0
    # The single worst (biggest loss) trade's P&L in cents
    worst_trade_pnl: int = 0
    # Gross profit / Gross loss. Above 1.0 = profitable. 2.0 = earn $2 per $1 lost.
    profit_factor: float = 0.0
    # Largest peak-to-trough decline in account equity in cents
    max_drawdown_cents: int = 0
    # How long we hold positions on average (hours)
    avg_holding_period_hours: float = 0.0
    # Ratio of actual return to predicted edge: 1.0 = perfect, <1.0 = leaking edge
    edge_realization_ratio: float = 0.0
    # Human-readable label for the Sharpe ratio: "Bad", "Good", or "Excellent"
    sharpe_label: str = "N/A"


class PerformanceTracker:
    """
    Stateful tracker that accumulates every completed trade and computes
    comprehensive performance analytics.

    Think of this as the "accountant" of the bot. After every trade, the
    tracker:
      1. Records the full trade details in a TradeRecord
      2. Updates the equity curve (running total of all P&L)
      3. Persists the trade to the database (if connected)

    Then, on demand, it can compute any of the metrics described in the
    module docstring above: Sharpe ratio, win rate, drawdown, etc.

    EQUITY CURVE:
    A list of data points that shows how the account value changes over time.
    Each data point is: {time: "...", equity_cents: 5200, trade_num: 15}
    Plotting this curve shows the "shape" of the strategy — a good strategy
    has a smooth upward curve; a bad one is jagged and declining.

    The equity curve starts at 0 (representing net P&L, not total balance).
    If the curve is at +$50 after 20 trades, the strategy has made $50 net.

    Key formulas used throughout this class:
    - log_return = ln(P_exit / P_entry)
    - Sharpe Ratio = (mean(returns) - risk_free_rate) / std(returns)
    - MAE/MFE tracking for exit optimization
    """

    # Risk-free rate for Sharpe Ratio calculation (configurable via config.py)
    # This represents the "free money" baseline — what you'd earn in a savings account.
    # Currently 5% annual = 5%/365 ≈ 0.014% per day.
    # If Sharpe uses this as the baseline, the ratio only rewards performance
    # that EXCEEDS what you'd get by doing nothing.
    @staticmethod
    def _get_risk_free_rate() -> float:
        try:
            from bot.config import config
            return config.risk_free_rate
        except Exception:
            return 0.05 / 365

    @property
    def RISK_FREE_DAILY(self) -> float:
        return self._get_risk_free_rate()

    def __init__(self, db=None, mode: str = "paper"):
        """Initialize the performance tracker.

        Args:
            db: Optional Database instance for persisting trades to Supabase.
            mode: "paper" or "live" — used as the mode tag when inserting trades to DB.
        """
        self.db = db                         # Optional Database instance for Supabase persistence
        self.mode = mode                     # "paper" or "live" — tags trades in DB
        self.trades: list[TradeRecord] = []  # Chronological list of all completed trades
        self.equity_curve: list[dict] = []   # [{time, equity_cents, trade_num}] for charting
        self._peak_equity = 0                # High-water mark for drawdown calculation
        self._current_equity = 0             # Running cumulative P&L in cents

    def record_trade(
        self,
        ticker: str,
        side: str,
        entry_price: float,
        exit_price: float,
        contracts: int = 1,
        mae: float = 0.0,
        mfe: float = 0.0,
        model_probability: float = 0.0,
        market_probability_at_entry: float = 0.0,
        entry_time: str = "",
        exit_time: str = "",
        category: str = "",
        notes: str = "",
        expected_price: float = 0.0,
        actual_price: float = 0.0,
    ) -> TradeRecord:
        """Record a completed trade, compute its P&L and log return, and update the equity curve.

        Args:
            ticker: Kalshi market ticker.
            side: "yes" or "no".
            entry_price: Entry price as decimal 0-1.
            exit_price: Exit price as decimal 0-1 (1.0 for winning YES, 0.0 for losing YES).
            contracts: Number of contracts.
            mae: Max Adverse Excursion observed during the trade.
            mfe: Max Favorable Excursion observed during the trade.
            model_probability: Model's predicted probability at entry.
            market_probability_at_entry: Market price at entry.
            entry_time: ISO timestamp of entry (defaults to now).
            exit_time: ISO timestamp of exit (defaults to now).
            category: Market category for per-category analytics.
            notes: Optional trade journal notes.

        Returns:
            The completed TradeRecord with computed pnl_cents, log_return, and won fields.
        """
        # Fix 31: Validate entry_price and exit_price are in [0, 1] range
        entry_price = max(0.0, min(1.0, entry_price))
        exit_price = max(0.0, min(1.0, exit_price))

        # Fix 32: Validate contracts >= 1
        contracts = max(1, contracts)

        # Fix 33: Calculate hold_time_hours if entry_time and exit_time provided
        hold_time_hours = 0.0
        if entry_time and exit_time:
            try:
                _entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                _exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                hold_time_hours = (_exit_dt - _entry_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                pass

        # ── P&L Calculation ────────────────────────────────────────────────
        # P&L (Profit and Loss) = (exit price - entry price) × contracts
        # For YES bets:
        #   If market resolves YES: exit_price = 1.0, so pnl = (1.0 - 0.40) × 100 × 5 = +300 cents
        #   If market resolves NO:  exit_price = 0.0, so pnl = (0.0 - 0.40) × 100 × 5 = -200 cents
        # For NO bets: the sign is flipped because we profit when YES price falls
        if side == "yes":
            pnl_cents = int((exit_price - entry_price) * 100 * contracts)
        else:
            # For NO bets: we profit when YES price goes DOWN
            # entry_price = what YES was priced when we entered
            # exit_price = what YES was priced when we exited
            # If YES went from 0.60 → 0.20, our NO bet made (0.60 - 0.20) × 100 = 40 cents per contract
            pnl_cents = int((entry_price - exit_price) * 100 * contracts)

        # ── Log Return Calculation ─────────────────────────────────────────
        # Log return = ln(payout / cost) — the natural logarithm of the ratio.
        # This measures: "How many times did my money multiply?"
        # ln(2.0) = 0.69 → doubled money
        # ln(0.5) = -0.69 → lost half
        # ln(1.0) = 0.0  → broke even
        #
        # Why use log returns instead of simple (%) returns?
        # Log returns are ADDITIVE: you can sum them to get compound return.
        # Simple returns are MULTIPLICATIVE (harder to work with).
        # Example: +50% then -50% in simple returns = net -25% (not 0%)
        #          In log returns: +0.405 + (-0.405) = 0.0 (correctly shows the loss)
        #
        # For YES bets: cost = what we paid, payout = what we received
        # For NO bets:  we're effectively buying the complement, so
        #               cost = 1 - entry_YES, payout = 1 - exit_YES
        if side == "yes":
            cost = entry_price
            payout = exit_price
        else:
            cost = 1 - entry_price
            payout = 1 - exit_price
        if cost > 0 and payout > 0:
            log_return = math.log(payout / cost)
        elif cost > 0 and payout <= 0:
            log_return = -1.0  # Total loss: -100% in simple terms, capped at -1.0
        else:
            log_return = 0.0

        won = pnl_cents > 0

        # Auto-tag the trade based on characteristics
        tags = []
        # Fix 46: Handle missing model_probability gracefully
        _model_prob = model_probability if model_probability is not None else 0.5
        _market_prob = market_probability_at_entry if market_probability_at_entry is not None else 0.5
        confidence = max(_model_prob, 1 - _model_prob)
        if confidence > 0.9:
            tags.append("high_confidence")
        edge = _model_prob - _market_prob
        if abs(edge) > 0.15:
            tags.append("high_edge")
        # Fix 47: Add "large_bet" tag when contracts > median
        if self.trades:
            median_contracts = sorted(t.contracts for t in self.trades)[len(self.trades) // 2]
            if contracts > median_contracts:
                tags.append("large_bet")
        # Fix 48: Add "quick_flip" tag when hold_time < 1 hour
        if hold_time_hours > 0 and hold_time_hours < 1.0:
            tags.append("quick_flip")
        # Contrarian vs momentum: compare model direction to market momentum
        if _model_prob > 0.5 and _market_prob < 0.4:
            tags.append("contrarian")
        elif _model_prob < 0.5 and _market_prob > 0.6:
            tags.append("contrarian")
        elif _model_prob > 0.5 and _market_prob > 0.5:
            tags.append("momentum")
        elif _model_prob < 0.5 and _market_prob < 0.5:
            tags.append("momentum")
        # Expiry play: check if entry_time is close to close_time (we approximate via exit_time)
        _entry_t = entry_time or datetime.now(timezone.utc).isoformat()
        _exit_t = exit_time or datetime.now(timezone.utc).isoformat()
        try:
            _entry_dt = datetime.fromisoformat(_entry_t.replace("Z", "+00:00"))
            _exit_dt = datetime.fromisoformat(_exit_t.replace("Z", "+00:00"))
            if (_exit_dt - _entry_dt).total_seconds() < 86400:
                tags.append("expiry_play")
        except (ValueError, TypeError):
            pass
        # Small cap proxy: low volume indicated by low market probability variance
        # We don't have volume here, but we can flag based on market_probability being extreme
        if _market_prob > 0 and _market_prob < 0.1:
            tags.append("small_cap")
        elif _market_prob > 0.9:
            tags.append("small_cap")

        trade = TradeRecord(
            ticker=ticker,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time or datetime.now(timezone.utc).isoformat(),
            exit_time=exit_time or datetime.now(timezone.utc).isoformat(),
            contracts=contracts,
            pnl_cents=pnl_cents,
            log_return=log_return,
            mae=mae,
            mfe=mfe,
            model_probability=model_probability,
            market_probability_at_entry=market_probability_at_entry,
            won=won,
            category=category,
            notes=notes,
            expected_price=expected_price,
            actual_price=actual_price,
            tags=tags,
        )
        self.trades.append(trade)

        # Fix 34: Log trade summary
        logger.info(f"Trade recorded: {ticker} {side.upper()} P&L={pnl_cents:+d}c ({'W' if won else 'L'})")

        # Persist to DB if connected
        if self.db:
            try:
                self.db.insert_trade(self.mode, trade)
            except Exception:
                pass  # DB writes are best-effort

        # Update equity curve
        self._current_equity += pnl_cents
        self._peak_equity = max(self._peak_equity, self._current_equity)
        self.equity_curve.append({
            "time": trade.exit_time,
            "equity_cents": self._current_equity,
            "trade_num": len(self.trades),
        })
        # Limit equity curve to prevent unbounded memory growth
        if len(self.equity_curve) > 10000:
            self.equity_curve = self.equity_curve[-10000:]

        return trade

    def get_metrics(self) -> PerformanceMetrics:
        """Compute all performance metrics from the accumulated trade history.

        Calculations:
          - Sharpe Ratio: (mean_log_return - risk_free_daily) / std_log_returns
          - Profit Factor: gross_profit / gross_loss
          - Max Drawdown: largest peak-to-trough decline in the equity curve
          - Win Rate: wins / total_trades
          - Average Edge: mean(model_prob - market_prob) across all trades

        Returns:
            PerformanceMetrics dataclass with all computed fields.
        """
        if not self.trades:
            return PerformanceMetrics()

        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]
        total = len(self.trades)

        # ── Simple Returns (for Sharpe Calculation) ───────────────────────
        # We use SIMPLE percentage returns for the Sharpe ratio, not log returns.
        # Why? Binary (prediction) markets always resolve to 0 or 1, and
        # ln(0) = negative infinity — which would break the math.
        # Simple return = (what you got back - what you paid) / what you paid
        # Example: paid 0.40, got 1.00 → return = (1.00 - 0.40) / 0.40 = +150%
        # Example: paid 0.40, got 0.00 → return = (0.00 - 0.40) / 0.40 = -100%
        simple_returns = []
        for t in self.trades:
            if t.side == "yes":
                cost = t.entry_price
                payout = t.exit_price
            else:
                cost = 1 - t.entry_price
                payout = 1 - t.exit_price
            if cost > 0:
                simple_returns.append((payout - cost) / cost)
            else:
                simple_returns.append(0.0)

        avg_return = sum(simple_returns) / total if total > 0 else 0

        # ── Sharpe Ratio Calculation ───────────────────────────────────────
        # Formula: Sharpe = (average_return - risk_free_rate) / std(returns)
        #
        # std_returns = standard deviation of returns (how much they vary)
        #   Low std = consistent returns → higher Sharpe
        #   High std = erratic returns → lower Sharpe
        #
        # RISK_FREE_DAILY = the daily equivalent of a risk-free savings rate.
        # We subtract it because any returns below this aren't worth the risk —
        # you could have just put the money in a savings account.
        #
        # Example: avg_return = 5%, risk_free = 2%, std = 10%
        # Sharpe = (5% - 2%) / 10% = 0.3 → "Bad" (not much reward for the risk)
        std_returns = self._std(simple_returns)
        sharpe = (avg_return - self.RISK_FREE_DAILY) / std_returns if std_returns > 0.0001 else 0

        # Keep log return stats for other analytics (not used in Sharpe)
        log_returns = [t.log_return for t in self.trades]
        avg_log_return = sum(log_returns) / total if total > 0 else 0

        # Human-readable label for how good the Sharpe ratio is
        if sharpe < 1:
            sharpe_label = "Bad"
        elif sharpe < 2:
            sharpe_label = "Good"
        else:
            sharpe_label = "Excellent"

        # Fix 37: Annualized Sharpe (daily * sqrt(365))
        # Multiplying by sqrt(365) converts a daily Sharpe to an annual Sharpe.
        # This is a statistical convention: risk scales with the square root of time.
        annualized_sharpe = round(sharpe * math.sqrt(365), 2)

        # Fix 38: Profit factor — handle no losing trades
        gross_profit = sum(t.pnl_cents for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl_cents for t in losses)) if losses else 0
        if gross_loss == 0:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        # Fix 39: Max consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        cur_wins = 0
        cur_losses = 0
        for t in self.trades:
            if t.won:
                cur_wins += 1
                cur_losses = 0
                max_consec_wins = max(max_consec_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_consec_losses = max(max_consec_losses, cur_losses)

        # Fix 42: Separate win rates for YES and NO sides
        yes_trades = [t for t in self.trades if t.side == "yes"]
        no_trades = [t for t in self.trades if t.side == "no"]
        yes_win_rate = sum(1 for t in yes_trades if t.won) / len(yes_trades) if yes_trades else 0.0
        no_win_rate = sum(1 for t in no_trades if t.won) / len(no_trades) if no_trades else 0.0

        # Fix 43: Separate P&L for YES and NO sides
        yes_pnl = sum(t.pnl_cents for t in yes_trades)
        no_pnl = sum(t.pnl_cents for t in no_trades)

        # Fix 44: Track time spent in drawdown (underwater days)
        underwater_days = 0
        if self.equity_curve:
            dd_peak = 0
            last_date = None
            for point in self.equity_curve:
                eq = point["equity_cents"]
                dd_peak = max(dd_peak, eq)
                if eq < dd_peak:
                    try:
                        cur_date = datetime.fromisoformat(point["time"].replace("Z", "+00:00")).date()
                        if cur_date != last_date:
                            underwater_days += 1
                            last_date = cur_date
                    except (ValueError, TypeError, KeyError):
                        pass

        # Max drawdown — initialize peak to 0 (initial balance baseline)
        max_dd = 0
        peak = self._current_equity if self._current_equity > 0 else 0
        for point in self.equity_curve:
            eq = point["equity_cents"]
            peak = max(peak, eq)
            dd = peak - eq
            max_dd = max(max_dd, dd)

        pnls = [t.pnl_cents for t in self.trades]

        # ── Kelly Optimal Fraction (Retrospective) ────────────────────────
        # Using ACTUAL historical win/loss data, what does the Kelly formula say
        # was the ideal bet fraction? This is "in hindsight" — after the fact.
        #
        # Formula: f* = (b*p - q) / b
        # Where: b = avg_win / avg_loss (the payout ratio)
        #        p = win_rate
        #        q = 1 - p = loss_rate
        #
        # Example: 60% wins, avg win = $10, avg loss = $8
        # b = 10/8 = 1.25
        # f* = (1.25 × 0.60 - 0.40) / 1.25 = (0.75 - 0.40) / 1.25 = 0.28
        # Translation: Kelly says bet 28% of your bankroll per trade.
        # This is stored for display but NOT used to directly size live trades
        # (the live sizing uses kelly_size() in risk_manager.py with current estimates).
        #
        # Fix 41: Kelly-optimal f* = (b*p - q) / b
        win_rate_val = len(wins) / total if total > 0 else 0
        loss_rate_val = 1.0 - win_rate_val
        avg_win_amt = sum(t.pnl_cents for t in wins) / len(wins) if wins else 0
        avg_loss_amt = abs(sum(t.pnl_cents for t in losses) / len(losses)) if losses else 0
        if avg_loss_amt > 0:
            b_ratio = avg_win_amt / avg_loss_amt
            kelly_optimal_f = max(0.0, (b_ratio * win_rate_val - loss_rate_val) / b_ratio) if b_ratio > 0 else 0.0
        else:
            kelly_optimal_f = 0.0

        return PerformanceMetrics(
            total_trades=total,
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / total if total > 0 else 0,
            total_pnl_cents=sum(pnls),
            sharpe_ratio=round(sharpe, 2),
            avg_log_return=round(avg_log_return, 4),
            avg_mae=round(sum(t.mae for t in self.trades) / total, 4) if total > 0 else 0,
            avg_mfe=round(sum(t.mfe for t in self.trades) / total, 4) if total > 0 else 0,
            avg_edge=round(
                sum(t.model_probability - t.market_probability_at_entry for t in self.trades) / total, 4
            ) if total > 0 else 0,
            best_trade_pnl=max(pnls) if pnls else 0,
            worst_trade_pnl=min(pnls) if pnls else 0,
            profit_factor=round(profit_factor, 2),
            max_drawdown_cents=max_dd,
            avg_holding_period_hours=round(self._compute_avg_holding_hours(), 1),
            edge_realization_ratio=round(self._compute_edge_realization(), 2),
            sharpe_label=sharpe_label,
        )

    def get_equity_curve(self) -> list[dict]:
        """Return equity curve data for charting. Fix 45/55: includes timestamps alongside trade numbers."""
        if not self.equity_curve:
            return []
        return self.equity_curve

    def get_trade_history(self) -> list[dict]:
        """Return trade history as serializable dicts."""
        return [
            {
                "ticker": t.ticker,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "contracts": t.contracts,
                "pnl_cents": t.pnl_cents,
                "log_return": round(t.log_return, 4),
                "mae": round(t.mae, 4),
                "mfe": round(t.mfe, 4),
                "model_probability": round(t.model_probability, 4),
                "market_probability_at_entry": round(t.market_probability_at_entry, 4),
                "won": t.won,
                "category": t.category,
                "notes": t.notes,
                "expected_price": round(t.expected_price, 4),
                "actual_price": round(t.actual_price, 4),
                "tags": t.tags,
            }
            for t in self.trades
        ]

    def get_execution_quality(self) -> dict:
        """Compute trade execution quality metrics: slippage and execution score.

        Slippage = actual_price - expected_price (positive = paid more than expected).
        Only considers trades where both expected_price and actual_price are set.

        Returns:
            Dict with avg_slippage, slippage_distribution (by bucket), execution_quality_score,
            and total_trades_with_data.
        """
        trades_with_data = [
            t for t in self.trades
            if t.expected_price > 0 and t.actual_price > 0
        ]

        if not trades_with_data:
            return {
                "avg_slippage": 0.0,
                "slippage_distribution": {},
                "execution_quality_score": 1.0,
                "total_trades_with_data": 0,
            }

        slippages = []
        for t in trades_with_data:
            # For YES side: slippage = actual - expected (positive = overpaid)
            # For NO side: slippage = expected - actual (positive = overpaid, since lower YES price is better for NO)
            if t.side == "yes":
                slippage = t.actual_price - t.expected_price
            else:
                slippage = t.expected_price - t.actual_price
            slippages.append(slippage)

        avg_slippage = sum(slippages) / len(slippages)

        # Slippage distribution by bucket
        distribution = {
            "negative (favorable)": sum(1 for s in slippages if s < -0.005),
            "neutral (-0.5% to +0.5%)": sum(1 for s in slippages if -0.005 <= s <= 0.005),
            "small (0.5% to 2%)": sum(1 for s in slippages if 0.005 < s <= 0.02),
            "large (>2%)": sum(1 for s in slippages if s > 0.02),
        }

        # Execution quality score: 1.0 = no slippage, lower = worse
        # Penalize based on average absolute slippage
        avg_abs_slippage = sum(abs(s) for s in slippages) / len(slippages)
        execution_quality_score = max(0.0, min(1.0, 1.0 - avg_abs_slippage * 10))

        return {
            "avg_slippage": round(avg_slippage, 6),
            "avg_abs_slippage": round(avg_abs_slippage, 6),
            "slippage_distribution": distribution,
            "execution_quality_score": round(execution_quality_score, 4),
            "total_trades_with_data": len(trades_with_data),
        }

    def get_metrics_by_category(self) -> dict:
        """Compute per-category performance metrics for the frontend heatmap/breakdown.

        Groups trades by their category field (politics, crypto, sports, etc.) and
        computes win rate, total P&L, and best/worst trade for each category.

        Returns:
            Dict mapping category name to a metrics dict.
        """
        # Fix 52: Handle empty trades
        if not self.trades:
            return {}
        from collections import defaultdict
        cats = defaultdict(list)
        for t in self.trades:
            cats[t.category or "uncategorized"].append(t)
        result = {}
        for cat, trades in cats.items():
            wins = [t for t in trades if t.won]
            pnls = [t.pnl_cents for t in trades]
            result[cat] = {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(trades) - len(wins),
                "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
                "total_pnl_cents": sum(pnls),
                "avg_pnl_cents": round(sum(pnls) / len(pnls), 2) if pnls else 0,
                "best_trade": max(pnls) if pnls else 0,
                "worst_trade": min(pnls) if pnls else 0,
            }
        return result

    def update_trade_notes(self, trade_index: int, notes: str) -> bool:
        """Update the journal notes on a specific trade (identified by list index).

        Persists the update to Supabase if connected. Returns False if the index
        is out of range.
        """
        if 0 <= trade_index < len(self.trades):
            self.trades[trade_index].notes = notes
            # Persist to DB if connected
            if self.db:
                try:
                    trade = self.trades[trade_index]
                    self.db.update_trade_notes(trade.ticker, trade.entry_time, notes)
                except Exception:
                    pass  # Best-effort DB persistence
            return True
        return False

    def _compute_avg_holding_hours(self) -> float:
        """Compute average holding period from entry/exit timestamps."""
        hours = []
        for t in self.trades:
            if t.entry_time and t.exit_time:
                try:
                    entry_dt = datetime.fromisoformat(t.entry_time.replace("Z", "+00:00"))
                    exit_dt = datetime.fromisoformat(t.exit_time.replace("Z", "+00:00"))
                    hours.append((exit_dt - entry_dt).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass
        return sum(hours) / len(hours) if hours else 0.0

    def _compute_edge_realization(self) -> float:
        """Ratio of realized return to predicted edge across all trades.

        "Edge realization" answers: "Of the edge we predicted, how much did
        we actually capture in real returns?"

        A ratio of 1.0 = perfect: realized returns matched predicted edges.
        A ratio of 0.5 = we're only capturing half the edge we're finding.
        A ratio of 0.0 = the model's predictions aren't translating to profit.

        This can be low because of:
          - Transaction costs eating into profits (spread, slippage)
          - The model overestimates its edge in certain conditions
          - Markets correcting before we can exit at the best price
        """
        if len(self.trades) < 5:
            return 0.0
        # predicted = how much edge the model thought it had on each trade
        # (positive = model thought YES was more likely than market priced)
        predicted = [t.model_probability - t.market_probability_at_entry for t in self.trades]
        # realized = what actually happened (the log return on each trade)
        realized = [t.log_return for t in self.trades]
        avg_p = sum(predicted) / len(predicted)
        avg_r = sum(realized) / len(realized)
        if avg_p <= 0:
            return 0.0
        # Capped at 3.0 to avoid absurd ratios from tiny predicted edges
        return max(0.0, min(3.0, avg_r / avg_p))

    def get_edge_decay_stats(self) -> dict:
        """Compute edge decay statistics from closed trades.

        "Edge decay" is the phenomenon where your advantage over the market
        erodes the longer you hold a position. Markets are somewhat efficient:
        the longer a trade is open, the more likely other traders have noticed
        the same opportunity and bid the price toward fair value.

        This function measures: on average, how fast does our edge shrink per hour?

        Practical use: if edge decays quickly (within hours), we should aim for
        shorter hold times and tighter exit targets. If edge is stable for days,
        we can hold positions longer without sacrificing expected return.

        OPTIMAL HOLD TIME: The function also tries to find the holding period
        that historically produced the best average P&L, by grouping trades
        into 5 buckets by hold time and comparing average profits.

        For each trade, calculates decay_rate = (exit_edge - entry_edge) / hold_time_hours.
        A negative decay_rate means the edge shrank over time (expected behavior).
        A positive decay_rate means the edge actually grew while holding (unusual).

        Returns:
            Dict with avg_decay_rate, median_hold_time_hours, optimal_hold_time_hours,
            and sample_count.
        """
        # Fix 53: Handle < 10 trades
        if len(self.trades) < 10:
            return {"avg_decay_rate": 0.0, "median_hold_time_hours": 0.0,
                    "optimal_hold_time_hours": 0.0, "sample_count": len(self.trades),
                    "note": "Need at least 10 trades for meaningful edge decay analysis"}

        decay_records = []
        hold_times = []

        for t in self.trades:
            if not t.entry_time or not t.exit_time:
                continue
            try:
                entry_dt = datetime.fromisoformat(t.entry_time.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(t.exit_time.replace("Z", "+00:00"))
                hold_hours = (exit_dt - entry_dt).total_seconds() / 3600
            except (ValueError, TypeError):
                continue
            if hold_hours <= 0:
                continue

            hold_times.append(hold_hours)
            entry_edge = abs(t.model_probability - t.market_probability_at_entry)
            if t.side == "yes":
                exit_edge = abs(t.model_probability - t.exit_price)
            else:
                exit_edge = abs((1 - t.model_probability) - (1 - t.exit_price))
            decay_rate = (exit_edge - entry_edge) / hold_hours
            decay_records.append({
                "hold_hours": hold_hours,
                "decay_rate": decay_rate,
                "pnl_cents": t.pnl_cents,
            })

        if not decay_records:
            return {"avg_decay_rate": 0.0, "median_hold_time_hours": 0.0,
                    "optimal_hold_time_hours": 0.0, "sample_count": 0}

        avg_decay = sum(r["decay_rate"] for r in decay_records) / len(decay_records)
        sorted_holds = sorted(hold_times)
        n = len(sorted_holds)
        median_hold = sorted_holds[n // 2] if n % 2 == 1 else (sorted_holds[n // 2 - 1] + sorted_holds[n // 2]) / 2

        optimal_hold = median_hold
        if len(decay_records) >= 5:
            sorted_by_hold = sorted(decay_records, key=lambda r: r["hold_hours"])
            bucket_size = max(1, len(sorted_by_hold) // 5)
            best_avg_pnl = float("-inf")
            for i in range(0, len(sorted_by_hold), bucket_size):
                bucket = sorted_by_hold[i:i + bucket_size]
                avg_pnl = sum(r["pnl_cents"] for r in bucket) / len(bucket)
                avg_hold = sum(r["hold_hours"] for r in bucket) / len(bucket)
                if avg_pnl > best_avg_pnl:
                    best_avg_pnl = avg_pnl
                    optimal_hold = avg_hold

        return {
            "avg_decay_rate": round(avg_decay, 6),
            "median_hold_time_hours": round(median_hold, 2),
            "optimal_hold_time_hours": round(optimal_hold, 2),
            "sample_count": len(decay_records),
        }

    def estimate_win_probability(self, position: dict, avg_hold_hours: float = 0.0) -> dict:
        """Estimate win probability for an open position based on price movement,
        historical win rate at similar edge levels, and time elapsed.

        Args:
            position: Dict with ticker, side, entry_price, model_prob, entry_time,
                      and optionally current_price.
            avg_hold_hours: Average hold time from closed trades (0 = compute internally).

        Returns:
            Dict with estimated_win_probability and factor breakdown.
        """
        entry_price = position.get("entry_price", 0.5)
        current_price = position.get("current_price", entry_price)
        side = position.get("side", "yes")
        model_prob = position.get("model_prob", 0.5)
        entry_time = position.get("entry_time", "")

        if side == "yes":
            price_factor = 0.5 + (current_price - entry_price) * 2
        else:
            price_factor = 0.5 + (entry_price - current_price) * 2
        price_factor = max(0.1, min(0.9, price_factor))

        entry_edge = abs(model_prob - entry_price)
        similar_trades = [
            t for t in self.trades
            if abs(abs(t.model_probability - t.market_probability_at_entry) - entry_edge) < 0.05
        ]
        if len(similar_trades) >= 3:
            hist_win_rate = sum(1 for t in similar_trades if t.won) / len(similar_trades)
        else:
            hist_win_rate = sum(1 for t in self.trades if t.won) / max(len(self.trades), 1) if self.trades else 0.5

        time_factor = 0.5
        if entry_time and avg_hold_hours <= 0:
            avg_hold_hours = self._compute_avg_holding_hours()
        if entry_time and avg_hold_hours > 0:
            try:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                elapsed_hours = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                time_ratio = elapsed_hours / max(avg_hold_hours, 0.1)
                time_factor = max(0.2, 1.0 - time_ratio * 0.3)
            except (ValueError, TypeError):
                pass

        estimated_prob = 0.4 * price_factor + 0.35 * hist_win_rate + 0.25 * time_factor
        estimated_prob = max(0.05, min(0.95, estimated_prob))
        confidence = "low" if len(similar_trades) < 3 else ("medium" if len(similar_trades) < 10 else "high")

        return {
            "estimated_win_probability": round(estimated_prob, 4),
            "price_factor": round(price_factor, 4),
            "historical_win_rate": round(hist_win_rate, 4),
            "time_factor": round(time_factor, 4),
            "similar_trades_count": len(similar_trades),
            "confidence": confidence,
        }

    @staticmethod
    def _std(values: list[float]) -> float:
        """Compute sample standard deviation (Bessel's correction: N-1 denominator).

        "Standard deviation" measures how spread out a list of numbers is.
        Low std = numbers are close to the average (consistent).
        High std = numbers are very spread out (volatile).

        Example: [1, 1, 1, 1] → std = 0.0 (no variation)
                 [0, 50, -30, 80] → std = large (very erratic returns)

        WHY N-1 (Bessel's Correction)?
        When estimating population standard deviation from a SAMPLE, dividing by
        (N-1) instead of N gives a better (unbiased) estimate. This is standard
        statistical practice when you have a sample but not the whole population.
        In our case, our trades are a sample of all possible trades.
        """
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        # Sum of squared deviations from the mean
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)

    # ── Trade Correlation Tracking ──────────────────────────────────────────

    def get_category_correlation(self) -> dict:
        """Compute win/loss correlation between market categories.

        Groups trades by category, computes per-category win rates, and returns
        a correlation matrix showing which categories tend to win/lose together.

        Returns:
            Dict with 'categories' (list), 'win_rates' (per-cat), and 'correlation_matrix'.
        """
        from collections import defaultdict

        cats = defaultdict(list)
        for t in self.trades:
            cats[t.category or "uncategorized"].append(t)

        cat_names = sorted(cats.keys())
        if len(cat_names) < 2:
            return {
                "categories": cat_names,
                "win_rates": {c: round(sum(1 for t in cats[c] if t.won) / max(len(cats[c]), 1), 4) for c in cat_names},
                "correlation_matrix": {},
            }

        # Build per-category win/loss sequences aligned by time windows
        # Group trades into time buckets (1-hour windows)
        time_buckets: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        for t in self.trades:
            # Use entry_time hour as bucket key
            bucket = t.entry_time[:13] if t.entry_time else "unknown"
            cat = t.category or "uncategorized"
            time_buckets[bucket][cat].append(t.won)

        # Build per-category outcome vectors (avg win rate per time bucket)
        all_buckets = sorted(time_buckets.keys())
        cat_vectors: dict[str, list[float]] = {c: [] for c in cat_names}
        for bucket in all_buckets:
            for cat in cat_names:
                outcomes = time_buckets[bucket].get(cat, [])
                if outcomes:
                    cat_vectors[cat].append(sum(outcomes) / len(outcomes))
                else:
                    cat_vectors[cat].append(0.5)  # neutral when no trades

        # Compute pairwise correlation
        matrix = {}
        for i, cat_a in enumerate(cat_names):
            matrix[cat_a] = {}
            for j, cat_b in enumerate(cat_names):
                if i == j:
                    matrix[cat_a][cat_b] = 1.0
                    continue
                va = cat_vectors[cat_a]
                vb = cat_vectors[cat_b]
                n = len(va)
                if n < 3:
                    matrix[cat_a][cat_b] = 0.0
                    continue
                mean_a = sum(va) / n
                mean_b = sum(vb) / n
                cov = sum((va[k] - mean_a) * (vb[k] - mean_b) for k in range(n)) / (n - 1)
                std_a = self._std(va)
                std_b = self._std(vb)
                if std_a > 0 and std_b > 0:
                    matrix[cat_a][cat_b] = round(cov / (std_a * std_b), 4)
                else:
                    matrix[cat_a][cat_b] = 0.0

        return {
            "categories": cat_names,
            "win_rates": {c: round(sum(1 for t in cats[c] if t.won) / max(len(cats[c]), 1), 4) for c in cat_names},
            "correlation_matrix": matrix,
        }

    # ── Time-of-Day Performance ─────────────────────────────────────────────

    def get_performance_by_hour(self) -> dict:
        """Group trades by hour-of-day (UTC) and return win rate and avg P&L per hour.

        Returns:
            Dict mapping hour (0-23) to {trades, wins, win_rate, avg_pnl_cents}.
        """
        hours: dict[int, list[TradeRecord]] = {}
        for t in self.trades:
            if t.entry_time:
                try:
                    dt = datetime.fromisoformat(t.entry_time.replace("Z", "+00:00"))
                    h = dt.hour
                    hours.setdefault(h, []).append(t)
                except (ValueError, TypeError):
                    pass

        result = {}
        for h in sorted(hours.keys()):
            trades = hours[h]
            wins = [t for t in trades if t.won]
            pnls = [t.pnl_cents for t in trades]
            result[h] = {
                "trades": len(trades),
                "wins": len(wins),
                "win_rate": round(len(wins) / len(trades), 4) if trades else 0,
                "avg_pnl_cents": round(sum(pnls) / len(pnls), 2) if pnls else 0,
                "total_pnl_cents": sum(pnls),
            }
        return result

    # ── Streak Analysis ─────────────────────────────────────────────────────

    def get_streak_analysis(self) -> dict:
        """Track current, longest, and average winning/losing streaks.

        Includes a streak-aware Kelly adjustment recommendation:
        during a long losing streak, reduce position sizes.

        Returns:
            Dict with current_streak, longest_win_streak, longest_loss_streak,
            avg_win_streak, avg_loss_streak, kelly_adjustment.
        """
        if not self.trades:
            return {
                "current_streak": {"type": "none", "length": 0},
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
                "avg_win_streak": 0.0,
                "avg_loss_streak": 0.0,
                "kelly_adjustment": 1.0,
            }

        # Build streak list
        win_streaks = []
        loss_streaks = []
        current_type = None
        current_length = 0

        for t in self.trades:
            if t.won:
                if current_type == "win":
                    current_length += 1
                else:
                    if current_type == "loss" and current_length > 0:
                        loss_streaks.append(current_length)
                    current_type = "win"
                    current_length = 1
            else:
                if current_type == "loss":
                    current_length += 1
                else:
                    if current_type == "win" and current_length > 0:
                        win_streaks.append(current_length)
                    current_type = "loss"
                    current_length = 1

        # Record final streak
        if current_type == "win":
            win_streaks.append(current_length)
        elif current_type == "loss":
            loss_streaks.append(current_length)

        longest_win = max(win_streaks) if win_streaks else 0
        longest_loss = max(loss_streaks) if loss_streaks else 0
        avg_win = sum(win_streaks) / len(win_streaks) if win_streaks else 0
        avg_loss = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0

        # Kelly adjustment: reduce sizing during losing streaks
        kelly_adj = 1.0
        if current_type == "loss" and current_length >= 3:
            kelly_adj = max(0.25, 1.0 - (current_length - 2) * 0.2)

        return {
            "current_streak": {"type": current_type or "none", "length": current_length},
            "longest_win_streak": longest_win,
            "longest_loss_streak": longest_loss,
            "avg_win_streak": round(avg_win, 1),
            "avg_loss_streak": round(avg_loss, 1),
            "kelly_adjustment": round(kelly_adj, 2),
            "total_win_streaks": len(win_streaks),
            "total_loss_streaks": len(loss_streaks),
        }

    # ── Trade Journaling with Auto-Tags (Task 4) ──────────────────────────

    def get_trades_by_tag(self, tag: str) -> list[dict]:
        """Return all trades matching a specific auto-tag.

        Args:
            tag: One of "high_confidence", "contrarian", "momentum", "expiry_play",
                 "high_edge", "small_cap".

        Returns:
            List of trade dicts matching the tag, with computed metrics for that subset.
        """
        matching = [t for t in self.trades if tag in t.tags]
        if not matching:
            return []
        wins = [t for t in matching if t.won]
        pnls = [t.pnl_cents for t in matching]
        return {
            "tag": tag,
            "total_trades": len(matching),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(matching), 4) if matching else 0,
            "total_pnl_cents": sum(pnls),
            "avg_pnl_cents": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "trades": [
                {
                    "ticker": t.ticker,
                    "side": t.side,
                    "pnl_cents": t.pnl_cents,
                    "won": t.won,
                    "tags": t.tags,
                    "entry_time": t.entry_time,
                }
                for t in matching
            ],
        }

    def get_all_tag_stats(self) -> dict:
        """Return stats for all auto-tags across all trades. Fix 54: Handle empty trades."""
        if not self.trades:
            return {}
        all_tags = set()
        for t in self.trades:
            all_tags.update(t.tags)
        result = {}
        for tag in sorted(all_tags):
            matching = [t for t in self.trades if tag in t.tags]
            wins = [t for t in matching if t.won]
            pnls = [t.pnl_cents for t in matching]
            result[tag] = {
                "total_trades": len(matching),
                "wins": len(wins),
                "win_rate": round(len(wins) / len(matching), 4) if matching else 0,
                "total_pnl_cents": sum(pnls),
                "avg_pnl_cents": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            }
        return result

    # ── Expected Value per Trade (Task 13) ─────────────────────────────────

    def get_ev_metrics(self) -> dict:
        """Calculate expected value per trade and rolling EV trend.

        EXPECTED VALUE (EV) is the single most important concept in trading.

        EV = (win_rate × avg_win) - (loss_rate × avg_loss)

        Example:
          60% win rate, avg win = $10, avg loss = $8
          EV = (0.60 × $10) - (0.40 × $8) = $6.00 - $3.20 = +$2.80 per trade

        This means: over many trades, we expect to make $2.80 per trade on average.
        A positive EV means the strategy is "profitable in expectation."
        A negative EV means we're losing money on average — stop trading.

        The ROLLING EV shows how EV changes over time. If rolling EV is declining,
        the strategy is getting worse (possibly market conditions changed, or
        the edge is being "arbitraged away" by other traders).

        Also computes theoretical Kelly EV for comparison and rolling EV over last 50 trades.

        Returns:
            Dict with ev_per_trade, kelly_ev, ev_trend (rolling 50), and ev_comparison.
        """
        if not self.trades:
            return {"ev_per_trade": 0.0, "kelly_ev": 0.0, "ev_trend": [], "ev_comparison": 0.0}

        wins = [t for t in self.trades if t.won]
        losses = [t for t in self.trades if not t.won]

        win_rate = len(wins) / len(self.trades) if self.trades else 0
        loss_rate = 1 - win_rate
        avg_win = sum(t.pnl_cents for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl_cents for t in losses) / len(losses)) if losses else 0

        ev_per_trade = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Fix 50: Handle division by zero in Kelly EV
        # Theoretical Kelly EV: f* = (bp - q) / b where b = avg_win/avg_loss
        b = avg_win / max(avg_loss, 1) if avg_loss > 0 else 0
        kelly_f = max(0, (b * win_rate - loss_rate) / b) if b > 0 else 0
        kelly_ev = kelly_f * ((b * win_rate) - loss_rate) * avg_loss if kelly_f > 0 and avg_loss > 0 else 0

        # Rolling EV over last 50 trades
        ev_trend = []
        window = 50
        for i in range(min(window, len(self.trades)), len(self.trades) + 1):
            start = max(0, i - window)
            chunk = self.trades[start:i]
            if not chunk:
                continue
            chunk_wins = [t for t in chunk if t.won]
            chunk_losses = [t for t in chunk if not t.won]
            wr = len(chunk_wins) / len(chunk)
            lr = 1 - wr
            aw = sum(t.pnl_cents for t in chunk_wins) / len(chunk_wins) if chunk_wins else 0
            al = abs(sum(t.pnl_cents for t in chunk_losses) / len(chunk_losses)) if chunk_losses else 0
            rolling_ev = (wr * aw) - (lr * al)
            ev_trend.append({
                "trade_num": i,
                "ev_cents": round(rolling_ev, 2),
            })

        return {
            "ev_per_trade": round(ev_per_trade, 2),
            "kelly_ev": round(kelly_ev, 2),
            "ev_comparison": round(ev_per_trade - kelly_ev, 2),
            "win_rate": round(win_rate, 4),
            "avg_win_cents": round(avg_win, 2),
            "avg_loss_cents": round(avg_loss, 2),
            "ev_trend": ev_trend,
        }

    # ── MAE/MFE Analysis (Task 14) ────────────────────────────────────────

    def get_mae_mfe_analysis(self) -> dict:
        """Enhanced MAE/MFE analysis grouped by trade outcome.

        Winners should have higher MFE than MAE.
        Losers should have higher MAE than MFE.
        Trade efficiency = actual_gain / MFE for winners.

        Returns:
            Dict with winner and loser MAE/MFE distributions, trade efficiency, and overall stats.
        """
        if not self.trades:
            return {
                "winners": {}, "losers": {}, "overall": {},
                "trade_efficiency": 0.0, "efficiency_distribution": [],
            }

        winners = [t for t in self.trades if t.won]
        losers = [t for t in self.trades if not t.won]

        def _mae_mfe_stats(trades_list):
            if not trades_list:
                return {"count": 0, "avg_mae": 0.0, "avg_mfe": 0.0, "mae_mfe_ratio": 0.0}
            maes = [t.mae for t in trades_list]
            mfes = [t.mfe for t in trades_list]
            avg_mae = sum(maes) / len(maes)
            avg_mfe = sum(mfes) / len(mfes)
            return {
                "count": len(trades_list),
                "avg_mae": round(avg_mae, 4),
                "avg_mfe": round(avg_mfe, 4),
                "max_mae": round(max(maes), 4) if maes else 0.0,
                "max_mfe": round(max(mfes), 4) if mfes else 0.0,
                "min_mae": round(min(maes), 4) if maes else 0.0,
                "min_mfe": round(min(mfes), 4) if mfes else 0.0,
                "mae_mfe_ratio": round(avg_mae / max(avg_mfe, 0.0001), 4),
            }

        # Trade efficiency for winners: actual_gain / MFE
        efficiencies = []
        for t in winners:
            if t.mfe > 0:
                actual_gain = t.pnl_cents / 100  # Convert to dollars for ratio
                efficiency = actual_gain / t.mfe if t.mfe > 0 else 0
                efficiencies.append(round(efficiency, 4))

        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0

        # Efficiency distribution by bucket
        efficiency_dist = {
            "excellent (>0.8)": sum(1 for e in efficiencies if e > 0.8),
            "good (0.5-0.8)": sum(1 for e in efficiencies if 0.5 <= e <= 0.8),
            "poor (<0.5)": sum(1 for e in efficiencies if e < 0.5),
        }

        return {
            "winners": _mae_mfe_stats(winners),
            "losers": _mae_mfe_stats(losers),
            "overall": _mae_mfe_stats(self.trades),
            "trade_efficiency": round(avg_efficiency, 4),
            "efficiency_distribution": efficiency_dist,
            "efficiencies": efficiencies[:100],  # Limit for UI
        }

    # ── Risk-Adjusted Return Metrics (Task 15) ────────────────────────────

    def get_advanced_metrics(self) -> dict:
        """Compute advanced risk-adjusted return metrics beyond Sharpe Ratio.

        Includes:
          - Sortino Ratio: Uses downside deviation only (penalizes losses, not upside volatility).
          - Calmar Ratio: Annualized return / max drawdown (measures return per unit of worst-case risk).
          - Information Ratio: Excess return vs buy-and-hold / tracking error.

        Returns:
            Dict with sortino_ratio, calmar_ratio, information_ratio, and supporting data.
        """
        if len(self.trades) < 2:
            return {
                "sortino_ratio": 0.0, "calmar_ratio": 0.0, "information_ratio": 0.0,
                "downside_deviation": 0.0, "annualized_return": 0.0,
            }

        # Compute simple returns
        simple_returns = []
        for t in self.trades:
            if t.side == "yes":
                cost = t.entry_price
                payout = t.exit_price
            else:
                cost = 1 - t.entry_price
                payout = 1 - t.exit_price
            if cost > 0:
                simple_returns.append((payout - cost) / cost)
            else:
                simple_returns.append(0.0)

        avg_return = sum(simple_returns) / len(simple_returns) if simple_returns else 0

        # Sortino Ratio: (Rp - Rf) / downside_deviation
        # Only considers negative returns for the deviation calculation
        downside_returns = [r for r in simple_returns if r < 0]
        if len(downside_returns) >= 2:
            downside_mean_sq = sum(r ** 2 for r in downside_returns) / len(downside_returns)
            downside_deviation = math.sqrt(downside_mean_sq)
        else:
            downside_deviation = 0.0

        sortino = (avg_return - self.RISK_FREE_DAILY) / downside_deviation if downside_deviation > 0 else 0.0

        # Calmar Ratio: annualized_return / max_drawdown
        # Approximate annualized return (assume ~1 trade/day avg)
        total_return = sum(simple_returns)
        n_days = max(1, len(self.trades))  # Approximate
        annualized_return = total_return * (365 / n_days)

        # Max drawdown from equity curve
        max_dd = 0
        peak = 0
        for point in self.equity_curve:
            eq = point["equity_cents"]
            peak = max(peak, eq)
            dd = peak - eq
            max_dd = max(max_dd, dd)

        # Convert max_dd to a fraction for Calmar
        total_invested = sum(
            max(1, int(t.entry_price * 100) if t.side == "yes" else int((1 - t.entry_price) * 100))
            * t.contracts for t in self.trades
        )
        max_dd_pct = max_dd / max(total_invested, 1)
        calmar = annualized_return / max_dd_pct if max_dd_pct > 0 else 0.0

        # Information Ratio: excess return vs buy-and-hold / tracking error
        # Buy-and-hold benchmark: assume each market resolves at 50% probability
        # so benchmark return = 0 per trade
        benchmark_returns = [0.0] * len(simple_returns)
        excess_returns = [s - b for s, b in zip(simple_returns, benchmark_returns)]
        avg_excess = sum(excess_returns) / len(excess_returns) if excess_returns else 0
        tracking_error = self._std(excess_returns)
        information_ratio = avg_excess / tracking_error if tracking_error > 0 else 0.0

        return {
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "information_ratio": round(information_ratio, 4),
            "downside_deviation": round(downside_deviation, 6),
            "annualized_return": round(annualized_return, 4),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "avg_return": round(avg_return, 6),
            "total_return": round(total_return, 4),
        }
