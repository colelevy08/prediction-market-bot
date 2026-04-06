"""
Environment-based configuration management for the prediction market bot.

Loads all settings from a .env file (via python-dotenv) and exposes them as
typed attributes on a Pydantic BaseModel. This provides validation, defaults,
and a single source of truth for all configurable parameters across the system.

Configuration groups:
  - Kalshi API:     API key ID, private key (file or env var), demo/production toggle.
  - Anthropic:      API key for Claude AI market analysis.
  - Supabase:       URL and service key for persistent database storage.
  - Trading Params: Max bet size, min edge threshold, daily loss limit, max positions,
                    max events to analyze per scan.
  - Kelly Criterion: Fractional Kelly multiplier (default 0.25 = quarter-Kelly).
  - Notifications:  Slack and Discord webhook URLs for trade alerts.
  - Webhook:        Secret token for authenticating inbound webhook triggers.
  - Retrain:        Scheduled retraining days (e.g., mon,wed,fri) and hour (UTC).

The module instantiates a singleton `config` object at import time, which is
imported throughout the codebase. The Kalshi private key is resolved with a
priority chain: KALSHI_PRIVATE_KEY env var (inline PEM string)
takes precedence over KALSHI_PRIVATE_KEY_PATH file (path to PEM file).

Used by: Every other module in the bot package.

---------------------------------------------------------------------------
EDUCATIONAL OVERVIEW
---------------------------------------------------------------------------

WHAT IS A CONFIGURATION FILE?
  Almost every real-world application needs tunable settings: passwords, URLs,
  feature flags, dollar limits. Hard-coding these values directly in your Python
  files is dangerous (you might accidentally push a password to GitHub) and
  inconvenient (you'd have to edit code just to change a dollar limit).
  The conventional solution is a "config" or "settings" module that reads values
  from the *environment* — a set of named variables the operating system makes
  available to your program. A .env file is just a text file of KEY=VALUE pairs
  that gets loaded into the environment at startup.

WHAT IS A .env FILE?
  A plain text file (never committed to version control) that looks like:
      KALSHI_API_KEY_ID=abc123
      MAX_BET_AMOUNT_CENTS=1000
  The python-dotenv library reads this file and injects every line into
  os.environ so that os.getenv("KALSHI_API_KEY_ID") returns "abc123".
  This keeps secrets out of your source code.

WHAT IS PYDANTIC?
  Pydantic is a Python library that lets you define data structures with type
  annotations and automatic validation. If you declare a field as `int` and
  someone passes "hello", Pydantic raises a clear error instead of silently
  breaking later. Here, `Config` inherits from `BaseModel`, giving all its
  fields automatic type-checking and a nice repr for debugging.

WHAT IS A SINGLETON?
  At the bottom of this file: `config = Config()`. This creates exactly ONE
  instance of Config when the module is first imported. Every other module
  does `from bot.config import config` and gets that same object — never
  creating a second one. This pattern (the "singleton") ensures all parts of
  the bot always see the same settings.

WHAT IS A PEM FILE?
  PEM (Privacy Enhanced Mail, despite the misleading name) is a format for
  storing cryptographic keys as Base64-encoded text. Kalshi uses RSA private
  keys for API authentication — you sign each request with your private key,
  and Kalshi verifies it using your public key on file. The PEM file or the
  inline KALSHI_PRIVATE_KEY env var is that private key.

WHAT IS A KELLY FRACTION?
  The Kelly Criterion (covered in depth in arbitrage.py and risk_manager.py)
  tells you what fraction of your bankroll to bet given your edge and odds.
  A kelly_fraction of 1.0 = "full Kelly" (mathematically optimal but
  psychologically brutal due to variance). Real traders typically use a
  fraction: 0.25 = quarter-Kelly, 0.35 = third-Kelly. Lower fractions mean
  smaller bets and slower growth, but far less risk of catastrophic loss.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

# python-dotenv: loads KEY=VALUE pairs from a .env file into os.environ.
# override=True means .env values take priority over already-set env vars.
from dotenv import load_dotenv

# Pydantic BaseModel: a class whose attributes are auto-validated by type.
# If validation fails, Pydantic raises a clear error at startup rather than
# a confusing error deep inside the bot logic later.
from pydantic import BaseModel

# Load .env file into the environment. This must happen before any os.getenv
# call so that all the variables are available when Config is instantiated.
load_dotenv(override=True)


def _safe_int(key: str, default: int) -> int:
    """Parse an env var as int, falling back to default on invalid values.

    WHY THIS EXISTS: os.getenv always returns a string (or None). If someone
    sets MAX_BET_AMOUNT_CENTS=ten_dollars in their .env, int("ten_dollars")
    would raise a ValueError and crash the bot at startup. This wrapper
    catches that and uses the sensible default instead, keeping the bot alive
    while making debugging easier (the bad value is silently ignored here;
    a log message would make it even more robust).
    """
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _safe_float(key: str, default: float) -> float:
    """Parse an env var as float, falling back to default on invalid values.

    Same rationale as _safe_int above. Floats are used for percentages and
    probabilities (0.0 – 1.0) and for the Kelly fraction.
    """
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


class Config(BaseModel):
    """Central configuration object loaded from environment variables (.env file).

    All fields have sensible defaults so the bot can start with minimal config.
    Only KALSHI_API_KEY_ID and a private key are strictly required for trading.

    WHY A CLASS INSTEAD OF JUST os.getenv() EVERYWHERE?
    Centralising all settings in one place means:
      - You can see every tunable parameter at a glance.
      - Type checking catches typos early (e.g., accidentally passing a string
        where a float is expected).
      - Other modules import `config` and reference `config.max_bet_amount_cents`
        — a readable name — rather than scattering magic strings like
        "MAX_BET_AMOUNT_CENTS" throughout the codebase.
    """

    # ── Kalshi API Credentials ────────────────────────────────────────────────
    # WHAT IS AN API KEY? An API (Application Programming Interface) is a set
    # of rules that lets two programs talk to each other. Kalshi exposes an API
    # so programs like this bot can place orders, check balances, and read market
    # data without logging into a website. An "API key" is like a username+password
    # combined into one string that identifies your account to Kalshi's servers.
    # Unlike a password, Kalshi uses asymmetric cryptography: you have a *private*
    # key (secret, stays on your machine) and a *public* key (registered with
    # Kalshi). You sign each API request with the private key; Kalshi verifies it.
    kalshi_api_key_id: str = os.getenv("KALSHI_API_KEY_ID", "")
    kalshi_private_key_path: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
    kalshi_private_key_raw: str = os.getenv("KALSHI_PRIVATE_KEY", "")  # PEM string (alternative to file path)
    kalshi_use_demo: bool = os.getenv("KALSHI_USE_DEMO", "true").lower() == "true"  # Demo vs production API
    # DEMO vs PRODUCTION: Kalshi provides a demo environment (fake money) for
    # testing. Setting KALSHI_USE_DEMO=true sends all API calls to the demo
    # server so you can experiment without risking real money. ALWAYS test in
    # demo mode first before switching to production.

    # ── Anthropic API (Claude AI Analysis) ────────────────────────────────────
    # WHAT IS THE ANTHROPIC API? Anthropic makes Claude, an AI assistant. This
    # bot can optionally send market data to Claude and ask it to evaluate whether
    # a market looks mispriced. The Anthropic API key authorises those requests.
    # Claude AI analysis is used by the CLI (bot.main), not the RF model pipeline.
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ── Supabase Database ─────────────────────────────────────────────────────
    # WHAT IS SUPABASE? Supabase is a hosted PostgreSQL database with a REST API.
    # The bot uses it to persist trade history, model training data, and performance
    # metrics across restarts. Without Supabase configured, the bot still works but
    # loses historical data when restarted. The "service role key" bypasses row-level
    # security — it's essentially a superuser password; keep it secret.
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")  # Supabase service role key

    # ── Trading Parameters ────────────────────────────────────────────────────
    # WHAT ARE CENTS? Kalshi uses whole cents as its currency unit internally.
    # $10.00 = 1000 cents. Working in integers (cents) avoids floating-point
    # rounding errors that could cause a $10.001 order to be rejected.
    max_bet_amount_cents: int = _safe_int("MAX_BET_AMOUNT_CENTS", 1000)     # $10 max per RF trade (hard floor)

    # MAX BET PCT: An alternative size cap as a percentage of your current
    # account balance. If your balance is $200, a 15% cap means no single
    # bet can exceed $30. This scales your risk proportionally as your
    # account grows or shrinks — a key principle in money management.
    max_bet_pct: float = _safe_float("MAX_BET_PCT", 0.15)                   # Max bet as % of current balance (proportional scaling)

    # WHAT IS AN "EDGE"? Edge = (your estimated true probability) - (market price).
    # If you think an event has a 60% chance of happening but the market prices
    # it at 45%, your edge is +15%. A positive edge means the bet is in your favour
    # on average. min_edge_threshold = 0.10 means the bot only trades when it
    # believes it has at least a 10% edge — filtering out marginal opportunities
    # where transaction costs and estimation errors could easily erode the profit.
    min_edge_threshold: float = _safe_float("MIN_EDGE_THRESHOLD", 0.10)     # 10% minimum edge — only real edges

    # DAILY LOSS LIMIT: The maximum dollar loss the bot will accept in one day.
    # Once this threshold is crossed, all trading stops until the next day. This
    # is a critical safety guard — without it, a misbehaving model or an
    # unexpected market event could quickly wipe out your account. $75 = 7500 cents.
    max_daily_loss_cents: int = _safe_int("MAX_DAILY_LOSS_CENTS", 7500)    # $75 daily loss limit

    # MAX OPEN POSITIONS: Limits how many separate market bets can be active at
    # once. Spreading across too many markets at once ties up capital and makes
    # the portfolio hard to monitor. This acts as a concentration guard alongside
    # max_positions_per_category below.
    max_open_positions: int = _safe_int("MAX_OPEN_POSITIONS", 15)            # 15 max (crypto + RF)

    # MAX EVENTS TO ANALYZE: Kalshi lists thousands of events. Analyzing all of
    # them with an AI or ML model is slow and expensive. This cap lets the bot
    # focus on the most promising subset (sorted by volume or recency) to keep
    # each scan fast.
    max_events_to_analyze: int = _safe_int("MAX_EVENTS_TO_ANALYZE", 50000)   # Events per on-demand scan

    # ENTRY THRESHOLD: Similar to min_edge_threshold but specifically for the
    # "undervalued" check — a market must be priced at least 10% below the
    # model's fair value before the bot considers it worth trading. This
    # ensures the bot only acts on meaningful mispricings, not noise.
    entry_threshold: float = _safe_float("ENTRY_THRESHOLD", 0.10)           # 10% undervalued — real mispricing only

    # MIN CONFIDENCE: The model must be at least 75% confident in its signal
    # before a trade is considered. Confidence here is a probability output
    # from the Random Forest model — it measures how consistently the trees
    # in the ensemble agree with each other. Low confidence means the trees
    # disagree, which suggests the signal is unreliable.
    min_confidence: float = _safe_float("MIN_CONFIDENCE", 0.75)             # 75%+ model confidence required

    # ── Kelly Criterion ───────────────────────────────────────────────────────
    # WHAT IS THE KELLY CRITERION?
    # The Kelly Criterion is a mathematical formula (published by J.L. Kelly Jr.
    # in 1956) for sizing bets to maximise the *long-run growth rate* of your
    # bankroll. The formula is: f* = (b*p - q) / b
    #   where p = probability of winning, q = 1 - p, b = net odds (profit/stake).
    #   f* = the fraction of your bankroll to bet.
    #
    # Full Kelly (kelly_fraction = 1.0) is theoretically optimal but produces
    # severe drawdowns in practice. Most professional traders use a fraction:
    #   0.25 (quarter-Kelly) — very conservative, slow but steady growth
    #   0.35 (third-Kelly)   — moderate, still much safer than full Kelly
    #   0.50 (half-Kelly)    — roughly half the variance of full Kelly
    #
    # The min(..., max(...)) clamps the value to [0.01, 1.0] so a misconfigured
    # env var can't set it below 1% or above 100%.
    kelly_fraction: float = min(1.0, max(0.01, _safe_float("KELLY_FRACTION", 0.35)))  # Third-Kelly — meaningful position sizing

    # ── Correlation & Drawdown Control ─────────────────────────────────────
    # WHY LIMIT POSITIONS PER CATEGORY? If you hold 10 crypto markets and crypto
    # crashes, all 10 positions lose simultaneously. This is "correlation risk"
    # — bets that look independent but move together. Limiting to 3 positions per
    # category (crypto, politics, sports, etc.) forces diversification.
    max_positions_per_category: int = _safe_int("MAX_POSITIONS_PER_CATEGORY", 3)  # Category concentration limit

    # MAX DRAWDOWN PCT: The maximum allowable decline in account equity *within
    # a single day* as a fraction (0.05 = 5%). If the account drops more than
    # this, trading halts. A "drawdown" is the decline from a recent peak —
    # a 5% intraday drawdown cap means "stop trading if we're down more than
    # 5% from today's opening balance." This protects against catastrophic
    # loss from a model failure or flash crash.
    max_drawdown_pct: float = _safe_float("MAX_DRAWDOWN_PCT", 0.05)  # 5% equity daily loss cap

    # ── Risk & Cost Parameters ───────────────────────────────────────────
    # RISK-FREE RATE: The return you could get with zero risk (e.g., a Treasury
    # bill or savings account). Used in the Sharpe ratio calculation:
    #   Sharpe = (strategy return - risk-free return) / standard deviation
    # A Sharpe ratio above 1.0 is considered good; it means you're earning more
    # than 1 unit of return per unit of risk. The daily rate here is the annual
    # rate (5%) divided by 365 — because the bot tracks daily returns.
    risk_free_rate: float = _safe_float("RISK_FREE_RATE", 0.05 / 365)  # Daily risk-free rate for Sharpe calculation

    # SLIPPAGE: The difference between the price you *expect* to pay and the
    # price you *actually* pay when your order fills. In a thin (illiquid)
    # market, your limit order might not fill at the midpoint; you may have
    # to pay a cent or two more. Slippage is modelled as a cost in backtests
    # to make simulated results more realistic than assuming perfect fills.
    slippage_cents: int = _safe_int("SLIPPAGE_CENTS", 1)  # Default slippage per trade in cents

    # COMMISSION: Kalshi charges no trading commission as of this writing, so
    # this defaults to 0. It's here as a parameter so the backtester can model
    # a realistic cost structure if that changes or if the bot is adapted for
    # a platform that does charge commissions.
    commission_cents: int = _safe_int("COMMISSION_CENTS", 0)  # Default commission per trade in cents

    # MAX CATEGORY EXPOSURE PCT: Even with per-category position limits, you
    # could have 3 large positions in crypto that together represent 80% of
    # your total portfolio. This cap limits any single category to 40% of the
    # total portfolio value, providing a second layer of concentration control.
    max_category_exposure_pct: float = _safe_float("MAX_CATEGORY_EXPOSURE_PCT", 0.40)  # Max category exposure cap

    # MAX POSITION SCALE: Controls whether the bot is allowed to add to an
    # existing position (scale in). A value of 1 means "no scaling" — one
    # entry per market, period. Scaling into winning positions is a more
    # advanced strategy that requires careful implementation to avoid
    # over-concentrating in one market.
    max_position_scale: int = _safe_int("MAX_POSITION_SCALE", 1)  # NO scaling — one entry per market

    # ── Notification Webhooks ─────────────────────────────────────────────────
    # WHAT IS A WEBHOOK? A webhook is a URL that accepts POST requests with a
    # JSON payload. Slack and Discord both let you create "incoming webhooks" —
    # special URLs that, when you POST a message to them, display that message
    # in a channel. The bot uses these to send real-time trade alerts to your
    # phone or desktop without you having to watch a terminal window.
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    discord_webhook_url: str = os.getenv("DISCORD_WEBHOOK_URL", "")

    # ── Email Notifications (SMTP) ────────────────────────────────────────────
    # WHAT IS SMTP? Simple Mail Transfer Protocol is the standard protocol for
    # sending email. Most email providers (Gmail, Outlook, etc.) let you send
    # email programmatically by connecting to their SMTP server with a username
    # and password. Gmail uses smtp.gmail.com on port 587 (STARTTLS) or 465
    # (SSL). You'll typically need to generate an "app password" rather than
    # using your main Google account password.
    notification_email: str = os.getenv("NOTIFICATION_EMAIL", "")  # Recipient email
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = _safe_int("SMTP_PORT", 587)
    smtp_user: str = os.getenv("SMTP_USER", "")  # SMTP login username
    smtp_pass: str = os.getenv("SMTP_PASS", "")  # SMTP login password / app password
    smtp_from: str = os.getenv("SMTP_FROM", "")   # From address (defaults to smtp_user)

    # ── Inbound Webhook Authentication ────────────────────────────────────────
    # The bot's web server exposes a /api/webhook endpoint that can trigger
    # scans or trades when called by an external service (e.g., a TradingView
    # alert). The webhook_secret is a shared password: the caller must include
    # it in each request so the bot knows the request is legitimate and not
    # from a random attacker on the internet.
    webhook_secret: str = os.getenv("WEBHOOK_SECRET", "")  # Shared secret for /api/webhook

    # ── Model Retrain Schedule ────────────────────────────────────────────────
    # WHY RETRAIN? The Random Forest model learns from historical trade data.
    # As new data accumulates and market conditions shift, the model's predictions
    # can become stale. Periodic retraining (here, every Mon/Wed/Fri at 3 AM UTC)
    # keeps it current. APScheduler is the library that runs code on a schedule —
    # "day_of_week=mon,wed,fri" and "hour=3" are its cron-style syntax.
    retrain_days: str = os.getenv("RETRAIN_DAYS", "mon,wed,fri")  # APScheduler cron day_of_week
    retrain_hour: int = min(23, max(0, _safe_int("RETRAIN_HOUR", 3)))  # Hour in UTC, clamped [0, 23]
    # The min(23, max(0, ...)) clamp ensures the hour stays in [0, 23] even if
    # someone accidentally sets RETRAIN_HOUR=99 in their .env file.

    # ── Dynamic Confidence Threshold ───────────────────────────────────────────
    # When enabled, the bot adjusts the minimum confidence threshold dynamically
    # based on recent performance. If the model has been accurate lately, it
    # may lower the bar slightly (taking more trades). If accuracy has been poor,
    # it raises the bar (becoming more selective). This is adaptive risk management.
    dynamic_confidence_enabled: bool = os.getenv("DYNAMIC_CONFIDENCE_ENABLED", "true").lower() == "true"

    # ── Auto-Scan on Boot ────────────────────────────────────────────────────
    # If auto_scan_on_boot is True, the bot immediately starts scanning for
    # trading opportunities when the server starts, without waiting for a manual
    # trigger. This is convenient in production but you may want to disable it
    # while developing to avoid accidental trades.
    auto_scan_on_boot: bool = os.getenv("AUTO_SCAN_ON_BOOT", "true").lower() == "true"

    # AUTO_SCAN_INTERVAL: How often (in seconds) the background loop rescans
    # Kalshi for new signals. 120 seconds = 2 minutes. The crypto strategy uses
    # 1-second intervals for near-realtime responsiveness to price moves.
    auto_scan_interval: float = _safe_float("AUTO_SCAN_INTERVAL", 120)          # RF eval interval (seconds) — 2min between scans

    # AUTO_FETCH_INTERVAL: How often (in seconds) the bot refreshes raw market
    # data from the Kalshi API. Shorter = fresher data but more API calls.
    # Kalshi may rate-limit excessive requests.
    auto_fetch_interval: int = _safe_int("AUTO_FETCH_INTERVAL", 60)             # Kalshi API fetch interval (seconds)

    @property
    def kalshi_base_url(self) -> str:
        """Return the Kalshi API base URL based on the demo/production toggle.

        WHAT IS A BASE URL? All Kalshi API endpoints start with a common prefix.
        Rather than hard-coding this prefix everywhere, we centralise it here
        and switch between demo and production URLs based on the config flag.
        Every API call in kalshi_client.py appends a path like /markets to this
        base URL to form the full endpoint address.
        """
        if self.kalshi_use_demo:
            # Demo server: uses fake money, safe for testing and development
            return "https://demo-api.kalshi.co/trade-api/v2"
        # Production server: real money, real markets
        return "https://api.elections.kalshi.com/trade-api/v2"

    @property
    def kalshi_private_key(self) -> str:
        """Resolve the Kalshi private key PEM content.

        Priority: KALSHI_PRIVATE_KEY env var (inline PEM string)
        > KALSHI_PRIVATE_KEY_PATH file (path to PEM file on disk).
        Returns empty string if neither is available.

        WHY TWO OPTIONS? Different deployment environments have different
        constraints. On your local machine, storing the key as a file is
        convenient. In a cloud environment (Docker, Heroku, etc.), it's
        easier and safer to pass secrets as environment variables rather
        than bundling key files into a container image.

        The @property decorator makes this look like a plain attribute
        (`config.kalshi_private_key`) even though it runs code each time
        it's accessed. This is a Python pattern for "computed attributes."
        """
        if self.kalshi_private_key_raw:
            # Prefer the inline env var: no file I/O needed
            return self.kalshi_private_key_raw
        path = Path(self.kalshi_private_key_path)
        if path.exists():
            # Fall back to reading the PEM file from disk
            return path.read_text()
        # Neither configured — return empty string; validate_kalshi() will catch this
        return ""

    def validate_kalshi(self) -> bool:
        """Check if Kalshi API credentials are configured (key ID + private key).

        Called at startup to give a clear error message if credentials are
        missing, rather than letting the bot crash with a cryptic KeyError
        later when it first tries to call the Kalshi API.
        """
        return bool(self.kalshi_api_key_id and self.kalshi_private_key)

    def validate_anthropic(self) -> bool:
        """Check if the Anthropic API key is configured for Claude AI analysis.

        Claude analysis (bot.analyzer) is optional — the RF model can run
        without it. But the CLI dry-run mode requires it, so this is checked
        before launching the analyzer.
        """
        return bool(self.anthropic_api_key)

    def validate_supabase(self) -> bool:
        """Check if Supabase URL and key are configured for database persistence.

        Supabase is optional. Without it, the bot trades normally but can't
        persist historical data or load pre-trained model weights across restarts.
        """
        return bool(self.supabase_url and self.supabase_key)


# ── Singleton instantiation ───────────────────────────────────────────────────
# This line runs once when Python first imports this module. Every other module
# does `from bot.config import config` and gets this exact object. Because Python
# caches module imports, this line never runs a second time — that is what makes
# it a singleton. All parts of the bot share one Config instance with consistent
# values throughout the lifetime of the process.
config = Config()
