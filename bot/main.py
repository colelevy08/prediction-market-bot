"""
CLI entry point for the prediction market trading bot.

Provides a command-line interface for running the bot in three modes:

  1. Default (dry run): Fetches markets from Kalshi, analyzes with Claude AI,
     displays signals in a rich terminal table, but does NOT place any orders.
     Use this to preview what the bot would do.

  2. --live: Same as default but actually executes trades on Kalshi. Each signal
     passes through the RiskManager before order placement. Portfolio is refreshed
     after each trade.

  3. --arbitrage: Scans for cross-platform price discrepancies between Kalshi
     and DraftKings, displaying opportunities in a terminal table.

  4. --portfolio: Simply displays current Kalshi portfolio (balance + positions).

The CLI uses the Rich library for formatted terminal output (tables, panels, colors).
It validates configuration (Kalshi credentials, Anthropic API key) before proceeding
and displays the current environment (DEMO vs PRODUCTION).

Note: The CLI uses Claude AI analysis (bot.analyzer), NOT the RF ensemble model.
The RF model is used by the server (bot.server) and paper trader (bot.backtester).

Connects to: bot.config (validation), bot.kalshi_client (market data + orders),
bot.analyzer (Claude AI signals), bot.risk_manager (trade validation),
bot.draftkings_client + bot.arbitrage (cross-platform scan).

Usage:
  python -m bot.main                  # Dry run with AI analysis
  python -m bot.main --live           # Execute real trades
  python -m bot.main --arbitrage      # Cross-platform arbitrage scan
  python -m bot.main --portfolio      # Show portfolio only

---------------------------------------------------------------------------
EDUCATIONAL OVERVIEW
---------------------------------------------------------------------------

WHAT IS A CLI (COMMAND-LINE INTERFACE)?
  A CLI is a text-based way to control a program by typing commands in a
  terminal rather than clicking buttons in a graphical interface. You run
  this bot by typing `python -m bot.main` in your terminal. Adding flags
  like `--live` or `--arbitrage` changes what the bot does. CLIs are the
  standard tool for server-side programs and automation scripts — they're
  scriptable, lightweight, and work over SSH connections.

WHAT IS argparse?
  argparse is Python's built-in library for parsing command-line arguments.
  You define what flags your program accepts (--live, --arbitrage, etc.)
  and argparse reads sys.argv (the list of words you typed in the terminal),
  matches them to your definitions, and gives you a nice object with the
  results. It also auto-generates a --help message.

WHAT IS THE Rich LIBRARY?
  Rich is a Python library for beautiful terminal output. Instead of plain
  print() statements, Rich lets you display coloured text, formatted tables,
  progress bars, and panel boxes. The Table class shown in print_signals()
  renders a grid with borders and column alignment. The Panel class creates
  a bordered box (used for the startup banner). The Console class is Rich's
  main entry point — all output goes through it.

WHAT IS A DRY RUN?
  A "dry run" (or "paper run") means the bot goes through the full analysis
  pipeline — fetching data, generating signals, checking risk limits — but
  does NOT actually place any orders. It's like a rehearsal. Dry runs are
  invaluable for testing: you can verify the bot's logic and see what it would
  do without risking real money. The --live flag turns off the dry run safety
  and enables real order placement.

WHAT IS THE RISKMANAGER?
  The RiskManager (bot.risk_manager) is a gatekeeper between signals and orders.
  Before any signal becomes a real trade, the RiskManager checks:
    - Is there enough edge? (>= min_edge_threshold)
    - Is the model confident enough? (>= min_confidence)
    - Would this trade exceed the daily loss limit?
    - Is the max open positions limit already reached?
    - Is there already too much exposure in this market category?
  Only signals that pass ALL of these checks become orders.

HOW DOES THE MAIN LOOP WORK?
  run_scan() is the heart of the CLI:
    1. Fetch all events from Kalshi (could be thousands of markets)
    2. Send them to Claude AI for analysis → list of TradingSignals
    3. Fetch the current portfolio to know the balance and open positions
    4. Display the signals and portfolio in the terminal
    5. In --live mode: for each signal, check with RiskManager, build the
       order, place it on Kalshi, and refresh the portfolio

  The `finally: kalshi.close()` at the end of main() ensures the HTTP
  connection to Kalshi is always properly closed, even if an exception
  occurs partway through. This is called the "try/finally" pattern for
  resource cleanup.

WHAT IS __name__ == "__main__"?
  Every Python file has a special variable `__name__`. When you run a file
  directly (python main.py), Python sets __name__ to "__main__". When a
  file is imported by another module, __name__ is set to the module's name
  (e.g., "bot.main"). The `if __name__ == "__main__": main()` guard means
  "only call main() when this file is run directly, not when it's imported."
  Without this guard, importing bot.main from another module would immediately
  run the CLI, which would be very surprising and probably crash.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse   # Standard library: parses command-line arguments (--live, --arbitrage, etc.)
import sys        # Standard library: provides sys.exit() and sys.argv

# Rich library imports for styled terminal output
from rich.console import Console   # Main output class — all printing goes through this
from rich.table import Table       # Renders a grid/table in the terminal with borders
from rich.panel import Panel       # A bordered box for the startup banner

from bot.config import config
from bot.kalshi_client import KalshiClient
from bot.draftkings_client import DraftKingsClient
from bot.analyzer import MarketAnalyzer
from bot.risk_manager import RiskManager
from bot.arbitrage import detect_arbitrage

# A single shared Console instance for all terminal output.
# Using one Console (rather than calling print() directly) ensures consistent
# formatting and allows Rich to manage terminal state (e.g., live tables).
console = Console()  # Rich console for formatted terminal output


def print_banner():
    """Display the bot's ASCII banner with name and description.

    Panel.fit() creates a bordered box that auto-sizes to its content.
    The [bold cyan]...[/] syntax is Rich's markup language — similar to HTML tags,
    it applies styles to the enclosed text. "bold" = bold text, "cyan" = cyan colour.
    """
    console.print(Panel.fit(
        "[bold cyan]Prediction Market Trading Bot[/]\n"
        "Kalshi + DraftKings | AI-Powered Analysis",
        border_style="cyan",
    ))


def print_signals(signals, risk_manager, portfolio):
    """Display trading signals in a formatted Rich table with risk check status.

    Each signal shows: ticker, market name, side, edge, confidence, fair/market
    probabilities, recommended size, and whether it passes the risk manager checks.

    READING THE TABLE:
    - Edge: How mispriced the model believes the market to be. +15.0% means the
            model thinks fair value is 15 cents higher than the current market price.
    - Confidence: How certain the model is in its estimate (0-100%).
    - Fair Prob: The model's estimated true probability of the event.
    - Mkt Prob: The current market-implied probability (the price we'd pay).
    - Status: READY = passes all risk checks. Otherwise shows the reason it's blocked.

    Args:
        signals: List of TradingSignal objects from the analyzer.
        risk_manager: RiskManager instance for checking each signal's eligibility.
        portfolio: Current PortfolioSummary from Kalshi (needed for risk checks).
    """
    if not signals:
        # [yellow]...[/] renders the text in yellow — a gentle warning colour
        console.print("[yellow]No actionable signals found.[/]")
        return

    table = Table(title="Trading Signals", show_lines=True)
    # Each add_column call adds a column with an optional style and max_width.
    # style="cyan": all cells in this column render in cyan text.
    # justify="right": numbers right-align (standard for financial data).
    # max_width=40: long market titles are truncated to 40 chars to keep the table narrow.
    table.add_column("Ticker", style="cyan")
    table.add_column("Market", max_width=40)
    table.add_column("Side", style="bold")
    table.add_column("Edge", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Fair Prob", justify="right")
    table.add_column("Mkt Prob", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Status")

    for sig in signals:
        # check_signal returns (True, "ok") or (False, "reason it was blocked")
        allowed, reason = risk_manager.check_signal(sig, portfolio)
        # Colour-code the side: green for YES (optimistic), red for NO (bearish)
        side_color = "green" if sig.side.value == "yes" else "red"
        status = "[green]READY[/]" if allowed else f"[red]{reason}[/]"

        table.add_row(
            sig.ticker,
            sig.market_title[:40],            # Truncate long titles
            f"[{side_color}]{sig.side.value.upper()}[/]",
            f"{sig.edge:+.1%}",               # :+.1% formats as e.g. "+15.0%" (always shows sign)
            f"{sig.confidence:.0%}",           # :.0% formats as e.g. "82%" (no decimal)
            f"{sig.fair_probability:.0%}",
            f"{sig.market_probability:.0%}",
            f"${sig.recommended_size_cents / 100:.2f}",  # Convert cents to dollars
            status,
        )
    console.print(table)


def print_portfolio(portfolio):
    """Display the current portfolio balance and open positions in Rich tables.

    Shows two tables:
    1. A summary table with total balance and number of open positions.
    2. If there are open positions, a detail table with one row per position
       showing ticker, side, quantity, and average entry price.

    Args:
        portfolio: A PortfolioSummary object from KalshiClient.get_portfolio_summary().
    """
    table = Table(title="Portfolio")
    table.add_column("Balance", style="green")   # Balance in green = money = positive
    table.add_column("Positions")
    table.add_row(
        f"${portfolio.balance_cents / 100:.2f}",   # Convert cents to dollars: 1000 → $10.00
        str(len(portfolio.positions)),
    )
    console.print(table)

    # Only show the positions detail table if there are open positions.
    # An empty table would just clutter the output.
    if portfolio.positions:
        pos_table = Table(title="Open Positions", show_lines=True)
        pos_table.add_column("Ticker", style="cyan")
        pos_table.add_column("Side")
        pos_table.add_column("Qty", justify="right")      # Qty = quantity of contracts held
        pos_table.add_column("Avg Price", justify="right")  # Average entry price in cents
        for pos in portfolio.positions:
            pos_table.add_row(
                pos.ticker,
                pos.side.upper(),          # "yes" or "no" → "YES" or "NO"
                str(pos.quantity),
                f"{pos.avg_price_cents}c", # Show price in cents with "c" suffix for clarity
            )
        console.print(pos_table)


def run_scan(kalshi: KalshiClient, analyzer: MarketAnalyzer, risk_manager: RiskManager, live: bool):
    """Main scan loop: fetch markets from Kalshi, analyze with Claude AI, display signals.

    In dry run mode (live=False), only displays signals without placing orders.
    In live mode (live=True), executes trades that pass the risk manager checks,
    refreshing the portfolio after each successful order.

    SCAN PIPELINE:
    1. get_all_events(): Fetches all open events (and their markets) from Kalshi's
       REST API. This may be thousands of markets — the scan respects
       config.max_events_to_analyze to avoid unbounded API usage.
    2. analyzer.analyze_events(): Sends market data to Claude AI and receives
       TradingSignal objects for markets with detected edge. Claude reads the
       market title and current price and estimates whether the market is mispriced.
    3. get_portfolio_summary(): Fetches current balance and open positions.
       This is needed for risk checks (e.g., are we already at max positions?).
    4. Print tables (always).
    5. Execute orders (only in --live mode).

    PORTFOLIO REFRESH AFTER EACH TRADE:
    In live mode, the portfolio is re-fetched after every successful order.
    This ensures that subsequent risk checks use the updated balance and
    position count — preventing the bot from exceeding limits due to stale data.

    Args:
        kalshi: Authenticated KalshiClient instance.
        analyzer: MarketAnalyzer instance (wraps Claude AI).
        risk_manager: RiskManager instance for pre-trade safety checks.
        live: If True, execute real trades. If False, display only (dry run).
    """
    console.print("\n[bold]Fetching events from Kalshi...[/]")
    events = kalshi.get_all_events()
    # Sum the market count across all events for the status message.
    # sum(len(e.markets) for e in events) is a generator expression:
    # it computes len(e.markets) for each event and adds them all up.
    console.print(f"  Found [cyan]{len(events)}[/] events with [cyan]{sum(len(e.markets) for e in events)}[/] markets")

    console.print("\n[bold]Analyzing markets with Claude...[/]")
    signals = analyzer.analyze_events(events)
    console.print(f"  Generated [cyan]{len(signals)}[/] actionable signals")

    portfolio = kalshi.get_portfolio_summary()
    print_portfolio(portfolio)
    print_signals(signals, risk_manager, portfolio)

    if not live:
        # Dry run: show what would have happened but don't actually trade.
        # The [yellow] text is a reminder that no money has moved.
        console.print("\n[yellow]DRY RUN — no orders placed. Use --live to execute trades.[/]")
        return

    # ── Live trading mode ─────────────────────────────────────────────────────
    # Iterate over each signal and attempt to place an order for qualifying ones.
    for sig in signals:
        # Re-check the signal against the current portfolio state.
        # We re-fetch the portfolio after each trade (below), so this check
        # always uses up-to-date balance and position counts.
        allowed, reason = risk_manager.check_signal(sig, portfolio)
        if not allowed:
            # Signal was blocked by a risk check — skip it silently.
            # The reason was already shown in the signals table.
            continue

        # build_order converts a TradingSignal into an OrderRequest,
        # applying Kelly sizing and other risk manager adjustments.
        order = risk_manager.build_order(sig)
        console.print(f"\n  Placing order: [cyan]{order.ticker}[/] {order.side.value.upper()} x{order.count} @ {order.price_cents}c")

        try:
            # Actually send the order to Kalshi's API.
            # This is the point of no return — after this line, real money moves.
            result = kalshi.place_order(order)
            console.print(f"  [green]Order placed![/] ID: {result.order_id}, Status: {result.status}")
            # record_trade() increments the RiskManager's trade count for the
            # current session, used for rate limiting and daily trade caps.
            risk_manager.record_trade()
            # Refresh portfolio so the next iteration's risk check has accurate data.
            # Without this, the bot wouldn't know about the position it just opened.
            portfolio = kalshi.get_portfolio_summary()
        except Exception as e:
            # Order placement failed (e.g., Kalshi rejected it, network error).
            # Log the error and continue to the next signal rather than crashing.
            console.print(f"  [red]Order failed:[/] {e}")


def run_arbitrage_scan(kalshi: KalshiClient, dk: DraftKingsClient):
    """Scan for cross-platform arbitrage opportunities between Kalshi and DraftKings.

    Fetches open markets from both platforms, runs the arbitrage detection algorithm,
    and displays any opportunities in a formatted Rich table.

    WHAT IS DISPLAYED?
    For each opportunity found:
    - Market: The event title (shared/matched between Kalshi and DK)
    - Kalshi YES: Current YES probability on Kalshi (e.g., "40%")
    - DK YES: Current YES probability on DraftKings (e.g., "50%")
    - Spread: The absolute price difference (e.g., "10.0%")
    - Action: Which platform to buy/sell on (e.g., "Buy YES on Kalshi @ 40%, sell YES on DK @ 50%")

    LIMITATIONS (reminder):
    The DraftKings scraper often returns incomplete data (no live prices), so
    the table may be empty even if real discrepancies exist. This feature requires
    manual verification and human execution on the DraftKings side.

    Args:
        kalshi: Authenticated KalshiClient instance.
        dk: DraftKingsClient instance (scraper, no auth required).
    """
    console.print("\n[bold]Scanning for cross-platform arbitrage...[/]")

    events = kalshi.get_all_events()
    # Flatten events into a list of individual Market objects, filtering to only
    # "open" markets (not "closed" or "settled" ones which can't be traded).
    # This is a list comprehension with a nested loop and a filter condition:
    #   for e in events: for each event
    #   for m in e.markets: for each market in that event
    #   if m.status == "open": only include open markets
    kalshi_markets = [m for e in events for m in e.markets if m.status == "open"]
    dk_markets = dk.get_prediction_markets()

    console.print(f"  Kalshi: {len(kalshi_markets)} markets | DraftKings: {len(dk_markets)} markets")

    # Run the arbitrage detection algorithm (defined in bot/arbitrage.py)
    opportunities = detect_arbitrage(kalshi_markets, dk_markets)

    if not opportunities:
        console.print("[yellow]No arbitrage opportunities detected.[/]")
        return

    table = Table(title="Arbitrage Opportunities", show_lines=True)
    table.add_column("Market", max_width=40)
    table.add_column("Kalshi YES", justify="right")
    table.add_column("DK YES", justify="right")
    table.add_column("Spread", justify="right", style="green")  # Green = money to be made
    table.add_column("Action", max_width=50)

    # Opportunities are already sorted by spread descending (largest first)
    # from the detect_arbitrage() function, so the best opportunities appear at top.
    for opp in opportunities:
        table.add_row(
            opp.title[:40],                       # Truncate long titles to 40 chars
            f"{opp.kalshi_yes_price:.0%}",         # :.0% formats 0.40 as "40%"
            f"{opp.dk_yes_price:.0%}",
            f"{opp.spread_pct:.1%}",               # :.1% formats 0.105 as "10.5%"
            opp.recommended_action,
        )
    console.print(table)


def main():
    """CLI entry point: parse arguments, validate config, and run the selected mode.

    Modes:
      - Default (dry run): Fetch + analyze + display signals (no orders placed).
      - --live:            Fetch + analyze + execute real trades on Kalshi.
      - --arbitrage:       Cross-platform arbitrage scan (Kalshi vs DraftKings).
      - --portfolio:       Display current Kalshi portfolio and exit.

    EXECUTION ORDER:
    1. Parse command-line arguments with argparse.
    2. Print the startup banner.
    3. Show whether we're in DEMO or PRODUCTION mode.
    4. Validate Kalshi credentials (always required).
    5. Route to the selected mode (portfolio / arbitrage / main scan).
    6. Always close the Kalshi HTTP client in the finally block.

    WHY sys.exit(1) ON VALIDATION FAILURE?
    sys.exit(1) terminates the program with a non-zero exit code. In Unix/Linux
    convention, exit code 0 = success, any non-zero code = error. This matters
    when the bot is run from shell scripts or CI pipelines: they can check the
    exit code to know if startup succeeded. Printing an error and exiting is much
    friendlier than letting the bot crash with a confusing traceback later.
    """
    # argparse.ArgumentParser: creates a parser for command-line arguments.
    # description= text appears in the auto-generated --help output.
    parser = argparse.ArgumentParser(description="Prediction Market Trading Bot")
    # action="store_true": if the flag is present, the attribute is True; otherwise False.
    # This is the standard pattern for boolean flags (no value needed: just --live).
    parser.add_argument("--live", action="store_true", help="Execute real trades (default: dry run)")
    parser.add_argument("--arbitrage", action="store_true", help="Scan for cross-platform arbitrage")
    parser.add_argument("--portfolio", action="store_true", help="Show portfolio only")
    # parse_args() reads sys.argv, matches flags to the definitions above,
    # and returns an object where args.live, args.arbitrage, args.portfolio are booleans.
    args = parser.parse_args()

    print_banner()

    # Display the current environment prominently so you always know if you're
    # using real money. DEMO is yellow (caution), PRODUCTION is red (danger).
    env_label = "[yellow]DEMO[/]" if config.kalshi_use_demo else "[red]PRODUCTION[/]"
    console.print(f"Environment: {env_label}")

    # Validate Kalshi credentials before doing anything else.
    # A clear error message here is much friendlier than a cryptic 401 Unauthorized
    # error from the Kalshi API later.
    if not config.validate_kalshi():
        console.print("[red]Missing Kalshi API credentials. Check .env file.[/]")
        sys.exit(1)

    # Create the Kalshi client and risk manager — used in all modes
    kalshi = KalshiClient()
    risk_manager = RiskManager()

    try:
        # Mode 1: Portfolio only — fetch and display, then exit immediately
        if args.portfolio:
            portfolio = kalshi.get_portfolio_summary()
            print_portfolio(portfolio)
            return   # Return from main() exits the CLI normally

        # Mode 2: Arbitrage scan — requires DraftKings client in addition to Kalshi
        if args.arbitrage:
            dk = DraftKingsClient()
            run_arbitrage_scan(kalshi, dk)
            # Always close DK client to release connection resources
            dk.close()
            return

        # Mode 3 & 4: Main scan (dry run or live trading) — requires Anthropic key
        if not config.validate_anthropic():
            console.print("[red]Missing Anthropic API key. Required for market analysis.[/]")
            sys.exit(1)

        # MarketAnalyzer wraps the Claude AI API; it needs the Anthropic key to work
        analyzer = MarketAnalyzer()
        # live=args.live: if --live was passed, execute real trades; otherwise dry run
        run_scan(kalshi, analyzer, risk_manager, live=args.live)

    finally:
        # The finally block runs regardless of whether the try block completed
        # normally, raised an exception, or returned early. This guarantees that
        # the Kalshi HTTP client is always closed and connection resources freed.
        kalshi.close()


# Standard Python entry point guard.
# When you run `python -m bot.main`, Python sets __name__ = "__main__" and
# calls main(). When another module imports bot.main, this block is skipped.
if __name__ == "__main__":
    main()
