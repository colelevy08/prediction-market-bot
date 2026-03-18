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
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from bot.config import config
from bot.kalshi_client import KalshiClient
from bot.draftkings_client import DraftKingsClient
from bot.analyzer import MarketAnalyzer
from bot.risk_manager import RiskManager
from bot.arbitrage import detect_arbitrage

console = Console()  # Rich console for formatted terminal output


def print_banner():
    """Display the bot's ASCII banner with name and description."""
    console.print(Panel.fit(
        "[bold cyan]Prediction Market Trading Bot[/]\n"
        "Kalshi + DraftKings | AI-Powered Analysis",
        border_style="cyan",
    ))


def print_signals(signals, risk_manager, portfolio):
    """Display trading signals in a formatted Rich table with risk check status.

    Each signal shows: ticker, market name, side, edge, confidence, fair/market
    probabilities, recommended size, and whether it passes the risk manager checks.
    """
    if not signals:
        console.print("[yellow]No actionable signals found.[/]")
        return

    table = Table(title="Trading Signals", show_lines=True)
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
        allowed, reason = risk_manager.check_signal(sig, portfolio)
        side_color = "green" if sig.side.value == "yes" else "red"
        status = "[green]READY[/]" if allowed else f"[red]{reason}[/]"

        table.add_row(
            sig.ticker,
            sig.market_title[:40],
            f"[{side_color}]{sig.side.value.upper()}[/]",
            f"{sig.edge:+.1%}",
            f"{sig.confidence:.0%}",
            f"{sig.fair_probability:.0%}",
            f"{sig.market_probability:.0%}",
            f"${sig.recommended_size_cents / 100:.2f}",
            status,
        )
    console.print(table)


def print_portfolio(portfolio):
    """Display the current portfolio balance and open positions in Rich tables."""
    table = Table(title="Portfolio")
    table.add_column("Balance", style="green")
    table.add_column("Positions")
    table.add_row(
        f"${portfolio.balance_cents / 100:.2f}",
        str(len(portfolio.positions)),
    )
    console.print(table)

    if portfolio.positions:
        pos_table = Table(title="Open Positions", show_lines=True)
        pos_table.add_column("Ticker", style="cyan")
        pos_table.add_column("Side")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Avg Price", justify="right")
        for pos in portfolio.positions:
            pos_table.add_row(
                pos.ticker,
                pos.side.upper(),
                str(pos.quantity),
                f"{pos.avg_price_cents}c",
            )
        console.print(pos_table)


def run_scan(kalshi: KalshiClient, analyzer: MarketAnalyzer, risk_manager: RiskManager, live: bool):
    """Main scan loop: fetch markets from Kalshi, analyze with Claude AI, display signals.

    In dry run mode (live=False), only displays signals without placing orders.
    In live mode (live=True), executes trades that pass the risk manager checks,
    refreshing the portfolio after each successful order.
    """
    console.print("\n[bold]Fetching events from Kalshi...[/]")
    events = kalshi.get_all_events()
    console.print(f"  Found [cyan]{len(events)}[/] events with [cyan]{sum(len(e.markets) for e in events)}[/] markets")

    console.print("\n[bold]Analyzing markets with Claude...[/]")
    signals = analyzer.analyze_events(events)
    console.print(f"  Generated [cyan]{len(signals)}[/] actionable signals")

    portfolio = kalshi.get_portfolio_summary()
    print_portfolio(portfolio)
    print_signals(signals, risk_manager, portfolio)

    if not live:
        console.print("\n[yellow]DRY RUN — no orders placed. Use --live to execute trades.[/]")
        return

    # Execute trades
    for sig in signals:
        allowed, reason = risk_manager.check_signal(sig, portfolio)
        if not allowed:
            continue

        order = risk_manager.build_order(sig)
        console.print(f"\n  Placing order: [cyan]{order.ticker}[/] {order.side.value.upper()} x{order.count} @ {order.price_cents}c")

        try:
            result = kalshi.place_order(order)
            console.print(f"  [green]Order placed![/] ID: {result.order_id}, Status: {result.status}")
            risk_manager.record_trade()
            # Refresh portfolio after each trade
            portfolio = kalshi.get_portfolio_summary()
        except Exception as e:
            console.print(f"  [red]Order failed:[/] {e}")


def run_arbitrage_scan(kalshi: KalshiClient, dk: DraftKingsClient):
    """Scan for cross-platform arbitrage opportunities between Kalshi and DraftKings.

    Fetches open markets from both platforms, runs the arbitrage detection algorithm,
    and displays any opportunities in a formatted Rich table.
    """
    console.print("\n[bold]Scanning for cross-platform arbitrage...[/]")

    events = kalshi.get_all_events()
    kalshi_markets = [m for e in events for m in e.markets if m.status == "open"]
    dk_markets = dk.get_prediction_markets()

    console.print(f"  Kalshi: {len(kalshi_markets)} markets | DraftKings: {len(dk_markets)} markets")

    opportunities = detect_arbitrage(kalshi_markets, dk_markets)

    if not opportunities:
        console.print("[yellow]No arbitrage opportunities detected.[/]")
        return

    table = Table(title="Arbitrage Opportunities", show_lines=True)
    table.add_column("Market", max_width=40)
    table.add_column("Kalshi YES", justify="right")
    table.add_column("DK YES", justify="right")
    table.add_column("Spread", justify="right", style="green")
    table.add_column("Action", max_width=50)

    for opp in opportunities:
        table.add_row(
            opp.title[:40],
            f"{opp.kalshi_yes_price:.0%}",
            f"{opp.dk_yes_price:.0%}",
            f"{opp.spread_pct:.1%}",
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
    """
    parser = argparse.ArgumentParser(description="Prediction Market Trading Bot")
    parser.add_argument("--live", action="store_true", help="Execute real trades (default: dry run)")
    parser.add_argument("--arbitrage", action="store_true", help="Scan for cross-platform arbitrage")
    parser.add_argument("--portfolio", action="store_true", help="Show portfolio only")
    args = parser.parse_args()

    print_banner()

    # Validate configuration and display environment (DEMO vs PRODUCTION)
    env_label = "[yellow]DEMO[/]" if config.kalshi_use_demo else "[red]PRODUCTION[/]"
    console.print(f"Environment: {env_label}")

    if not config.validate_kalshi():
        console.print("[red]Missing Kalshi API credentials. Check .env file.[/]")
        sys.exit(1)

    kalshi = KalshiClient()
    risk_manager = RiskManager()

    try:
        if args.portfolio:
            portfolio = kalshi.get_portfolio_summary()
            print_portfolio(portfolio)
            return

        if args.arbitrage:
            dk = DraftKingsClient()
            run_arbitrage_scan(kalshi, dk)
            dk.close()
            return

        if not config.validate_anthropic():
            console.print("[red]Missing Anthropic API key. Required for market analysis.[/]")
            sys.exit(1)

        analyzer = MarketAnalyzer()
        run_scan(kalshi, analyzer, risk_manager, live=args.live)

    finally:
        kalshi.close()


if __name__ == "__main__":
    main()
