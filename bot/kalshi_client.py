"""
Kalshi Trade API v2 client with RSA-PSS cryptographic authentication.

================================================================================
WHAT IS THIS FILE?
================================================================================

This file is the bot's "telephone" to the Kalshi prediction market exchange.
Every time the bot needs to look up a market price, check your account balance,
or place a trade, it uses the code in this file to make that request.

Think of Kalshi as a bank. You don't walk in and hand them a note — you use
their official app or website, which talks to their servers. This file IS that
official channel, but written in code so our bot can use it automatically.

================================================================================
WHAT IS AN API? (Application Programming Interface)
================================================================================

An API is a set of rules for how two computer programs talk to each other.
Imagine a restaurant: you (the customer) don't walk into the kitchen — you use
a menu and a waiter. The waiter is the API. You say "I want item #4", the waiter
goes to the kitchen and comes back with your food.

Kalshi's API works the same way:
  - You (this bot) send a "request" saying "give me the current BTC market data"
  - Kalshi's servers process the request and send back a "response" with the data
  - The response is formatted as JSON (explained below)

================================================================================
WHAT IS A REST API?
================================================================================

REST (Representational State Transfer) is the most common style of API on the
internet. It uses standard web verbs you may have heard of:
  - GET    — "Give me data" (like loading a webpage)
  - POST   — "Here's new data, please create/process it" (like submitting a form)
  - DELETE — "Remove this thing"

Each verb targets a "path" (URL ending), like:
  - GET  /events       → "give me a list of all events"
  - POST /portfolio/orders  → "place a new order"
  - DELETE /portfolio/orders/abc123 → "cancel order abc123"

Kalshi's full URL looks like: https://api.elections.kalshi.com/trade-api/v2/events

================================================================================
WHAT IS JSON?
================================================================================

JSON (JavaScript Object Notation) is a text format for sending structured data.
It looks like a Python dictionary. Example of what Kalshi sends back:

    {
        "ticker": "KXBTC15M-26MAR230915-B87500",
        "yes_bid": 45,
        "yes_ask": 47,
        "volume": 1234
    }

Our code reads these JSON responses and converts them into Python objects
(called "models") that the rest of the bot can use.

================================================================================
WHAT IS AUTHENTICATION / SIGNING?
================================================================================

You wouldn't want anyone to be able to place orders on your Kalshi account.
Authentication proves to Kalshi that a request really came from you.

Kalshi uses "cryptographic signing" — a mathematical technique using two related
keys (a "key pair"):
  - Private key: Only YOU have this. Never share it. Lives in your .env file.
  - Public key:  Kalshi has this. They registered it when you created your account.

When making a request, the bot:
  1. Takes the current timestamp + the HTTP method + the URL path
  2. Runs that through a mathematical function using your private key
  3. This produces a "signature" (a long random-looking string of letters/numbers)
  4. Sends the signature in the request header
  5. Kalshi uses your public key to verify the signature is genuine

It's like a wax seal on a letter — only you have the signet ring, so only you
could have made that seal. Even if someone intercepts the request, they can't
forge a valid signature without your private key.

RSA-PSS and ECDSA are two specific mathematical schemes for doing this signing.
This file supports both because Kalshi allows either type.

================================================================================
WHAT IS A BID/ASK SPREAD?
================================================================================

On any market (stock exchange, crypto exchange, prediction market), prices work
in pairs:
  - Bid price: The highest price a BUYER is willing to pay right now
  - Ask price: The lowest price a SELLER is willing to accept right now
  - Spread:    The gap between bid and ask

Example on Kalshi for "Bitcoin above $87,500 at 9:15 AM?":
  - yes_bid = 45 cents (buyers will pay up to 45¢ per YES contract)
  - yes_ask = 47 cents (sellers want at least 47¢ per YES contract)
  - Spread  = 2 cents

If you buy at the ask (47¢) and the market resolves YES, you get $1.00, making
53¢ profit. The market makers (who set bid/ask) keep a small slice as profit.

This file fetches bid/ask prices for every market the bot evaluates.

================================================================================
WHAT IS A CIRCUIT BREAKER?
================================================================================

In electrical systems, a circuit breaker cuts power when something goes wrong,
preventing damage. In software, it's the same idea:

If Kalshi's API starts failing repeatedly (network problem, server overload,
etc.), the circuit breaker "opens" and the bot stops hammering a broken API.
After 60 seconds it tries once more ("half-open") — if that works, normal
operation resumes. This prevents the bot from accumulating errors and burning
API rate limits on a known-bad connection.

================================================================================
WHAT IS RATE LIMITING?
================================================================================

APIs often limit how many requests you can make per second/minute/hour to prevent
abuse. If you exceed the limit, you get a "429 Too Many Requests" error. This
file handles 429 errors by waiting (1s, then 2s, then 4s) before retrying —
a strategy called "exponential backoff."

================================================================================
ARCHITECTURE: WHERE THIS FILE FITS
================================================================================

This module handles all direct communication with the Kalshi prediction market
exchange. Kalshi requires every API request to be signed with an RSA or ECDSA
private key using PSS padding (RSA) or ECDSA(SHA256).

Key responsibilities:
  - Authentication: Signs each request with timestamp + method + path using the
    private key loaded from a PEM file or environment variable.
  - Market Data: Fetches events (with nested markets), individual markets, and
    orderbooks. Supports both paginated (get_events) and full-catalog
    (get_all_events, ~5000 events / ~41000 markets) retrieval.
  - Portfolio: Retrieves account balance, open positions, and portfolio summary.
  - Order Execution: Places limit/market orders, cancels orders, lists open orders.
  - Field Parsing: Handles Kalshi's dual field naming conventions (legacy cents-based
    fields vs. newer _dollars/_fp suffixed fields) via helper functions.

Helper functions (module-level):
  - _dollars_to_cents(): Converts Kalshi dollar strings/floats to integer cents (0-100).
  - _parse_fp(): Parses _fp (floating point) fields to integers.
  - _parse_market(): Builds a Market model from a raw Kalshi API dict, handling
    both old and new API field names gracefully.

Connects to:
  - Kalshi demo API (demo-api.kalshi.co) or production API (api.elections.kalshi.com)
    depending on KALSHI_USE_DEMO config flag.
  - bot.config for API credentials and base URL selection.
  - bot.models for Market, Event, Position, OrderRequest, OrderResponse, PortfolioSummary.

Used by: bot.server (all market/portfolio/order endpoints), bot.backtester
(HistoricalDataFetcher), bot.main (CLI scan and trade execution).
"""

import base64    # Used to encode binary signature bytes as text (for HTTP headers)
import logging   # Python's built-in way to write diagnostic messages (not print())
import time      # Used to get current timestamp for request signing
from typing import Any, Optional  # Type hints — helps IDEs and humans understand what functions expect/return

import httpx  # A modern HTTP library for making web requests (like requests, but newer)

logger = logging.getLogger(__name__)
# The cryptography library handles the math behind RSA-PSS and ECDSA signing.
# "hazmat" = "hazardous materials" — this is low-level crypto; use carefully.
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, ec

from bot.config import config       # Loads settings from the .env file (API keys, URLs, etc.)
from bot.models import (            # Pydantic models — Python classes that define the shape of our data
    Event, Market, OrderRequest, OrderResponse, Position, PortfolioSummary,
)


def _dollars_to_cents(val) -> int:
    """Convert a Kalshi dollars string/float/None to integer cents (0-100).

    BACKGROUND — WHY CENTS?
    Kalshi's newer API returns prices as dollar strings like "0.4500" (45 cents).
    Internally, the bot works in integer cents (whole numbers: 0-100) because:
      - It avoids floating-point rounding errors (0.1 + 0.2 != 0.3 in floats!)
      - It's simpler to compare "45 cents < 70 cents" than "0.45 < 0.70"
      - Every Kalshi contract pays out exactly $1.00 = 100 cents

    Examples:
      "0.4500" -> 45    (45 cents, a 45% probability contract)
      "1.0"    -> 100   (100 cents = $1.00, a certainty)
      0.3      -> 30    (float input also works)
      None     -> 0     (missing data defaults to zero)
    """
    if val is None:
        return 0
    try:
        # float(val) handles both string "0.45" and number 0.45
        # * 100 converts dollars to cents
        # round() handles floating point imprecision (e.g., 0.45 * 100 = 44.999...)
        # int() converts the rounded float to a whole number
        return int(round(float(val) * 100))
    except (ValueError, TypeError):
        # If the value can't be converted (e.g., empty string, weird format), return 0
        return 0


def _parse_fp(val) -> int:
    """Parse a _fp (floating point) field to int.

    WHAT IS _fp?
    Kalshi's API went through a naming evolution. Older fields used bare integers
    (e.g., "volume": 1234). Newer fields use a "_fp" suffix (fixed-point) and
    return decimal strings like "1234.00" — the same number, just in a different
    format. This function normalizes them all to plain integers.

    Unlike _dollars_to_cents(), this does NOT multiply by 100 — _fp fields are
    already in the right unit (contracts, not dollars). We just round to the
    nearest whole number.

    Examples:
      "1234.00" -> 1234   (volume: 1234 contracts traded)
      "10.00"   -> 10     (position size: 10 contracts)
      None      -> 0      (missing data defaults to zero)
    """
    if val is None:
        return 0
    try:
        # float() handles string or number input
        # round() handles any decimal fraction
        # int() converts to whole number
        return int(round(float(val)))
    except (ValueError, TypeError):
        return 0


def _parse_market(m: dict) -> "Market":
    """Parse a market dict from the Kalshi API, handling both old and new field names.

    WHAT IS THIS FUNCTION DOING?
    When Kalshi sends us market data, it arrives as a raw Python dictionary (converted
    from JSON). This function reads all the fields we care about from that dictionary
    and builds a clean, typed Market object that the rest of the bot can use safely.

    WHY THE OLD/NEW FIELD DANCE?
    Kalshi upgraded their API and changed how they name price fields:
      OLD format:  {"yes_bid": 45, "yes_ask": 47}    ← prices as integers (cents)
      NEW format:  {"yes_bid_dollars": "0.4500", ...} ← prices as dollar strings

    During the transition, the API might send either format (or both). This function
    checks for the new "_dollars" fields first, and falls back to the old bare fields
    if the new ones aren't present. This keeps the bot working regardless of which
    format Kalshi sends.

    Args:
        m: A raw dictionary from the Kalshi API representing one market.

    Returns:
        A Market object with all fields normalized to the bot's internal format
        (prices in integer cents, volumes as integers, etc.).
    """
    # New API uses _dollars suffix; old API uses bare names
    # Use _dollars fields if present (preserving valid zeros); otherwise fall back to legacy fields
    yes_bid = _dollars_to_cents(m.get("yes_bid_dollars")) if m.get("yes_bid_dollars") is not None else (m.get("yes_bid", 0) or 0)
    # YES ask: lowest price you can BUY a YES contract for right now
    yes_ask = _dollars_to_cents(m.get("yes_ask_dollars")) if m.get("yes_ask_dollars") is not None else (m.get("yes_ask", 0) or 0)
    # NO bid: highest price someone will pay for a NO contract right now
    no_bid = _dollars_to_cents(m.get("no_bid_dollars")) if m.get("no_bid_dollars") is not None else (m.get("no_bid", 0) or 0)
    # NO ask: lowest price you can BUY a NO contract for right now
    no_ask = _dollars_to_cents(m.get("no_ask_dollars")) if m.get("no_ask_dollars") is not None else (m.get("no_ask", 0) or 0)
    # Volume: total number of contracts that have traded (higher = more liquid market)
    volume = _parse_fp(m.get("volume_fp")) if m.get("volume_fp") is not None else (m.get("volume", 0) or 0)
    # Open interest: number of contracts currently outstanding (not yet settled/cancelled)
    # Think of this as "how many bets are currently active on this market"
    open_interest = _parse_fp(m.get("open_interest_fp")) if m.get("open_interest_fp") is not None else (m.get("open_interest", 0) or 0)
    # Last price: price of the most recent trade that actually filled
    last_price = _dollars_to_cents(m.get("last_price_dollars")) if m.get("last_price_dollars") is not None else (m.get("last_price", 0) or 0)
    # Previous price: last price from an earlier snapshot (used to compute price change)
    prev_price = _dollars_to_cents(m.get("previous_price_dollars")) if m.get("previous_price_dollars") is not None else (m.get("previous_price", 0) or 0)
    # Live order book depth: contracts available at best ask/bid
    # "Size" = how many contracts are sitting at that price level in the order book.
    # A small size (e.g., 5 contracts) means the price will move quickly if you buy.
    # A large size (e.g., 500 contracts) means the market has deep liquidity.
    yes_ask_size = _parse_fp(m.get("yes_ask_size_fp")) if m.get("yes_ask_size_fp") is not None else (m.get("yes_ask_size", 0) or 0)
    yes_bid_size = _parse_fp(m.get("yes_bid_size_fp")) if m.get("yes_bid_size_fp") is not None else (m.get("yes_bid_size", 0) or 0)

    # Ensure all price/quantity fields are integers (not floats) for downstream comparisons.
    # isinstance() checks whether the value is a number type before converting.
    yes_bid = int(yes_bid) if isinstance(yes_bid, (int, float)) else 0
    yes_ask = int(yes_ask) if isinstance(yes_ask, (int, float)) else 0
    no_bid = int(no_bid) if isinstance(no_bid, (int, float)) else 0
    no_ask = int(no_ask) if isinstance(no_ask, (int, float)) else 0
    volume = int(volume) if isinstance(volume, (int, float)) else 0
    open_interest = int(open_interest) if isinstance(open_interest, (int, float)) else 0
    yes_ask_size = int(yes_ask_size) if isinstance(yes_ask_size, (int, float)) else 0
    yes_bid_size = int(yes_bid_size) if isinstance(yes_bid_size, (int, float)) else 0

    # Build and return a Market object with all the parsed fields.
    # m.get("field", default) safely reads from the dictionary — if the field
    # is missing, it returns the default value instead of crashing.
    return Market(
        ticker=m.get("ticker", ""),          # Unique ID for this market, e.g. "KXBTC15M-26MAR230915-B87500"
        event_ticker=m.get("event_ticker", ""),  # The parent event, e.g. "KXBTC15M-26MAR230915"
        title=m.get("title", ""),            # Human-readable question, e.g. "Bitcoin above $87,500?"
        subtitle=m.get("subtitle", ""),      # Optional extra description
        yes_bid=yes_bid,                     # Highest buy price for YES (cents, 0-100)
        yes_ask=yes_ask,                     # Lowest sell price for YES (cents, 0-100)
        no_bid=no_bid,                       # Highest buy price for NO (cents, 0-100)
        no_ask=no_ask,                       # Lowest sell price for NO (cents, 0-100)
        volume=volume,                       # Total contracts traded (higher = more liquid)
        open_interest=open_interest,         # Active contracts not yet settled
        status=m.get("status", "open"),      # "open", "closed", "settled"
        close_time=m.get("close_time", ""),  # ISO timestamp when trading stops
        category=m.get("category", ""),      # "crypto", "economics", "politics", etc.
        result=m.get("result", ""),          # "yes", "no", or "" if not yet resolved
        last_price=last_price,               # Most recent trade price (cents)
        prev_price=prev_price,               # Earlier trade price (cents) — for tracking movement
        # floor_strike and cap_strike are the price boundaries for range/bracket markets.
        # For "Bitcoin above $87,500?" markets, floor_strike = 87500.0
        floor_strike=float(m.get("floor_strike") or 0),
        cap_strike=float(m.get("cap_strike") or 0),
        yes_ask_size=yes_ask_size,           # Contracts available at the YES ask price
        yes_bid_size=yes_bid_size,           # Contracts available at the YES bid price
    )


class KalshiClient:
    """HTTP client for the Kalshi Trade API v2 with cryptographic request signing.

    WHAT IS A CLASS?
    A class is a blueprint for creating objects. Think of it like a recipe card:
    the class defines what ingredients (data) and steps (methods/functions) an
    object will have. When you write `client = KalshiClient()`, you're baking
    a cake from that recipe — creating a usable client object.

    WHAT DOES THIS CLASS DO?
    KalshiClient is the single object responsible for ALL communication with
    Kalshi's servers. The bot creates one instance of this at startup and uses
    it for everything: checking prices, checking the account balance, placing orders.

    Key design decisions:
    - Every method that talks to Kalshi goes through `_request()`, which handles
      authentication headers, retries, and circuit breaking in one place.
    - The circuit breaker prevents the bot from hammering a failed API endpoint.
    - Request signing (RSA/ECDSA) is done fresh on every call so timestamps are valid.

    Handles authentication, market data retrieval, portfolio management, and
    order execution. Each API request is signed with the user's RSA or ECDSA
    private key using the Kalshi-specific signature scheme (timestamp + method + path).

    Includes a circuit breaker: after 5 consecutive failures, the circuit opens
    for 60 seconds, rejecting all requests. After 60s it enters half-open state
    and allows one probe request. Success closes the circuit; failure re-opens it.
    """

    # Circuit breaker configuration — these are class-level constants (shared by all instances).
    # 5 failures in a row = circuit opens; 60 seconds of cooling off before retrying.
    CIRCUIT_FAILURE_THRESHOLD = 5
    CIRCUIT_RECOVERY_TIMEOUT = 60  # seconds

    def __init__(self):
        """Initialize the client with API credentials from config.

        WHAT IS __init__?
        __init__ is Python's constructor — it runs automatically when you create
        a new KalshiClient() object. It sets up everything the object needs to
        start working: the API URL, the credentials, the HTTP connection pool,
        and the circuit breaker state.

        WHAT IS self?
        `self` refers to the specific instance being created. When you have two
        KalshiClient objects, `self` keeps their data separate — like how two
        cars from the same factory each have their own fuel gauge.

        Loads the private key (from env var or PEM file) for request signing.
        Uses a 30-second timeout for all HTTP requests to handle Kalshi's
        occasionally slow responses during high-traffic periods.
        """
        self.base_url = config.kalshi_base_url      # The Kalshi API root URL from .env
        self.api_key_id = config.kalshi_api_key_id  # Your API key ID (public, not secret)
        self._private_key = None   # Will be loaded below — kept private (underscore prefix = internal)
        # httpx.Client is a persistent HTTP connection pool — reusing TCP connections
        # is much faster than opening a new connection for every request.
        # timeout=120 means wait up to 2 minutes for a response (needed for large paginated fetches)
        self._client = httpx.Client(timeout=120)  # 2 min for large paginated fetches

        # Circuit breaker state — tracks the health of the API connection.
        # "closed" = normal operation (counterintuitive: "closed" circuit = electricity flows = works)
        # "open"   = failing, rejecting all requests to avoid hammering a broken API
        # "half_open" = testing one request after the cooldown period to see if API recovered
        self._circuit_state = "closed"  # "closed", "open", "half_open"
        self._consecutive_failures = 0   # Counter: how many requests failed in a row
        self._circuit_opened_at = 0.0    # Unix timestamp of when the circuit tripped
        self._last_failure_time = 0.0    # Unix timestamp of the most recent failure

        # Load the private key from the environment variable (KALSHI_PRIVATE_KEY in .env)
        if config.kalshi_private_key:
            self._load_private_key(config.kalshi_private_key)

    def get_circuit_state(self) -> dict:
        """Return the current circuit breaker state and metadata.

        Called by the /api/status endpoint so the dashboard can display
        whether the Kalshi connection is healthy. Returns a dictionary
        so it can be directly serialized to JSON for the frontend.
        """
        now = time.time()
        # If the circuit has been open long enough, promote it to half-open
        # so the next real request can test whether the API recovered.
        if self._circuit_state == "open":
            elapsed = now - self._circuit_opened_at
            if elapsed >= self.CIRCUIT_RECOVERY_TIMEOUT:
                self._circuit_state = "half_open"
        return {
            "state": self._circuit_state,
            "consecutive_failures": self._consecutive_failures,
            # Only show circuit_opened_at if the circuit is not in normal operation
            "circuit_opened_at": self._circuit_opened_at if self._circuit_state != "closed" else None,
            "last_failure_time": self._last_failure_time if self._last_failure_time > 0 else None,
            "recovery_timeout_seconds": self.CIRCUIT_RECOVERY_TIMEOUT,
            "failure_threshold": self.CIRCUIT_FAILURE_THRESHOLD,
        }

    def _circuit_breaker_check(self):
        """Check circuit breaker state before making a request.

        Raises RuntimeError if circuit is open and recovery timeout has not elapsed.
        Transitions open -> half_open if timeout has elapsed.
        """
        now = time.time()
        if self._circuit_state == "open":
            elapsed = now - self._circuit_opened_at
            if elapsed < self.CIRCUIT_RECOVERY_TIMEOUT:
                raise RuntimeError(
                    f"Circuit breaker OPEN: {self._consecutive_failures} consecutive failures. "
                    f"Retry in {int(self.CIRCUIT_RECOVERY_TIMEOUT - elapsed)}s."
                )
            # Transition to half-open: allow one probe request
            self._circuit_state = "half_open"
            logger.info("Circuit breaker: open -> half_open (attempting probe request)")

    def _circuit_breaker_success(self):
        """Record a successful request. Closes the circuit if half-open."""
        if self._circuit_state == "half_open":
            logger.info("Circuit breaker: half_open -> closed (probe succeeded)")
        self._circuit_state = "closed"
        self._consecutive_failures = 0

    def _circuit_breaker_failure(self):
        """Record a failed request. Opens the circuit if threshold is reached."""
        self._consecutive_failures += 1
        self._last_failure_time = time.time()
        if self._consecutive_failures >= self.CIRCUIT_FAILURE_THRESHOLD:
            self._circuit_state = "open"
            self._circuit_opened_at = time.time()
            logger.warning(
                f"Circuit breaker OPENED after {self._consecutive_failures} consecutive failures"
            )

    def _load_private_key(self, pem_data: str):
        """Load and parse the PEM-encoded private key for request signing.

        WHAT IS A PEM FILE?
        PEM (Privacy-Enhanced Mail) is a text format for storing cryptographic keys.
        It looks like this in your .env file:
            -----BEGIN PRIVATE KEY-----
            MIIEvQIBADANBgkqhkiG9w0BAQEFAASC...
            -----END PRIVATE KEY-----

        The long middle section is your private key encoded in base64 (a way to
        represent binary data as plain text characters). This function takes that
        text, decodes it, and loads it into a Python cryptography object that can
        be used to sign requests.

        WHY NOT STORE THE KEY AS A PLAIN NUMBER?
        Cryptographic keys are very large numbers (RSA keys are 2048 or 4096 bits).
        PEM format is the standard way to store them as text files.
        """
        try:
            # .encode() converts the Python string to bytes (cryptography library needs bytes)
            # password=None means the key file itself is not password-protected
            self._private_key = serialization.load_pem_private_key(
                pem_data.encode(), password=None
            )
        except Exception as e:
            # If the key is malformed, log the error but don't crash the whole bot.
            # All requests will fail until the key is fixed.
            logger.error(f"Failed to load private key: {e}")
            self._private_key = None

    def _sign_request(self, method: str, full_path: str, timestamp_ms: int) -> str:
        """Generate a cryptographic signature for Kalshi API authentication.

        HOW SIGNING WORKS — STEP BY STEP:
        1. We create a "message" by concatenating three pieces of information:
               timestamp_ms + METHOD + path
           Example: "1711450000000GET/trade-api/v2/events"
           The timestamp ensures old signatures can't be replayed by attackers.

        2. We run this message through a signing algorithm using our private key.
           This produces a unique "signature" — a sequence of bytes that can ONLY
           be produced with our specific private key and this specific message.

        3. We encode the signature as base64 text so it can be sent in an HTTP header.

        Kalshi's server then verifies the signature using our public key (which they
        have on file). If it verifies correctly, they know the request is genuine.

        WHAT IS SHA256?
        SHA256 is a "hash function" — it takes any amount of data and produces a
        fixed-size 256-bit fingerprint. Two different inputs always produce different
        fingerprints (for all practical purposes). Signing works on the hash of the
        message, not the message itself — this is more efficient for large messages.

        WHAT IS PSS? WHAT IS MGF1?
        These are technical details of HOW RSA signing applies randomness to make
        the signature harder to forge. Kalshi requires PSS (not the older PKCS1v15).
        MGF1 is the "mask generation function" used inside PSS — it's a standard
        component. You don't need to understand the math; just know Kalshi requires
        these specific settings.

        WHAT IS ECDSA?
        Elliptic Curve Digital Signature Algorithm — an alternative to RSA that uses
        different math (elliptic curves instead of prime factoring). It's faster and
        produces smaller signatures. Both RSA and ECDSA are supported here.

        Kalshi's auth scheme requires signing: "{timestamp_ms}{METHOD}{path}" (no query params).
        Supports both RSA keys (PSS padding with SHA256) and ECDSA keys (SHA256).

        Args:
            method: HTTP method in uppercase (GET, POST, DELETE).
            full_path: Full API path including /trade-api/v2 prefix (query params stripped).
            timestamp_ms: Current time in milliseconds since epoch.

        Returns:
            Base64-encoded signature string for the KALSHI-ACCESS-SIGNATURE header.
        """
        if self._private_key is None:
            raise RuntimeError("Kalshi private key not loaded — check KALSHI_PRIVATE_KEY env var")
        # Strip query params before signing — Kalshi only signs the path portion.
        # E.g., "/trade-api/v2/events?limit=200" becomes "/trade-api/v2/events"
        path_no_query = full_path.split("?")[0]
        # Construct the message: timestamp + method + path (no body, no query params).
        # .encode("utf-8") converts the Python string to bytes — required by the crypto library.
        message = f"{timestamp_ms}{method}{path_no_query}".encode("utf-8")

        if isinstance(self._private_key, ec.EllipticCurvePrivateKey):
            # ECDSA signing for EC private keys
            signature = self._private_key.sign(
                message, ec.ECDSA(hashes.SHA256())
            )
        else:
            # RSA signing — Kalshi requires PSS padding (not PKCS1v15)
            # Salt length = digest length (32 bytes for SHA256)
            signature = self._private_key.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH,
                ),
                hashes.SHA256(),
            )
        # base64.b64encode converts binary bytes to text (safe to send in HTTP headers)
        # .decode() converts the result from bytes to a Python string
        return base64.b64encode(signature).decode()

    def _auth_headers(self, method: str, full_path: str) -> dict[str, str]:
        """Build the authentication headers required by every Kalshi API request.

        WHAT ARE HTTP HEADERS?
        When a web browser or program makes a request, it sends not just the URL
        but also a set of "headers" — key-value pairs with metadata about the
        request. Headers are like the envelope on a letter: they tell the recipient
        who sent it, what's inside, and how to process it.

        Kalshi requires four specific headers on every authenticated request:
          - KALSHI-ACCESS-KEY:       Your API key ID (identifies your account)
          - KALSHI-ACCESS-SIGNATURE: The cryptographic proof that this request is yours
          - KALSHI-ACCESS-TIMESTAMP: When the request was made (Kalshi rejects stale requests)
          - Content-Type:            Tells Kalshi the body is JSON-formatted text

        Returns a dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE,
        KALSHI-ACCESS-TIMESTAMP, and Content-Type headers.
        """
        # time.time() returns seconds since Jan 1, 1970 (Unix epoch) as a float.
        # * 1000 converts to milliseconds; int() drops the decimal part.
        # Kalshi's server checks that the timestamp is within a few seconds of its own
        # clock — this prevents "replay attacks" where someone re-sends an old request.
        ts = int(time.time() * 1000)  # Current time in milliseconds
        sig = self._sign_request(method.upper(), full_path, ts)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,       # Your public API key ID
            "KALSHI-ACCESS-SIGNATURE": sig,              # Your cryptographic signature
            "KALSHI-ACCESS-TIMESTAMP": str(ts),          # Millisecond timestamp (as string for header)
            "Content-Type": "application/json",          # Tells Kalshi to parse the body as JSON
        }

    # Retry configuration for rate limits (429) and server errors (5xx).
    # MAX_RETRIES = 3 means we try up to 4 times total (1 initial + 3 retries).
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0  # seconds; doubles each retry (1s, 2s, 4s) = "exponential backoff"

    def _request(
        self, method: str, path: str, params: dict | None = None, json: dict | None = None
    ) -> dict[str, Any]:
        """Execute an authenticated HTTP request to the Kalshi API.

        This is the heart of the client — every public method (get_events,
        place_order, get_balance, etc.) calls this function to do the actual
        network communication. It handles three concerns:

        1. CIRCUIT BREAKING: Before even attempting the request, check if we've
           had too many recent failures. If so, raise immediately instead of
           making a doomed request.

        2. AUTHENTICATION: Build fresh signed headers for every attempt
           (the timestamp in the signature must be current).

        3. RETRY WITH BACKOFF: If Kalshi returns a rate limit error (429) or
           server error (5xx), wait and try again automatically. The wait time
           doubles each retry: 1s → 2s → 4s. This is "exponential backoff" —
           it gives the server time to recover without overwhelming it further.

        WHAT ARE HTTP STATUS CODES?
        Every HTTP response includes a 3-digit status code:
          2xx (200-299): Success — the request worked
          4xx (400-499): Client error — something wrong with your request
            400: Bad request (malformed data)
            401: Unauthorized (bad credentials)
            403: Forbidden (valid credentials but not allowed)
            404: Not found (the resource doesn't exist)
            429: Too many requests (you're being rate-limited)
          5xx (500-599): Server error — Kalshi's servers have a problem
            500: Internal server error
            503: Service unavailable (server overloaded)

        We only retry on 429 and 5xx — these are temporary problems. 4xx errors
        (except 429) mean something is wrong with our request, so retrying won't help.

        Checks the circuit breaker before making the request. Retries on 429
        (rate limit) and 5xx (server error) responses with exponential backoff
        (1s, 2s, 4s). Max 3 retries before propagating.

        Args:
            method: HTTP method (GET, POST, DELETE).
            path: API path relative to /trade-api/v2 (e.g., "/events", "/portfolio/orders").
            params: Optional query parameters (appended to URL as ?key=value).
            json: Optional JSON request body (for POST requests).

        Returns:
            Parsed JSON response as a dict.

        Raises:
            RuntimeError: If the circuit breaker is open.
            httpx.HTTPStatusError: On non-2xx responses after all retries are exhausted.
        """
        # Circuit breaker: reject request if circuit is open (unless recovery timeout elapsed)
        self._circuit_breaker_check()

        # Build the full URL: base_url is the Kalshi host, path is the endpoint
        url = f"{self.base_url}{path}"
        # The signing function needs the full path including the /trade-api/v2 prefix
        full_path = f"/trade-api/v2{path}"
        last_exc: Exception | None = None

        for attempt in range(self.MAX_RETRIES + 1):
            # Re-sign on each attempt (timestamp must be fresh — stale timestamps are rejected)
            headers = self._auth_headers(method.upper(), full_path)
            try:
                # Make the actual HTTP request. `params` are added to the URL (?limit=200&status=open).
                # `json` is serialized to a JSON string and sent as the request body.
                resp = self._client.request(method, url, headers=headers, params=params, json=json)
                # raise_for_status() converts HTTP error codes into Python exceptions.
                # If status is 200-299 it does nothing; otherwise it raises HTTPStatusError.
                resp.raise_for_status()
                # Request succeeded — close the circuit breaker if it was half-open
                self._circuit_breaker_success()
                # .json() parses the JSON response body into a Python dictionary
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                # Log the response body for debugging 4xx errors (e.g., invalid order parameters)
                if status < 500:
                    try:
                        error_body = exc.response.text[:500]  # [:500] limits to first 500 characters
                        logger.error(f"Kalshi API {status} on {method} {path}: {error_body}")
                    except Exception:
                        pass
                if status == 429 or status >= 500:
                    # These are temporary errors worth retrying
                    if attempt < self.MAX_RETRIES:
                        # 2 ** attempt = 1, 2, 4 (exponential backoff: each retry waits twice as long)
                        delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            f"Kalshi API {status} on {method} {path}, "
                            f"retry {attempt + 1}/{self.MAX_RETRIES} in {delay:.1f}s"
                        )
                        time.sleep(delay)  # Wait before retrying
                        continue  # Jump back to the top of the for loop
                # Non-retryable status (4xx) or retries exhausted — record failure and raise
                self._circuit_breaker_failure()
                raise
            except Exception as exc:
                # Network error (DNS failure, connection refused, timeout, etc.)
                last_exc = exc
                self._circuit_breaker_failure()
                raise

        # Should not reach here (loop always raises or returns), but safety net
        self._circuit_breaker_failure()
        raise last_exc  # type: ignore[misc]

    # ── Market Data ──────────────────────────────────────────────
    # These methods fetch information about what markets exist and what prices they show.
    # They do NOT place orders — they only READ data from Kalshi.

    def get_events(
        self, limit: int = 50, status: str = "open", with_nested_markets: bool = True
    ) -> list[Event]:
        """Fetch top events sorted by volume.

        WHAT IS AN EVENT vs. A MARKET?
        On Kalshi, an "event" is the parent container and a "market" is one specific
        binary question within that event.

        Example:
          Event:  "Bitcoin 15-Minute Market — March 26, 9:15 AM"
          Markets inside:
            - "Bitcoin above $87,000?" (ticker: KXBTC15M-26MAR230915-B87000)
            - "Bitcoin above $87,250?" (ticker: KXBTC15M-26MAR230915-B87250)
            - "Bitcoin above $87,500?" (ticker: KXBTC15M-26MAR230915-B87500)

        This function fetches a limited number of events (sorted by trading volume)
        without paginating through ALL of Kalshi. Useful for quick scans.
        Use get_all_events() if you want every market on the platform.

        Args:
            limit: Maximum number of events to return (capped at 200 per API limit).
            status: "open" for active markets, "closed" for resolved ones.
            with_nested_markets: If True, include the markets inside each event.

        Returns:
            List of Event objects, sorted by combined market volume (highest first).
        """
        data = self._request("GET", "/events", params={
            "limit": min(limit, 200),   # min() ensures we never ask for more than 200 (API maximum)
            "status": status,
            "with_nested_markets": str(with_nested_markets).lower(),  # API expects "true"/"false" strings
        })
        events = []
        for ev in data.get("events", []):  # .get("events", []) safely returns [] if the key is missing
            markets = []
            for m in ev.get("markets", []):
                markets.append(_parse_market(m))  # Convert each raw dict to a typed Market object
            events.append(Event(
                event_ticker=ev.get("event_ticker", ""),
                title=ev.get("title", ""),
                category=ev.get("category", ""),
                markets=markets,
                # Sum the volume across all markets in this event (more active events = higher priority)
                volume=sum(m.volume for m in markets),
            ))
        # Sort by volume descending — highest-volume (most active) events first
        events.sort(key=lambda e: e.volume, reverse=True)
        return events[:limit]

    def get_all_events(self, status: str = "open", with_nested_markets: bool = True) -> list[Event]:
        """Fetch ALL events from Kalshi using cursor-based pagination.

        WHAT IS PAGINATION?
        APIs rarely return thousands of results in one response — it would be too slow
        and use too much memory. Instead, they break results into "pages". You request
        page 1, get 200 items; then request page 2, get the next 200 items; etc.

        WHAT IS A CURSOR?
        A cursor is a bookmark. Instead of saying "give me page 3", you say "give me
        the items that come AFTER this cursor value". Cursors are more reliable than
        page numbers because they work correctly even if items are added/removed while
        you're paginating.

        The pattern here:
          - First request: no cursor → get first 200 events + a cursor for the next page
          - Second request: pass that cursor → get next 200 events + new cursor
          - Continue until cursor is empty (no more pages) or page limit is reached

        WHY THIS IS EXPENSIVE:
        Kalshi has ~5,000 events with ~41,000 markets total. Loading everything takes
        25+ API requests and several seconds. This is why the bot uses
        get_markets_by_series() in parallel for targeted crypto scans instead.

        This is the method used by the auto-scan background job. For lighter on-demand
        scans, use get_events() with a limit instead.
        """
        all_events = []
        cursor = None   # No cursor on first request
        max_pages = 200  # Safety limit: 200 pages × 200 events = 40,000 max
        page = 0
        while page < max_pages:
            params = {
                "limit": 200,    # Maximum allowed by the API
                "status": status,
                "with_nested_markets": str(with_nested_markets).lower(),
            }
            if cursor:
                params["cursor"] = cursor   # Tell the API where to continue from
            data = self._request("GET", "/events", params=params)
            for ev in data.get("events", []):
                markets = []
                for m in ev.get("markets", []):
                    markets.append(_parse_market(m))
                all_events.append(Event(
                    event_ticker=ev.get("event_ticker", ""),
                    title=ev.get("title", ""),
                    category=ev.get("category", ""),
                    markets=markets,
                    volume=sum(m.volume for m in markets),
                ))
            # The API puts the next page's cursor in the response.
            # If it's empty or missing, we've reached the last page.
            cursor = data.get("cursor", None)
            page += 1
            if not cursor or not data.get("events"):
                break  # No more pages — stop paginating
        # Sort by volume descending
        all_events.sort(key=lambda e: e.volume, reverse=True)
        return all_events

    def get_markets_by_series(self, series_ticker: str, status: str = "open") -> list[Market]:
        """Fetch all markets for a single series (e.g. 'KXBTC15M' or 'KXBTC').

        WHAT IS A SERIES?
        A series is a group of related events. For example:
          Series "KXBTC15M" = all Bitcoin 15-minute markets (new ones created every 15 min)
          Series "KXETH15M" = all Ethereum 15-minute markets

        Instead of downloading all 41,000 markets and filtering, we ask specifically
        "give me all markets for KXBTC15M" — this is much faster. The bot runs
        these requests for 9-13 series in parallel, replacing what used to take
        25 sequential pagination requests.

        WHY NOT EXCEPTION ON MISSING SERIES?
        Some series tickers are speculative — maybe Kalshi added new coins. This
        function returns an empty list instead of crashing so the bot can safely
        attempt all known series prefixes and ignore the ones that don't exist yet.

        Uses GET /markets?series_ticker=X — a single targeted API call instead of
        paginating through 41,000 markets. 9-13 of these in parallel via asyncio.gather
        replaces the 25-request get_all_events() pagination loop.

        Returns an empty list (not an exception) if the series doesn't exist, so
        callers can safely pass all candidate series tickers and discard empties.
        """
        try:
            markets: list[Market] = []
            cursor = None
            for _ in range(20):  # Safety: 20 pages × 200 = 4000 markets max per series
                params: dict = {
                    "series_ticker": series_ticker,
                    "status": status,
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor
                data = self._request("GET", "/markets", params=params)
                for m in data.get("markets", []):
                    markets.append(_parse_market(m))
                cursor = data.get("cursor")
                if not cursor or not data.get("markets"):
                    break
            return markets
        except Exception:
            return []

    def get_market(self, ticker: str) -> Market | None:
        """Fetch a single market by ticker.

        Used to get a fresh, up-to-date snapshot of one specific market.
        The `ticker` is the unique identifier for a market, like:
            "KXBTC15M-26MAR230915-B87500"

        Returns None (instead of raising an exception) if the market doesn't exist
        or if there's an HTTP error, so callers can handle missing markets gracefully.
        """
        try:
            data = self._request("GET", f"/markets/{ticker}")  # f-string builds the URL path
            m = data.get("market", {})  # The API wraps the market in a "market" key
            return _parse_market(m)
        except httpx.HTTPStatusError:
            return None  # Market not found or other HTTP error — return None instead of crashing

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Fetch the full orderbook for a market.

        WHAT IS AN ORDERBOOK?
        The orderbook is the complete list of all pending buy and sell orders
        for a market, organized by price level. It looks like:

            Sell (ask) side:       Buy (bid) side:
              47¢ — 50 contracts      45¢ — 80 contracts
              48¢ — 30 contracts      44¢ — 120 contracts
              50¢ — 200 contracts     40¢ — 500 contracts

        The "spread" is the gap between the lowest ask (47¢) and highest bid (45¢).
        A tight spread (1-2¢) means liquid market; a wide spread (5-10¢) means illiquid.

        This gives more detail than just the top bid/ask — you can see depth at
        each price level, which tells you how much you can buy without moving the price.
        """
        return self._request("GET", f"/markets/{ticker}/orderbook")

    # ── Portfolio ────────────────────────────────────────────────
    # These methods read your account state: how much cash you have, what positions
    # you hold, and the history of your orders.

    def get_balance(self) -> int:
        """Get account balance (cash available to bet) in cents.

        This is your "dry powder" — the cash sitting in your Kalshi account that
        hasn't been bet yet. If your balance is 10000, you have $100.00 to spend.

        Returns:
            Integer cents. E.g., 10000 = $100.00
        """
        data = self._request("GET", "/portfolio/balance")
        return data.get("balance", 0)

    def get_portfolio_value(self) -> int:
        """Get portfolio value (the current market value of your open positions) in cents.

        BALANCE vs. PORTFOLIO VALUE:
          - Balance: Cash you haven't bet yet (immediately spendable)
          - Portfolio value: The current market value of all contracts you hold

        If you bought a YES contract at 40¢ and it's now worth 65¢, the
        portfolio value reflects the 65¢ mark-to-market price — your unrealized gain.

        Returns:
            Integer cents representing the current value of all open positions.
        """
        data = self._request("GET", "/portfolio/balance")
        return data.get("portfolio_value", 0)

    def get_positions(self) -> list[Position]:
        """Get all open positions from Kalshi.

        WHAT IS A POSITION?
        A position is a contract you currently hold. For example:
          - You bought 5 YES contracts for "Bitcoin above $87,500" at 40¢ each
          - Your position is: ticker="KXBTC15M-...", side="yes", quantity=5, avg_price=40¢

        WHY BOTH YES AND NO POSITIONS?
        On Kalshi you can bet EITHER way. Buying a YES contract profits if the
        outcome happens. Buying a NO contract profits if it DOESN'T happen.
        A positive position_fp means you hold YES contracts; negative means NO.

        Kalshi API v2 uses fixed-point strings:
          - position_fp: "10.00" (positive=YES, negative=NO contracts)
          - market_exposure_dollars: "0.5600" (cost basis in dollars — how much you spent)

        WHAT IS COST BASIS / AVERAGE PRICE?
        If you bought 5 contracts at different prices, average price is:
            (total money spent) / (number of contracts)
        Kalshi gives us "market_exposure_dollars" which is total money spent.
        We divide by quantity to get the average price per contract.
        """
        data = self._request("GET", "/portfolio/positions", params={
            "limit": 200,
            "count_filter": "position",  # Only return positions with non-zero count (ignore empties)
        })
        if not data:
            return []
        raw_positions = data.get("market_positions") or []
        logger.info(f"[get_positions] {len(raw_positions)} positions from Kalshi")
        positions = []
        for p in raw_positions:
            ticker = p.get("ticker", "")
            # Parse position_fp: positive = YES contracts, negative = NO contracts
            # "10.00" means 10 YES contracts; "-5.00" means 5 NO contracts
            position_fp = p.get("position_fp", "0")
            try:
                position_count = float(position_fp)
            except (ValueError, TypeError):
                position_count = 0
            if position_count == 0:
                # Fallback to legacy fields — older API responses use yes_amount/no_amount separately
                position_count = p.get("yes_amount", 0) or -(p.get("no_amount", 0) or 0)
            side = "yes" if position_count > 0 else "no"  # Positive = YES, negative = NO
            qty = abs(int(position_count))  # abs() = absolute value (always positive)
            # Parse market_exposure_dollars for avg price.
            # "market_exposure" = total money you spent on this position.
            exposure_str = p.get("market_exposure_dollars", "0")
            try:
                exposure_dollars = float(exposure_str)
            except (ValueError, TypeError):
                exposure_dollars = 0
            # Divide total exposure by quantity to get average price per contract.
            # max(qty, 1) prevents division by zero if qty is somehow 0.
            avg_price_cents = round(exposure_dollars * 100 / max(qty, 1)) if qty > 0 else 0
            if qty > 0:
                positions.append(Position(
                    ticker=ticker,
                    event_ticker="",  # event_ticker is not in MarketPosition schema (not provided by API here)
                    side=side,
                    quantity=qty,
                    avg_price_cents=avg_price_cents,
                ))
                logger.info(f"[get_positions] {ticker}: {side} x{qty} @ {avg_price_cents}c (fp={position_fp})")
        return positions

    def get_portfolio_summary(self) -> PortfolioSummary:
        """Get full portfolio state with a single balance API call."""
        data = self._request("GET", "/portfolio/balance")
        balance = data.get("balance", 0)
        portfolio_value = data.get("portfolio_value", 0)
        positions = self.get_positions()
        return PortfolioSummary(
            balance_cents=balance,
            portfolio_value_cents=portfolio_value,
            positions=positions,
        )

    def get_rewards(self) -> dict:
        """Fetch daily LP rewards score and estimated payout from Kalshi.

        WHAT ARE LP REWARDS?
        LP = "Liquidity Provider". Kalshi rewards traders who place limit orders
        (resting on the order book) because they provide liquidity — they make
        it easier for others to trade by having orders waiting to fill.

        The LP score is earned by having limit orders sitting in the order book.
        At the end of each day, Kalshi distributes a rewards pool among all LPs
        proportional to their score. Higher score = bigger share of the pool.

        The "estimated_payout" is the projected dollar amount you'll receive
        based on your current score and the total pool size.

        Returns a dict with keys:
          - daily_lp_score:    Cumulative LP score for today (float)
          - estimated_payout:  Projected dollar payout based on pool share (float, USD)
          - raw:               Full raw response from Kalshi for debugging

        Returns empty dict with error key on failure (endpoint may not exist
        for all account tiers or may not yet be accessible).
        """
        try:
            data = self._request("GET", "/portfolio/rewards")
            return {
                "daily_lp_score": float(data.get("daily_lp_score") or 0),
                "estimated_payout": float(data.get("estimated_payout") or 0),
                "raw": data,
            }
        except Exception as exc:
            return {"error": str(exc), "daily_lp_score": 0.0, "estimated_payout": 0.0}

    # ── Orders ───────────────────────────────────────────────────
    # These methods create, cancel, and inspect orders on Kalshi.
    # An "order" is an instruction to buy or sell contracts.

    def place_order(self, order: OrderRequest) -> OrderResponse:
        """Place an order on Kalshi.

        WHAT IS A LIMIT ORDER vs. MARKET ORDER?
          - Market order: "Buy NOW at whatever price is available." Fast, but you
            might pay more than expected if the spread is wide.
          - Limit order: "Buy ONLY if the price is at or below X cents."
            You might not get filled immediately, but you control the price you pay.

        This bot uses limit orders exclusively, placing them at bid+1 (one cent
        above the current best bid). This gives a better fill price than the ask
        while still matching quickly.

        HOW KALSHI'S ORDER SIDES WORK:
        Kalshi only understands "YES price" internally. For NO orders, we send
        "no_price" instead of "yes_price" and let Kalshi handle the conversion.
        Note: YES price + NO price = 100 cents (because one must win and pay $1.00).
        So buying NO at 55¢ is the same as implying YES is worth 45¢.

        WHAT IS count_fp?
        The Kalshi v2 API requires the quantity as BOTH an integer ("count") and
        a decimal string ("count_fp": "5.00"). Including both ensures compatibility
        regardless of which API version processes the request.

        For limit orders, the API requires yes_price in cents. If the order side
        is NO, we convert: yes_price = 100 - no_price (since YES + NO = 100c).
        Market orders do not require a price field.

        Args:
            order: OrderRequest with ticker, side, type, count, and price.

        Returns:
            OrderResponse with order_id, status, and fill information.
        """
        # Validate price_cents is in valid range before submitting.
        # 0 and 100 are not valid (0 = free, 100 = certain — neither makes sense to trade).
        if not (1 <= order.price_cents <= 99):
            raise ValueError(f"price_cents must be 1-99, got {order.price_cents}")

        # Build the order body that Kalshi's API expects.
        # .value on an Enum extracts the string value (e.g., OrderSide.YES.value = "yes")
        body = {
            "ticker": order.ticker,         # Which market to trade in
            "action": order.action.value,   # "buy" or "sell"
            "side": order.side.value,       # "yes" or "no"
            "count": order.count,           # Number of contracts (integer)
            "count_fp": f"{order.count:.2f}",  # Same count as decimal string (e.g., "5.00")
            "time_in_force": order.time_in_force,  # "good_till_canceled", "immediate_or_cancel", or "fill_or_kill"
        }
        # Kalshi requires EXACTLY ONE price field — either yes_price OR no_price, not both.
        # The price is in cents (integer), e.g., 45 means 45¢ = $0.45 per contract.
        # For IOC orders we use an aggressive (ask+buffer) price to sweep available liquidity.
        if order.side.value == "yes":
            body["yes_price"] = order.price_cents
        else:
            body["no_price"] = order.price_cents

        # POST to /portfolio/orders sends the order to Kalshi's matching engine.
        # The matching engine tries to match our order against existing orders.
        # If it matches, we get a "fill". If not, it "rests" in the order book.
        data = self._request("POST", "/portfolio/orders", json=body)
        o = data.get("order", {})
        # Parse price: prefer yes_price_dollars, fall back to legacy yes_price
        price_cents = _dollars_to_cents(o.get("yes_price_dollars")) if o.get("yes_price_dollars") is not None else (o.get("yes_price", 0) or 0)
        # Parse counts: prefer _fp fields, fall back to legacy integers.
        # fill_count = how many contracts were immediately matched.
        # remaining = how many contracts are still resting in the order book waiting to fill.
        fill_count = _parse_fp(o.get("fill_count_fp")) if o.get("fill_count_fp") is not None else (o.get("fill_count", 0) or 0)
        remaining = _parse_fp(o.get("remaining_count_fp")) if o.get("remaining_count_fp") is not None else (o.get("remaining_count", 0) or 0)
        total_count = fill_count + remaining if fill_count or remaining else (o.get("count", 0) or 0)
        return OrderResponse(
            order_id=o.get("order_id", ""),   # Unique ID assigned by Kalshi (used for cancellation)
            ticker=o.get("ticker", ""),
            status=o.get("status", ""),        # "resting", "executed", "cancelled"
            side=o.get("side", ""),
            action=o.get("action", ""),
            price_cents=price_cents,
            count=total_count,
            remaining_count=remaining,         # > 0 means partially unfilled (waiting in order book)
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open (resting) order so it won't fill.

        WHY CANCEL?
        If we placed a limit order and the market moved away from our price, the
        order sits unfilled in the order book, tying up our capital. Cancelling
        frees up that capital for better opportunities.

        The bot also cancels existing orders before placing updated ones when
        the target price changes.

        Args:
            order_id: The unique order ID returned by place_order().

        Returns:
            True if successfully cancelled, False if the order wasn't found
            (already filled or already cancelled).
        """
        try:
            # DELETE request to remove the order from the order book
            self._request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except httpx.HTTPStatusError:
            # 404 = order not found (already filled or expired) — not a crash-worthy error
            return False

    def get_open_orders(self) -> list[dict]:
        """Get all open (resting) orders — orders that have been placed but not yet filled.

        WHAT IS A "RESTING" ORDER?
        When you place a limit order at a price where no one is selling (for a buy),
        the order "rests" in the order book waiting for someone to sell at your price.
        It stays resting until: filled, cancelled, or the market closes.

        WHY TRACK OPEN ORDERS?
        The bot periodically checks its open orders to:
          1. Avoid duplicating orders on the same market
          2. Cancel stale orders when the market has moved
          3. Display them on the dashboard

        Enriches each order dict with parsed cents/integer fields from FixedPoint
        fields (yes_price_dollars, no_price_dollars, fill_count_fp, remaining_count_fp)
        for downstream consumers that expect integer cents values.
        """
        data = self._request("GET", "/portfolio/orders", params={"status": "resting"})
        orders = data.get("orders", [])
        for o in orders:
            # Enrich each order dict with normalized cent values.
            # The rest of the bot uses integer cents; the API returns dollar strings.
            # We add new keys ("yes_price", "fill_count", etc.) alongside the originals.
            if o.get("yes_price_dollars") is not None:
                o["yes_price"] = _dollars_to_cents(o["yes_price_dollars"])
            if o.get("no_price_dollars") is not None:
                o["no_price"] = _dollars_to_cents(o["no_price_dollars"])
            if o.get("fill_count_fp") is not None:
                o["fill_count"] = _parse_fp(o["fill_count_fp"])
            if o.get("remaining_count_fp") is not None:
                o["remaining_count"] = _parse_fp(o["remaining_count_fp"])
        return orders

    def get_executed_orders(self, limit: int = 200) -> list[dict]:
        """Get executed (filled) orders from Kalshi trade history.

        WHAT IS AN EXECUTED / FILLED ORDER?
        When a limit order matches with another trader's opposing order, it "fills"
        (gets executed). Filled orders appear in your trade history.

        WHY FETCH TRADE HISTORY ON STARTUP?
        The bot's PerformanceTracker needs historical trade data to calculate P&L,
        win rate, and Sharpe ratio. Without this, the performance dashboard would
        show empty stats after every restart, even if you've made hundreds of trades.

        By loading the last 200 executed orders on startup, the bot has a meaningful
        performance history immediately.

        WHAT IS P&L?
        P&L = Profit and Loss. For a resolved YES contract you bought at 40¢:
            P&L = settlement value ($1.00) - cost paid (40¢) - fee ≈ +58¢

        Used to populate the PerformanceTracker on startup so the performance
        tab shows real historical P&L instead of an empty state.

        Returns each order enriched with parsed integer cents values from
        the _dollars/_fp fixed-point fields.
        """
        try:
            data = self._request("GET", "/portfolio/orders", params={"status": "executed", "limit": limit})
        except Exception as e:
            logger.warning(f"[get_executed_orders] Failed: {e}")
            return []
        orders = data.get("orders", []) if data else []
        for o in orders:
            if o.get("yes_price_dollars") is not None:
                o["yes_price"] = _dollars_to_cents(o["yes_price_dollars"])
            if o.get("no_price_dollars") is not None:
                o["no_price"] = _dollars_to_cents(o["no_price_dollars"])
            if o.get("fill_count_fp") is not None:
                o["fill_count"] = _parse_fp(o["fill_count_fp"])
            elif "fill_count" not in o:
                o["fill_count"] = _parse_fp(o.get("count_fp")) if o.get("count_fp") is not None else (o.get("count", 0) or 0)
        logger.info(f"[get_executed_orders] {len(orders)} executed orders from Kalshi")
        return orders

    def close(self):
        """Close the underlying HTTP client connection pool.

        WHAT IS A CONNECTION POOL?
        httpx.Client maintains a pool of open TCP connections to Kalshi's servers.
        TCP connections take time to open (a "handshake" process), so reusing existing
        connections for multiple requests is much faster than opening a new one each time.

        When the bot shuts down, we should cleanly close these connections rather than
        abandoning them — otherwise the server might hold them open unnecessarily.
        This is called "graceful shutdown" and is good programming practice.
        """
        self._client.close()
