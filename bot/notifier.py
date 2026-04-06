"""
Notification system for trade alerts via Slack, Discord, and Email.

Sends formatted alert messages for key bot events:
  - Trade entries:  Ticker, side, price, size, and signal source (RF or AI).
  - Trade exits:    Entry/exit prices, P&L in dollars.
  - New signals:    Ticker, side, edge percentage, and confidence level.
  - Model retrains: Number of training samples and cross-validation accuracy.
  - Risk alerts:    Threshold breach details.
  - Test messages:  Connectivity verification for configured channels.

Messages use Markdown formatting (Discord-native, converted to Slack bold syntax).
Email uses HTML formatting for rich notifications.
All channels are optional; the bot works without any notifications configured.
All webhook/email calls use timeouts and log errors without crashing.

Connects to: Slack Incoming Webhooks API, Discord Webhooks API, SMTP (via httpx/smtplib).
Configuration: SLACK_WEBHOOK_URL, DISCORD_WEBHOOK_URL, NOTIFICATION_EMAIL,
               SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS environment variables.
Used by: bot.server (auto-trade alerts, notification test endpoint).

---------------------------------------------------------------------------
EDUCATIONAL OVERVIEW
---------------------------------------------------------------------------

WHY SEND NOTIFICATIONS?
  The bot runs continuously as a background process. You're not watching a
  terminal — you're doing other things. Notifications let the bot reach you
  wherever you are (phone, laptop, desktop) whenever something important
  happens: a trade was placed, a position was closed, a risk limit was hit,
  or the model was retrained. Without notifications, you'd have to constantly
  check the server logs or dashboard to know what the bot did.

WHAT IS A SLACK/DISCORD WEBHOOK?
  Slack and Discord both let server admins create "incoming webhooks" —
  special HTTPS URLs that, when you POST a JSON payload to them, cause a
  message to appear in a designated channel. No login is required; the URL
  itself acts as the authentication token. Keep it private (treat it like a
  password). The JSON format differs slightly between platforms:
    Slack:   {"text": "message here"}
    Discord: {"content": "message here"}

WHAT IS SMTP?
  SMTP (Simple Mail Transfer Protocol) is the internet standard for sending
  email. Think of it as the postal system: smtplib connects to an email
  provider's "post office" (e.g., smtp.gmail.com) and deposits a message to
  be delivered. You authenticate with a username and password (or app password).
  Port 587 uses STARTTLS (upgrades a plain connection to encrypted).
  Port 465 uses SSL from the start (always encrypted). Both are secure.

WHAT IS A DEAD LETTER QUEUE?
  A dead letter queue (DLQ) is a holding area for messages that failed to
  deliver after all retry attempts are exhausted. Instead of silently discarding
  failed notifications, the bot stores them in self.dead_letter_queue so you
  can inspect them via the API or logs and understand what went wrong. The DLQ
  is capped at MAX_DEAD_LETTER=50 entries to avoid unbounded memory growth.

WHAT IS EXPONENTIAL BACKOFF?
  When a request fails (e.g., Slack is temporarily down), retrying immediately
  often fails again. Backoff means waiting progressively longer between retries:
  wait 2s, then 4s, then 6s (linear here, but exponential is also common).
  This gives the destination time to recover and avoids hammering a struggling
  server with rapid retries.

WHAT IS MARKDOWN?
  Markdown is a lightweight text formatting syntax: **text** becomes bold,
  *text* becomes italic, and so on. Discord natively renders Markdown in
  messages. Slack uses a similar but slightly different syntax (*bold* instead
  of **bold**), so the _send_slack method converts between them.

WHAT IS MIME?
  MIME (Multipurpose Internet Mail Extensions) is the standard that allows
  email to carry formatted content, attachments, and multiple versions of the
  same message. A "multipart/alternative" MIME message contains both a plain-
  text version (for email clients that don't support HTML) and an HTML version
  (for rich formatting). Email clients pick the best version they can display.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import smtplib   # Python's built-in SMTP library for sending email
import time
from datetime import datetime, timezone
# MIMEMultipart: container for a multi-part email (plain text + HTML)
# MIMEText: a single text part of an email (either plain or HTML)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx

from bot.config import config

# Standard Python logging — messages go to the console / log file.
# "predictionbot" is the logger name; all bot modules use the same name so
# their log output is grouped together and controllable with one logger config.
logger = logging.getLogger("predictionbot")


class Notifier:
    """Sends formatted alert messages to Slack, Discord, and/or Email.

    All notification methods are fire-and-forget: they log errors but never
    raise exceptions, so notification failures never block trading operations.

    DESIGN PRINCIPLE — "NOTIFICATIONS MUST NEVER CRASH TRADING":
    The bot's primary purpose is to trade, not to send messages. If Slack is
    down, Discord is over capacity, or the SMTP server rejects the connection,
    that should NEVER prevent the bot from executing a trade. All notification
    methods catch exceptions internally and log them. The worst case is a
    missed notification — not a missed trade.

    FIRE-AND-FORGET PATTERN:
    "Fire and forget" means the caller sends a notification and immediately
    moves on without waiting for confirmation that it was received. This is
    appropriate for alerts (we don't need to wait for Slack to confirm before
    placing the next order) but means you won't get synchronous feedback about
    failures — hence the dead letter queue and logging.
    """

    # Max retries for failed webhook calls
    # WHY 3? One attempt + 2 retries is a reasonable balance between persistence
    # (giving transient failures a chance to resolve) and giving up quickly
    # (not blocking the bot for too long on a broken notification channel).
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = 2  # Wait 2s after first failure, 4s after second, etc.
    # Max items in the dead letter queue
    # Capped at 50 to prevent unbounded memory growth if notifications keep failing
    MAX_DEAD_LETTER = 50

    def __init__(self):
        """Initialize with a shared HTTP client (10s timeout for webhook calls).

        WHY A SHARED CLIENT? Creating a new httpx.Client for every notification
        is wasteful — it opens a new TCP connection each time. A shared client
        maintains a connection pool (reusing connections to the same host),
        which is faster and uses fewer system resources.

        timeout=10: If Slack or Discord doesn't respond within 10 seconds,
        the attempt is considered failed and retried (or eventually added to
        the dead letter queue). 10 seconds is generous — webhooks typically
        respond in < 1 second when working correctly.
        """
        self._client = httpx.Client(timeout=10)
        self.dead_letter_queue: list[dict] = []  # Failed notifications for inspection

    @property
    def channels(self) -> list[str]:
        """List of configured notification channel names.

        Returns only the channels that have been configured in .env.
        If no Slack URL is set, "slack" won't appear in this list.
        Used by send_test() to report which channels are active, and by
        the API to show the bot's notification status.
        """
        ch = []
        if config.slack_webhook_url:
            ch.append("slack")
        if config.discord_webhook_url:
            ch.append("discord")
        # Email requires BOTH a recipient address AND an SMTP host to be configured
        if config.notification_email and config.smtp_host:
            ch.append("email")
        return ch

    @property
    def is_configured(self) -> bool:
        """Return True if at least one notification channel is configured.

        Used as a quick check before attempting to send. If no channels are
        configured, there's no point constructing the message.
        """
        return len(self.channels) > 0

    def notify_trade_entry(self, ticker: str, side: str, price: float, size_cents: int, source: str = "RF", edge: float = 0.0, confidence: float = 0.0):
        """Send a trade entry alert with ticker, side, price, size, signal source, edge and confidence.

        Called immediately after a trade order is successfully placed on Kalshi.
        Provides a real-time record of every trade the bot executes.

        Args:
            ticker: The Kalshi market ticker (e.g., "KXBTC-24MAR14-T100000").
            side: "yes" or "no" — which side of the contract was purchased.
            price: Entry price as a decimal (e.g., 0.42 = 42 cents).
            size_cents: Dollar size of the trade in cents (e.g., 1000 = $10.00).
            source: Signal source — "RF" for Random Forest model, "AI" for Claude.
            edge: The edge that triggered this trade (e.g., 0.15 = 15% edge).
            confidence: Model confidence at time of entry (0.0 to 1.0).
        """
        msg = (
            f"**ENTRY** {ticker}\n"
            f"Side: {side.upper()} | Price: {price*100:.0f}c | Size: ${size_cents/100:.2f}\n"
            # :.1f formats the float to 1 decimal place; *100 converts decimal to percentage
            f"Edge: {edge*100:.1f}% | Confidence: {confidence*100:.0f}%\n"
            f"Source: {source} | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        self._send(msg, subject=f"Trade Entry: {ticker} {side.upper()}")

    def notify_trade_exit(self, ticker: str, side: str, entry_price: float, exit_price: float, pnl_cents: int):
        """Send a trade exit alert with entry/exit prices and P&L.

        Called when a position is closed (sold) or when a market settles.
        Shows the full round-trip: where we entered, where we exited, and
        how much we made or lost.

        Args:
            ticker: The market ticker.
            side: "yes" or "no".
            entry_price: Original purchase price as decimal (0.0-1.0).
            exit_price: Sale/settlement price as decimal (0.0-1.0).
            pnl_cents: Profit or loss in cents (negative = loss).
        """
        # emoji string variable: "+" prefix if profitable, empty string if a loss.
        # When prepended to the dollar amount, "+$3.50" or "-$1.20" makes the
        # direction of the P&L immediately visible at a glance.
        emoji = "+" if pnl_cents >= 0 else ""
        msg = (
            f"**EXIT** {ticker}\n"
            f"Side: {side.upper()} | Entry: {entry_price*100:.0f}c -> Exit: {exit_price*100:.0f}c\n"
            f"P&L: {emoji}${pnl_cents/100:.2f} | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        self._send(msg, subject=f"Trade Exit: {ticker} P&L {emoji}${pnl_cents/100:.2f}")

    def notify_signal(self, ticker: str, side: str, edge: float, confidence: float):
        """Send a new trading signal alert with edge and confidence percentages.

        Called when the model generates a signal but BEFORE a trade is placed
        (or in dry-run mode where no trade will be placed). Useful for monitoring
        what opportunities the bot is seeing without necessarily acting on them.

        Args:
            ticker: The market ticker for the signal.
            side: Recommended side ("yes" or "no").
            edge: The calculated edge (e.g., 0.12 = 12% edge).
            confidence: Model confidence (0.0-1.0).
        """
        msg = (
            f"**SIGNAL** {ticker}\n"
            f"Side: {side.upper()} | Edge: {edge*100:.1f}% | Confidence: {confidence*100:.0f}%"
        )
        self._send(msg, subject=f"New Signal: {ticker} Edge {edge*100:.1f}%")

    def notify_retrain(self, samples: int, cv_accuracy: float, training_duration_secs: float = 0.0):
        """Send a model retrain completion alert with sample count, accuracy, and duration.

        Called after the Random Forest model finishes retraining on new historical
        data. Useful for monitoring model health: if accuracy is declining over time,
        the model may need more data or feature engineering improvements.

        WHAT IS CROSS-VALIDATION ACCURACY?
        Cross-validation (CV) is a technique for estimating how well a model will
        perform on unseen data. Rather than testing on the training set (which would
        give an overly optimistic score), we split the data into K "folds," train on
        K-1 folds, and test on the remaining fold. We repeat this K times and average
        the accuracy. A CV accuracy of 65% means the model correctly predicted the
        market outcome 65% of the time on data it hadn't seen during training.

        Args:
            samples: Number of historical data points used for training.
            cv_accuracy: Cross-validation accuracy as a decimal (e.g., 0.68 = 68%).
            training_duration_secs: How long the retraining took in seconds.
        """
        # Only show duration if it was provided (> 0)
        duration_str = f" | Duration: {training_duration_secs:.1f}s" if training_duration_secs > 0 else ""
        msg = (
            f"**MODEL RETRAINED**\n"
            f"Samples: {samples} | CV Accuracy: {cv_accuracy*100:.1f}%{duration_str}"
        )
        self._send(msg, subject=f"Model Retrained: {cv_accuracy*100:.1f}% accuracy")

    def notify_alert(self, alerts: list[str]):
        """Send a risk alert notification with threshold breach details.

        Called by the RiskManager when a safety threshold is crossed:
        - Daily loss limit reached
        - Max drawdown exceeded
        - Unusual number of consecutive losses
        These alerts are urgent; they indicate the bot has stopped trading
        and you may want to investigate.

        Args:
            alerts: A list of alert message strings describing the breaches.
        """
        # Build a bulleted list of alert messages.
        # f"- {a}" prefixes each alert with a dash to create a readable list.
        msg = "**RISK ALERT**\n" + "\n".join(f"- {a}" for a in alerts)
        self._send(msg, subject="Risk Alert: Threshold Breached")

    def notify_scan_complete(self, markets_scanned: int, signals_found: int, duration_secs: float = 0.0):
        """Send a scan completion notification.

        Called after each scan loop completes. Provides a regular heartbeat
        confirming the bot is alive and reporting its activity level. If you
        stop receiving these, the bot may have crashed or stalled.

        Args:
            markets_scanned: Total number of Kalshi markets evaluated.
            signals_found: How many met the edge/confidence thresholds.
            duration_secs: How long the scan took in seconds.
        """
        msg = (
            f"**SCAN COMPLETE**\n"
            f"Markets: {markets_scanned} | Signals: {signals_found}\n"
            f"Duration: {duration_secs:.1f}s | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        self._send(msg, subject=f"Scan: {signals_found} signals from {markets_scanned} markets")

    def notify_error(self, error_type: str, details: str):
        """Send an error notification for system issues.

        Called when the bot encounters an unhandled exception or critical
        failure (e.g., Kalshi API returns 500, database is unreachable, model
        file is corrupt). Gives you an immediate alert so you can intervene.

        Args:
            error_type: Short label for the error category (e.g., "KalshiAPIError").
            details: Full error message or traceback snippet.
        """
        msg = (
            f"**ERROR** {error_type}\n"
            f"{details}\n"
            f"{datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        self._send(msg, subject=f"Error: {error_type}")

    def send_test(self) -> dict:
        """Send a test notification to verify connectivity. Returns channel results.

        Use this when first setting up notifications to confirm that webhook URLs
        and SMTP credentials are correct. The server exposes this via
        POST /api/notifications/test. A result of {"slack": "ok"} means the
        Slack webhook accepted the message successfully.

        Returns:
            A dict with keys "channels" (list of configured channels) and
            "results" (dict mapping each channel name to "ok" or "error:...").
        """
        msg = f"**PredictionBot Test** - Notifications working! {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        results = self._send(msg, subject="PredictionBot Test Notification")
        return {"channels": self.channels, "results": results}

    def _send(self, message: str, subject: str = "PredictionBot Alert") -> dict:
        """Dispatch a message to all configured channels. Returns {channel: status} dict.

        This is the internal routing method called by all the public notify_*
        methods. It checks which channels are configured and calls the appropriate
        private send method for each. The subject parameter is only used for email
        (Slack and Discord don't have separate subject lines).

        Args:
            message: The Markdown-formatted message body.
            subject: Email subject line (ignored for Slack/Discord).

        Returns:
            Dict mapping channel name ("slack", "discord", "email") to
            status string ("ok" or "error:...").
        """
        results = {}
        if config.slack_webhook_url:
            results["slack"] = self._send_slack(message)
        if config.discord_webhook_url:
            results["discord"] = self._send_discord(message)
        # Both a recipient email AND an SMTP server are required for email delivery
        if config.notification_email and config.smtp_host:
            results["email"] = self._send_email(message, subject)
        return results

    def _send_with_retry(self, channel: str, url: str, payload: dict, success_codes: tuple = (200,)) -> str:
        """Send a webhook request with retry logic and dead letter queue.

        RETRY LOGIC EXPLAINED:
        Attempt 1: Send immediately.
        If it fails, wait RETRY_BACKOFF_SECONDS * 1 = 2 seconds, then try again.
        If it fails again, wait RETRY_BACKOFF_SECONDS * 2 = 4 seconds, then try again.
        After MAX_RETRIES=3 total attempts, give up and log to dead_letter_queue.

        WHY DIFFERENT SUCCESS CODES FOR DISCORD?
        Slack returns HTTP 200 on success. Discord returns HTTP 204 (No Content)
        on success — it accepted the message but has nothing to return. Treating
        204 as an error would cause all Discord notifications to appear to fail.

        Args:
            channel: Channel name for logging ("slack" or "discord").
            url: The full webhook URL to POST to.
            payload: The JSON payload to send.
            success_codes: HTTP status codes that indicate success.

        Returns:
            "ok" on success, or "error:<message>" after all retries fail.
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                # POST the payload as JSON to the webhook URL
                r = self._client.post(url, json=payload)
                if r.status_code in success_codes:
                    return "ok"
                # HTTP response received but status indicates failure
                last_error = f"HTTP {r.status_code}"
            except Exception as e:
                # Network error, timeout, etc.
                last_error = str(e)

            # Don't sleep after the last attempt — we're about to give up anyway
            if attempt < self.MAX_RETRIES - 1:
                # Linear backoff: wait longer between each successive attempt
                time.sleep(self.RETRY_BACKOFF_SECONDS * (attempt + 1))

        # All retries exhausted — log to dead letter queue
        logger.error(f"{channel} notification failed after {self.MAX_RETRIES} retries: {last_error}")
        raw_message = payload.get("text") or payload.get("content", "")
        # Truncate long messages in the DLQ to avoid excessive memory usage.
        # [:500] is Python slice syntax: take the first 500 characters.
        self.dead_letter_queue.append({
            "channel": channel,
            "message": raw_message[:500] if len(raw_message) > 500 else raw_message,
            "error": last_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retries": self.MAX_RETRIES,
        })
        # Trim the DLQ to at most MAX_DEAD_LETTER entries.
        # [-MAX_DEAD_LETTER:] keeps only the most recent 50 entries.
        if len(self.dead_letter_queue) > self.MAX_DEAD_LETTER:
            self.dead_letter_queue = self.dead_letter_queue[-self.MAX_DEAD_LETTER:]

        return f"error:{last_error}"

    def _send_slack(self, message: str) -> str:
        """Post to Slack Incoming Webhook with retry.

        Slack uses a slightly different Markdown dialect than Discord:
        *single asterisks* for bold in Slack vs **double asterisks** in Discord.
        The replace("**", "*") converts Discord-style bold to Slack-style bold
        so messages look correct in both platforms without maintaining two
        separate message formatting functions.
        """
        # Convert Discord Markdown (**bold**) to Slack Markdown (*bold*)
        text = message.replace("**", "*")
        return self._send_with_retry("slack", config.slack_webhook_url, {"text": text}, success_codes=(200,))

    def _send_discord(self, message: str) -> str:
        """Post to Discord Webhook with retry.

        Discord natively renders **double-asterisk bold** Markdown, so no
        conversion is needed. Discord returns 204 No Content on success
        (unlike Slack's 200 OK), so both 200 and 204 are accepted as success.
        """
        return self._send_with_retry("discord", config.discord_webhook_url, {"content": message}, success_codes=(200, 204))

    def _send_email(self, message: str, subject: str) -> str:
        """Send an HTML email notification via SMTP with retry logic.

        SMTP PROTOCOL FLOW:
        1. Connect to the SMTP server (e.g., smtp.gmail.com:587)
        2. Call STARTTLS to encrypt the connection (unless port is 465, which
           is SSL from the start)
        3. Log in with username/password
        4. Send the email (SMTP "sendmail" command)
        5. Disconnect ("quit")

        The email has both a plain text part and an HTML part. Email clients
        that support HTML (most modern ones) will display the styled HTML version.
        Older or plain-text clients will fall back to the plain text part.

        Args:
            message: Markdown-formatted message body (converted to HTML).
            subject: Email subject line.

        Returns:
            "ok" on success, or "error:<message>" after all retries fail.
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                # Convert the Markdown message to styled HTML for the email body
                html_body = self._markdown_to_html(message)

                # MIMEMultipart("alternative"): a container for both plain text and HTML.
                # Email clients display the last (best) version they can render.
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                # Use smtp_from if set, otherwise default to the SMTP username
                msg["From"] = config.smtp_from or config.smtp_user
                msg["To"] = config.notification_email

                # Plain text fallback — strip **bold** markers so they don't appear
                # literally in plain-text email clients.
                plain = message.replace("**", "")
                msg.attach(MIMEText(plain, "plain"))
                # HTML version — richer formatting, displayed by most email clients
                msg.attach(MIMEText(html_body, "html"))

                # Choose the correct SSL mode based on port number:
                # Port 465: SSL from the start (SMTP_SSL)
                # Port 587: Plain connection that upgrades to TLS via STARTTLS
                use_ssl = config.smtp_port == 465
                if use_ssl:
                    server = smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=15)
                else:
                    server = smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=15)
                    # STARTTLS upgrades the connection to encrypted TLS.
                    # Must be called BEFORE login to protect credentials.
                    server.starttls()

                try:
                    # Only login if credentials are configured.
                    # Some internal SMTP relays don't require authentication.
                    if config.smtp_user and config.smtp_pass:
                        server.login(config.smtp_user, config.smtp_pass)

                    # sendmail takes: from address, list of recipients, full message string
                    server.sendmail(
                        config.smtp_from or config.smtp_user,
                        config.notification_email,
                        msg.as_string()
                    )
                finally:
                    # Always disconnect from the SMTP server, even if sendmail fails.
                    # The finally block runs whether or not an exception was raised.
                    server.quit()
                logger.info(f"Email notification sent to {config.notification_email}: {subject}")
                return "ok"

            except Exception as e:
                last_error = str(e)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_BACKOFF_SECONDS * (attempt + 1))

        # All retries failed — log to dead letter queue (same pattern as _send_with_retry)
        logger.error(f"Email notification failed after {self.MAX_RETRIES} retries: {last_error}")
        self.dead_letter_queue.append({
            "channel": "email",
            "message": message[:500],
            "error": last_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retries": self.MAX_RETRIES,
        })
        if len(self.dead_letter_queue) > self.MAX_DEAD_LETTER:
            self.dead_letter_queue = self.dead_letter_queue[-self.MAX_DEAD_LETTER:]

        return f"error:{last_error}"

    def _markdown_to_html(self, message: str) -> str:
        """Convert bot markdown messages to styled HTML email.

        WHY CONVERT INSTEAD OF JUST USING HTML DIRECTLY?
        The notify_* methods build messages with Markdown because that format
        works natively in Discord and (with minor conversion) in Slack. Rather
        than maintaining separate HTML templates for email, we write messages
        once in Markdown and convert at send time. This is the DRY principle:
        Don't Repeat Yourself.

        HTML ESCAPING: Before converting Markdown to HTML tags, we escape the
        three characters that have special meaning in HTML:
          & → &amp;   (if unescaped, could break HTML entities)
          < → &lt;    (if unescaped, could be interpreted as an HTML tag)
          > → &gt;    (same)
        This prevents message content from accidentally injecting HTML.

        Args:
            message: Markdown-formatted message string.

        Returns:
            A complete HTML document as a string, styled with inline CSS.
        """
        import re
        # Escape HTML entities before markdown conversion
        # ORDER MATTERS: escape & first, otherwise &lt; would become &amp;lt;
        message = message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # re.sub: a regex replacement. r'\*\*(.+?)\*\*' matches **any text here**
        # and r'<strong>\1</strong>' replaces it with HTML bold tags.
        # The (.+?) is a "capture group" — \1 in the replacement refers back to
        # whatever text was inside the ** delimiters.
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', message)
        # Convert newlines to HTML line breaks for correct rendering in email clients
        html = html.replace('\n', '<br>')

        # Return a complete HTML document with inline CSS styling.
        # Inline CSS (style="...") is used instead of a <style> tag because many
        # email clients strip <style> blocks but preserve inline styles.
        return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0a0a0b;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
  <div style="max-width:520px;margin:24px auto;background:#111113;border:1px solid #1e1e22;border-radius:12px;overflow:hidden;">
    <div style="background:linear-gradient(135deg,#34d399 0%,#22d3ee 100%);padding:16px 24px;">
      <span style="color:#000;font-size:14px;font-weight:700;letter-spacing:0.5px;">PREDICTION<span style="opacity:0.7">BOT</span></span>
    </div>
    <div style="padding:24px;color:#fafafa;font-size:14px;line-height:1.7;">
      {html}
    </div>
    <div style="padding:12px 24px;border-top:1px solid #1e1e22;color:#71717a;font-size:11px;">
      {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} &middot; PredictionBot Alerts
    </div>
  </div>
</body>
</html>"""

    def close(self):
        """Close the underlying HTTP client and release connection resources.

        Call this when the Notifier will no longer be used (e.g., at server
        shutdown). Mirrors the same pattern as DraftKingsClient.close() and
        KalshiClient.close() — all classes that hold an httpx client should
        close it when done to avoid leaked connections.
        """
        self._client.close()
