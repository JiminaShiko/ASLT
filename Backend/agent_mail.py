"""
AgentMail channel for the ASLT backend agent.

The backend keeps its own email address here: it opens (or reuses) one AgentMail
inbox, sends messages from it, and answers the replies it receives through the
signature-verified AgentMail webhook. A reply is built the same way the "Text to
Sign" panel signs a sentence (user.text_to_sign.plan), so the agent answers word by
word with whether the app uses the word model's sign or fingerspells it.

Credentials come from the environment only (.env at the repository root, through
config/environment.py):
  AGENTMAIL_API_KEY          API key from https://console.agentmail.to
  AGENTMAIL_WEBHOOK_SECRET   signing secret (whsec_...) of the AgentMail webhook
                             pointed at /api/agent/email/webhook

Routes:
  POST /api/agent/email/send     send a message from the agent inbox (signed-in user)
  POST /api/agent/email/webhook  AgentMail events; a received message is answered

Check the key and open the inbox from a shell (from Backend/):
  python agent_mail.py                       # opens (or reuses) the inbox
  python agent_mail.py you@example.com       # also sends one test email
"""

from __future__ import annotations

import html
import json
import os
import sys
import threading

from flask import Blueprint, jsonify, request, session

from config.environment import load_environment
from user.text_to_sign import plan as sign_plan

load_environment()

# Fixed client id, so the API hands back the same inbox on every restart instead
# of creating a new address each time.
AGENT_INBOX_CLIENT_ID = "aslt-backend-agent-v1"
AGENT_DISPLAY_NAME = "ASLT Sign Language Translator"

# The SDKs are used by this channel only; a guarded import keeps the rest of the
# website running until requirements.txt has been installed.
try:
    from agentmail import AgentMail
    from agentmail.inboxes import CreateInboxRequest
except ImportError:  # pragma: no cover - reported by _client()
    AgentMail = None
    CreateInboxRequest = None

try:
    from svix.webhooks import Webhook
except ImportError:  # pragma: no cover - reported by the webhook route
    Webhook = None

agent_mail_bp = Blueprint("agent_mail_api", __name__)

_LOCK = threading.Lock()
_INBOX = None

_MISSING_SDK = (
    "The AgentMail SDK is not installed. Run `pip install -r requirements.txt` "
    "from the repository root."
)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip().strip('"').strip("'")


def _client():
    """AgentMail client for the API key in the environment."""
    if AgentMail is None:
        raise RuntimeError(_MISSING_SDK)
    api_key = _env("AGENTMAIL_API_KEY")
    if not api_key:
        raise RuntimeError("AGENTMAIL_API_KEY is not set in the environment (.env).")
    return AgentMail(api_key=api_key)


def inbox(api=None):
    """The agent's own inbox, created once and reused afterward."""
    global _INBOX
    with _LOCK:
        if _INBOX is None:
            if CreateInboxRequest is None:
                raise RuntimeError(_MISSING_SDK)
            api = api or _client()
            _INBOX = api.inboxes.create(
                request=CreateInboxRequest(
                    client_id=AGENT_INBOX_CLIENT_ID,
                    display_name=AGENT_DISPLAY_NAME,
                )
            )
        return _INBOX


def send_email(to: str, subject: str, text: str, html_body: str | None = None) -> dict:
    """Send one message from the agent inbox."""
    api = _client()
    box = inbox(api)
    sent = api.inboxes.messages.send(
        box.inbox_id, to=to, subject=subject, text=text, html=html_body
    )
    return {
        "inbox_id": box.inbox_id,
        "from": box.email,
        "message_id": sent.message_id,
        "thread_id": sent.thread_id,
    }


def reply_to_message(message: dict, text: str, html_body: str | None = None) -> dict:
    """Reply inside the thread of a received message."""
    sent = _client().inboxes.messages.reply(
        message["inbox_id"], message["message_id"], text=text, html=html_body
    )
    return {"message_id": sent.message_id, "thread_id": sent.thread_id}


def sign_answer(body: str) -> str:
    """Plain-text explanation of how the app would sign the emailed sentence."""
    try:
        result = sign_plan(body[:200])
    except Exception:  # noqa: BLE001 - a lookup problem must not break the reply
        result = None

    lines = []
    for item in (result or {}).get("items") or []:
        if item.get("type") != "word":
            continue
        if item.get("sign") == "word":
            lines.append(f"  {item['text']}: sign for '{item['word']}'")
            continue
        spelled = " ".join(letter["letter"] for letter in item.get("letters") or [])
        lines.append(f"  {item['text']}: fingerspelled {spelled or '(no letter images yet)'}")

    if not lines:
        return (
            "Thanks for writing to the ASLT agent. Send an English sentence and "
            "this reply will show, word by word, whether the app signs it or "
            "fingerspells it."
        )

    summary = result["summary"]
    return (
        "How the ASLT app signs your sentence:\n\n"
        + "\n".join(lines)
        + "\n\n"
        + f"{summary['words_signed']} word(s) signed, "
        + f"{summary['words_fingerspelled']} fingerspelled, "
        + f"{summary['letters']} letter sign(s).\n"
        + "Reply with another sentence to try it."
    )


def _answer(message: dict) -> bool:
    """Reply to a received message. Returns True when a reply was sent."""
    body = str(message.get("extracted_text") or message.get("text") or "").strip()
    inbox_id = message.get("inbox_id")
    message_id = message.get("message_id")
    sender = (message.get("from_") or [""])[0]
    if not (inbox_id and message_id and body):
        return False
    if sender and sender.lower() == (inbox().email or "").lower():
        return False  # never answer the agent's own mail
    text = sign_answer(body)
    reply_to_message(message, text=text, html_body=f"<pre>{html.escape(text)}</pre>")
    return True


@agent_mail_bp.post("/api/agent/email/send")
def agent_email_send():
    if not session.get("user_email"):
        return jsonify({"error": "Authentication required."}), 401
    data = request.get_json(silent=True) or {}
    to = str(data.get("to") or "").strip()
    subject = str(data.get("subject") or "").strip() or "ASLT agent"
    text = str(data.get("text") or "").strip()
    if not to or not text:
        return jsonify({"error": "Both 'to' and 'text' are required."}), 400
    try:
        return jsonify(send_email(to=to, subject=subject, text=text))
    except Exception as error:  # noqa: BLE001 - reported to the caller
        return jsonify({"error": f"AgentMail could not send the message: {error}"}), 502


@agent_mail_bp.post("/api/agent/email/webhook")
def agent_email_webhook():
    if Webhook is None:
        return jsonify({"error": _MISSING_SDK}), 503
    secret = _env("AGENTMAIL_WEBHOOK_SECRET")
    if not secret:
        return (
            jsonify({"error": "AGENTMAIL_WEBHOOK_SECRET is not set in the environment (.env)."}),
            503,
        )
    body = request.get_data()
    try:
        Webhook(secret).verify(body, request.headers)
        event = json.loads(body)
    except Exception:  # noqa: BLE001 - a bad signature or body is a bad request
        return jsonify({"error": "Invalid AgentMail webhook signature or payload."}), 400
    if not isinstance(event, dict) or event.get("event_type") != "message.received":
        return "", 204
    try:
        answered = _answer(event.get("message") or {})
    except Exception as error:  # noqa: BLE001 - AgentMail retries a 5xx
        return jsonify({"error": f"Could not answer the message: {error}"}), 500
    return jsonify({"event_id": event.get("event_id"), "replied": answered}), 200


def _main() -> int:
    to = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    try:
        box = inbox()
        print(f"AgentMail answered: inbox {box.email} ({box.inbox_id})")
        if to:
            sent = send_email(
                to=to,
                subject="ASLT agent inbox check",
                text="The ASLT backend agent can send email through AgentMail.",
            )
            print(f"Sent {sent['message_id']} from {sent['from']} to {to}")
    except Exception as error:  # noqa: BLE001 - shown on the console
        print(f"AgentMail did not answer: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())