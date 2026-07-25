import json
import os
import threading
import uuid
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google import genai
from google.cloud import storage

load_dotenv()

app = Flask(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
LOG_BUCKET = os.environ["LOG_BUCKET"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
storage_client = storage.Client()
bucket = storage_client.bucket(LOG_BUCKET)

# Cloud Run is configured with max-instances=1, so this keeps short
# multi-turn Telegram conversations available to the model.
conversations = {}
seen_updates = set()
state_lock = threading.Lock()

SYSTEM_PROMPT = """
You are a rigorous data-analysis agent.

Your task is to solve the latest question in a Telegram conversation.

Questions may:
- Contain data directly in the message.
- Refer to public datasets such as MOSPI.
- Require web research.
- Require arithmetic, statistics, filtering, comparison, or forecasting.
- Be part of a short multi-turn conversation.

Use Google Search when public or current information is needed.
Use code execution when calculation or data processing is useful.

The user will specify the exact JSON shape expected in the Telegram reply.
Your output must contain only the value that belongs under the application's
top-level "answer" key.

Return exactly one valid JSON object with exactly one top-level key:

{"answer": <requested answer>}

Examples:

If the requested Telegram response shape is:
{"answer": 20, "log_url": "<URL>"}

return:
{"answer": 20}

If the requested Telegram response shape is:
{"answer": {"state": "<state name>"}, "log_url": "<URL>"}

return:
{"answer": {"state": "Assam"}}

Rules:
- Return valid JSON only.
- Do not use Markdown code fences.
- Do not include explanations before or after the JSON.
- Do not include log_url; the application adds it.
- Preserve the requested data types.
- Numbers must be JSON numbers, not strings, unless requested otherwise.
- Check calculations carefully.
- Prefer authoritative primary sources.
- Do not invent unavailable values.
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def add_event(events, event_type, **values):
    events.append(
        {
            "timestamp": utc_now(),
            "type": event_type,
            **values,
        }
    )


def parse_model_json(text):
    if not text:
        raise ValueError("Gemini returned an empty response")

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Defensive extraction if Gemini adds accidental surrounding text.
        start = text.find("{")
        end = text.rfind("}")

        if start < 0 or end <= start:
            raise ValueError(
                f"Gemini did not return a JSON object: {text[:500]}"
            )

        value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("Gemini output is not a JSON object")

    if "answer" not in value:
        # If Gemini returns the requested answer object directly,
        # wrap it under the required top-level key.
        value = {"answer": value}

    return {"answer": value["answer"]}


def upload_jsonl(blob_name, events):
    content = "\n".join(
        json.dumps(
            event,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        for event in events
    ) + "\n"

    bucket.blob(blob_name).upload_from_string(
        content,
        content_type="application/x-ndjson",
    )


def send_telegram(chat_id, payload):
    # Compact JSON with no Markdown or surrounding prose.
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    response = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=30,
    )

    response.raise_for_status()


def build_transcript(history):
    return "\n\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in history
    )


def call_gemini(full_prompt, events):
    """
    First try Google Search and code execution together.

    If the current model or free-tier account does not permit that exact
    combination, retry with Search only, followed by a final model-only
    fallback.
    """

    attempts = [
        [
            {"type": "google_search"},
            {"type": "code_execution"},
        ],
        [
            {"type": "google_search"},
        ],
        [],
    ]

    last_error = None

    for tools in attempts:
        try:
            add_event(
                events,
                "model_attempt",
                model=GEMINI_MODEL,
                tools=[tool["type"] for tool in tools],
            )

            arguments = {
                "model": GEMINI_MODEL,
                "input": full_prompt,
            }

            if tools:
                arguments["tools"] = tools

            interaction = gemini_client.interactions.create(**arguments)

            step_types = [
                getattr(step, "type", type(step).__name__)
                for step in getattr(interaction, "steps", [])
            ]

            add_event(
                events,
                "model_attempt_succeeded",
                interaction_id=interaction.id,
                tools=[tool["type"] for tool in tools],
                step_types=step_types,
            )

            return interaction

        except Exception as error:
            last_error = error

            add_event(
                events,
                "model_attempt_failed",
                tools=[tool["type"] for tool in tools],
                error_type=type(error).__name__,
                message=str(error),
            )

    raise last_error or RuntimeError("All Gemini attempts failed")


def solve_question(chat_id, message_text, events):
    chat_key = str(chat_id)

    with state_lock:
        history = conversations.setdefault(chat_key, [])

        history.append(
            {
                "role": "user",
                "content": message_text,
            }
        )

        # Keep only recent context to control token usage.
        history = history[-10:]
        conversations[chat_key] = history

    transcript = build_transcript(history)

    add_event(
        events,
        "agent_request",
        model=GEMINI_MODEL,
        transcript=transcript,
    )

    full_prompt = f"""
{SYSTEM_PROMPT}

CONVERSATION TRANSCRIPT:

{transcript}

Solve the latest USER message.

Return exactly one JSON object with exactly one top-level key named
"answer". Do not include log_url, citations, Markdown, or explanatory
text in the final output.
"""

    interaction = call_gemini(full_prompt, events)
    raw_output = interaction.output_text
    parsed = parse_model_json(raw_output)

    add_event(
        events,
        "agent_response",
        interaction_id=interaction.id,
        raw_output=raw_output,
        parsed_answer=parsed["answer"],
    )

    with state_lock:
        conversations[chat_key].append(
            {
                "role": "assistant",
                "content": json.dumps(
                    parsed,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

        conversations[chat_key] = conversations[chat_key][-10:]

    return parsed["answer"]


@app.get("/")
def health():
    return jsonify(
        {
            "status": "ok",
            "provider": "gemini",
            "model": GEMINI_MODEL,
        }
    )


@app.post("/telegram")
def telegram_webhook():
    supplied_secret = request.headers.get(
        "X-Telegram-Bot-Api-Secret-Token",
        "",
    )

    if supplied_secret != WEBHOOK_SECRET:
        return jsonify({"error": "unauthorized"}), 403

    update = request.get_json(silent=True) or {}
    update_id = update.get("update_id")

    with state_lock:
        if update_id in seen_updates:
            return jsonify(
                {
                    "ok": True,
                    "duplicate": True,
                }
            )

        seen_updates.add(update_id)

        # Avoid keeping update IDs forever.
        if len(seen_updates) > 5000:
            seen_updates.clear()
            seen_updates.add(update_id)

    message = update.get("message") or {}
    text = message.get("text")
    chat_id = (message.get("chat") or {}).get("id")

    # Ignore stickers, photos, service events, and other non-text updates.
    if not text or chat_id is None:
        return jsonify(
            {
                "ok": True,
                "ignored": True,
            }
        )

    run_id = uuid.uuid4().hex
    blob_name = f"runs/{run_id}.jsonl"
    log_url = (
        f"https://storage.googleapis.com/"
        f"{LOG_BUCKET}/{blob_name}"
    )

    events = []

    add_event(
        events,
        "run_started",
        run_id=run_id,
        update_id=update_id,
        chat_id=chat_id,
        question=text,
        provider="gemini",
        model=GEMINI_MODEL,
    )

    try:
        answer = solve_question(
            chat_id=chat_id,
            message_text=text,
            events=events,
        )

        final_reply = {
            "answer": answer,
            "log_url": log_url,
        }

        add_event(
            events,
            "final_reply",
            reply=final_reply,
        )

        add_event(
            events,
            "run_finished",
            status="success",
        )

        # Upload before replying so log_url is already usable.
        upload_jsonl(blob_name, events)
        send_telegram(chat_id, final_reply)

    except Exception as error:
        add_event(
            events,
            "run_failed",
            error_type=type(error).__name__,
            message=str(error),
        )

        try:
            upload_jsonl(blob_name, events)
        except Exception as log_error:
            print(
                "Failed to upload error log:",
                type(log_error).__name__,
                str(log_error),
                flush=True,
            )

        fallback_reply = {
            "answer": None,
            "log_url": log_url,
        }

        try:
            send_telegram(chat_id, fallback_reply)
        except Exception as telegram_error:
            print(
                "Failed to send Telegram fallback:",
                type(telegram_error).__name__,
                str(telegram_error),
                flush=True,
            )

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )