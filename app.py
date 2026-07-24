import json
import os
import threading
import uuid
from datetime import datetime, timezone

import requests
from flask import Flask, jsonify, request
from google.cloud import storage
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
LOG_BUCKET = os.environ["LOG_BUCKET"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")

openai_client = OpenAI()
storage_client = storage.Client()
bucket = storage_client.bucket(LOG_BUCKET)

# Cloud Run is configured with max-instances=1 so short Telegram
# conversations normally retain their context.
conversations = {}
seen_updates = set()
state_lock = threading.Lock()

SYSTEM_PROMPT = """
You are a rigorous data-analysis agent.

Solve the latest Telegram question accurately. Questions can contain inline
data or refer to public datasets such as MOSPI. Use web search for public
sources and Code Interpreter for calculations whenever useful.

The conversation transcript may contain earlier messages. Answer the latest
user message, while using earlier messages as context.

The user's latest message specifies the exact shape required inside "answer".
Return exactly one valid JSON object with one key named "answer".

Example:
{"answer":{"state":"Assam"}}

Rules:
- Do not include markdown or prose.
- Do not invent unavailable values.
- Check calculations carefully.
- Preserve requested types: numbers as numbers, arrays as arrays, etc.
- Do not include log_url; the application adds that after validation.
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def add_event(events, event_type, **values):
    events.append({
        "timestamp": utc_now(),
        "type": event_type,
        **values,
    })


def parse_model_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```").strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return a JSON object")
        value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("Model output is not a JSON object")

    if "answer" not in value:
        # Defensive fallback if the model returned the requested shape directly.
        value = {"answer": value}

    return {"answer": value["answer"]}


def upload_jsonl(blob_name, events):
    content = "\n".join(
        json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        for event in events
    ) + "\n"

    bucket.blob(blob_name).upload_from_string(
        content,
        content_type="application/x-ndjson",
    )


def send_telegram(chat_id, payload):
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    response = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=30,
    )
    response.raise_for_status()


def solve_question(chat_id, message_text, events):
    with state_lock:
        history = conversations.setdefault(str(chat_id), [])
        history.append({"role": "user", "content": message_text})
        history = history[-10:]
        conversations[str(chat_id)] = history

    transcript = "\n\n".join(
        f"{item['role'].upper()}: {item['content']}"
        for item in history
    )

    add_event(
        events,
        "agent_request",
        model=OPENAI_MODEL,
        transcript=transcript,
    )

    tools = [
        {
            "type": "web_search",
            "search_context_size": "medium",
        },
        {
            "type": "code_interpreter",
            "container": {"type": "auto", "memory_limit": "4g"},
        },
    ]

    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "medium"},
            tools=tools,
            instructions=SYSTEM_PROMPT,
            input=transcript,
        )
    except Exception as first_error:
        # Fallback in case Code Interpreter is unavailable for the account.
        add_event(
            events,
            "tool_fallback",
            error_type=type(first_error).__name__,
            message=str(first_error),
        )
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            reasoning={"effort": "medium"},
            tools=[{"type": "web_search", "search_context_size": "medium"}],
            instructions=SYSTEM_PROMPT,
            input=transcript,
        )

    raw_output = response.output_text
    parsed = parse_model_json(raw_output)

    add_event(
        events,
        "agent_response",
        response_id=response.id,
        raw_output=raw_output,
        parsed_answer=parsed["answer"],
    )

    with state_lock:
        conversations[str(chat_id)].append({
            "role": "assistant",
            "content": json.dumps(parsed, ensure_ascii=False),
        })
        conversations[str(chat_id)] = conversations[str(chat_id)][-10:]

    return parsed["answer"]


@app.get("/")
def health():
    return jsonify({"status": "ok"})


@app.post("/telegram")
def telegram_webhook():
    supplied_secret = request.headers.get(
        "X-Telegram-Bot-Api-Secret-Token", ""
    )
    if supplied_secret != WEBHOOK_SECRET:
        return jsonify({"error": "unauthorized"}), 403

    update = request.get_json(silent=True) or {}
    update_id = update.get("update_id")

    with state_lock:
        if update_id in seen_updates:
            return jsonify({"ok": True, "duplicate": True})
        seen_updates.add(update_id)

        if len(seen_updates) > 5000:
            seen_updates.clear()

    message = update.get("message") or {}
    text = message.get("text")
    chat_id = (message.get("chat") or {}).get("id")

    if not text or chat_id is None:
        return jsonify({"ok": True, "ignored": True})

    run_id = uuid.uuid4().hex
    blob_name = f"runs/{run_id}.jsonl"
    log_url = f"https://storage.googleapis.com/{LOG_BUCKET}/{blob_name}"
    events = []

    add_event(
        events,
        "run_started",
        run_id=run_id,
        update_id=update_id,
        chat_id=chat_id,
        question=text,
    )

    try:
        answer = solve_question(chat_id, text, events)

        final_reply = {
            "answer": answer,
            "log_url": log_url,
        }

        add_event(events, "final_reply", reply=final_reply)
        add_event(events, "run_finished", status="success")

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
        except Exception:
            pass

        # Still return exactly one JSON object to Telegram.
        fallback = {
            "answer": None,
            "log_url": log_url,
        }
        send_telegram(chat_id, fallback)

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
    )