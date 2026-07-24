# Telegram Data Analyst Bot

A Telegram bot that answers data-analysis questions using an LLM agent,
web search, and Python-based analysis.

Each reply is exactly one JSON object:

```json
{"answer": {}, "log_url": "https://storage.googleapis.com/bucket/runs/run.jsonl"}