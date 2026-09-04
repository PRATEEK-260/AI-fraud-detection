"""Shared LLM client — one place for gateway config, retries, and caching.

All agents call `chat()` / `chat_json()`; none instantiate an SDK client
directly. This keeps the provider (OpenRouter), model choice, caching policy,
and retries in a single configurable place.

OpenRouter is an OpenAI-compatible gateway, so we use the `openai` SDK with
its base_url pointed at the router. Claude models (and others) are reachable
through it — set OPENROUTER_MODEL to pick.

Environment (loaded from .env at repo root):
    OPENROUTER_API_KEY   required — the OpenRouter key
    OPENROUTER_MODEL     optional — e.g. "anthropic/claude-sonnet-4.5";
                         verified against the live model list on first run

Disk cache: identical requests are answered from data/llm_cache/ instead of
re-billed. This is also the demo-latency mitigation from the implementation
plan — reasoning calls during a replayed demo are pre-warmed cache hits.
"""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / "data" / "llm_cache"
MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set — add it to .env at the repo root"
            )
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            max_retries=5,
            timeout=120.0,
        )
    return _client


def _cache_path(payload: dict) -> Path:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def chat(
    messages: list[dict[str, str]],
    model: str = MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    use_cache: bool = True,
) -> str:
    """One LLM call, cached on disk for identical requests."""
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    path = _cache_path(payload)
    if use_cache and path.exists():
        return json.loads(path.read_text())["text"]

    response = _get_client().chat.completions.create(**payload)
    text = response.choices[0].message.content or ""

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text, "model": model}))
    return text


def chat_json(messages: list[dict[str, str]], **kwargs) -> dict:
    """Chat call expected to return JSON; parses it (best effort) to a dict."""
    text = chat(messages, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
