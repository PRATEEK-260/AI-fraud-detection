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
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / "data" / "llm_cache"
MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-5-nano")

# Reasoning models (the gpt-5 family included) spend their token budget on
# hidden reasoning first. Asked for a short JSON verdict with a 300-token cap,
# gpt-5-nano returned an EMPTY completion after burning 1,152 reasoning tokens
# — billed, and useless. "minimal" effort turns that off and the same call
# returns valid JSON in ~125 tokens. Models without a reasoning mode ignore
# the field, so it is safe to send by default; set REASONING_EFFORT="" to
# disable.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "minimal")

# The model used by the reasoning agents (Content Forensics' detector, Ring
# Detector's verifier, the Adjudicator's arbiter). Kept in one place so a
# single env var re-points all three, and so the README can state plainly
# which model produced the reported numbers.
REASONING_MODEL = os.environ.get("REASONING_MODEL", "openai/gpt-5-nano")

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
    reasoning_effort: str | None = None,
) -> str:
    """One LLM call, cached on disk for identical requests."""
    effort = REASONING_EFFORT if reasoning_effort is None else reasoning_effort
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    # Part of the cache key: the same prompt at a different reasoning effort
    # is a different request and must not collide with a cached answer.
    extra = {"reasoning": {"effort": effort}} if effort else {}
    path = _cache_path({**payload, "_extra": extra})
    if use_cache and path.exists():
        return json.loads(path.read_text())["text"]

    response = _get_client().chat.completions.create(
        **payload, **({"extra_body": extra} if extra else {}))
    text = response.choices[0].message.content or ""

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"text": text, "model": model, "reasoning_effort": effort}))
    return text


class JSONReplyError(ValueError):
    """The model did not return usable JSON. Callers should count these rather
    than let one malformed reply abort a whole evaluation run."""

    def __init__(self, text: str) -> None:
        super().__init__(f"unparseable JSON reply: {text[:200]!r}")
        self.text = text


def _repair_json(text: str) -> dict:
    """Best-effort recovery from the ways small models mangle JSON.

    Seen in practice: prose or a code fence around the object, and — the one
    that actually broke a run — an unescaped double quote inside a reason
    string, which no amount of slicing fixes. For that case the scalar fields
    are pulled out by pattern, since verdict and confidence are all the
    callers strictly need.
    """
    fenced = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(),
                    flags=re.MULTILINE)
    start, end = fenced.find("{"), fenced.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(fenced[start : end + 1])
        except json.JSONDecodeError:
            pass

    out: dict = {}
    for field in ("verdict", "decision", "winning_argument"):
        m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', fenced)
        if m:
            out[field] = m.group(1)
    m = re.search(r'"confidence"\s*:\s*([0-9.]+)', fenced)
    if m:
        try:
            out["confidence"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r'"(?:reasons|rationale)"\s*:\s*(.+)', fenced, re.S)
    if m:
        out["reasons"] = [r.strip(' "\n') for r in
                          re.findall(r'"([^"]{8,})"', m.group(1))][:4]
    if not out:
        raise JSONReplyError(text)
    return out


def chat_json(messages: list[dict[str, str]], **kwargs) -> dict:
    """Chat call expected to return JSON; parses it (best effort) to a dict."""
    text = chat(messages, **kwargs)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _repair_json(text)
