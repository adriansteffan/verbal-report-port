"""Shared client for LLM extracted features."""

import contextlib
import hashlib
import json
import os
import sqlite3
import threading

from dotenv import load_dotenv
from openai import OpenAI

from utils.paths import OUTPUT, ROOT

load_dotenv(ROOT / ".env")
client = OpenAI(
    base_url=os.environ.get("OPENWEBUI_BASE_URL", "https://ai.cogpsy.fun/ollama/v1"),
    api_key=os.environ["OPENWEBUI_API_KEY"],
)

DEFAULT_MODEL = "qwen3.8:27b"
TRIES = 3  # a reply that never closes its JSON is worth asking again; super rare model failure failsafe
MAX_TOKENS = 2500

OUTPUT.mkdir(exist_ok=True)
db = sqlite3.connect(OUTPUT / "llm_cache.sqlite", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
_db_lock = threading.Lock()


_captured: list[dict] | None = None


@contextlib.contextmanager
def capture():
    """Collect every request made inside the block, with its reply, in order.
    pipeline.export_calls() uses this to read back what a config asked. cleaner
    solutions exists for this, whatevs"""
    global _captured
    _captured = []
    try:
        yield _captured
    finally:
        _captured = None


def _complete(
    messages: list[dict], response_format: dict, seed: int, model: str
) -> str:
    content = ""
    for _ in range(TRIES):
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            response_format=response_format,  # pyright: ignore[reportArgumentType]
            max_tokens=MAX_TOKENS,
            seed=seed,
            temperature=0.6,
        )
        content = (response.choices[0].message.content or "").split("</think>")[-1]
        content = content.strip()
        try:
            json.JSONDecoder().raw_decode(content)
            return content
        except ValueError:
            continue
    raise ValueError(
        f"{model} returned no usable JSON in {TRIES} tries. Raw: {content[:200]!r}"
    )


def judge(
    messages: list[dict],
    response_format: dict,
    seed: int = 0,
    model: str = DEFAULT_MODEL,
) -> dict:

    key = hashlib.sha256(
        json.dumps([model, messages, response_format, seed], sort_keys=True).encode()
    ).hexdigest()

    with _db_lock:
        hit = db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()

    content = hit[0] if hit else _complete(messages, response_format, seed, model)

    # a model sometimes carries on chatting past the closing brace, so take the
    # object and ignore whatever follows
    parsed, _ = json.JSONDecoder().raw_decode(content)

    if not hit:
        with _db_lock:
            db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, content))
            db.commit()

    if _captured is not None:
        _captured.append({"seed": seed, "messages": messages, "response": content})
    return parsed
