"""Shared client for LLM extracted features."""

import contextlib
import hashlib
import json
import os
import sqlite3

from dotenv import load_dotenv
from openai import OpenAI

from utils.paths import OUTPUT, ROOT

load_dotenv(ROOT / ".env")
client = OpenAI(
    base_url=os.environ.get("OPENWEBUI_BASE_URL", "https://ai.cogpsy.fun/api"),
    api_key=os.environ["OPENWEBUI_API_KEY"],
)

DEFAULT_MODEL = "qwen3.8:27b"

OUTPUT.mkdir(exist_ok=True)
db = sqlite3.connect(OUTPUT / "llm_cache.sqlite")
db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")


_captured: list[dict] | None = None


@contextlib.contextmanager
def capture():
    """Collect every request made inside the block, with its reply, in order.
    pipeline.export_calls() uses this to read back what a config asked.
    Contextvars would be cleaner, but we don't need multithreading anyways"""
    global _captured
    _captured = []
    try:
        yield _captured
    finally:
        _captured = None


def judge(
    messages: list[dict],
    response_format: dict,
    seed: int = 0,
    model: str = DEFAULT_MODEL,
) -> dict:

    key = hashlib.sha256(
        json.dumps([model, messages, response_format, seed], sort_keys=True).encode()
    ).hexdigest()

    hit = db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()

    if hit:
        content = hit[0]
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            response_format=response_format,  # pyright: ignore[reportArgumentType]
            max_tokens=8000,
            seed=seed,
            temperature=0.6,
        )
        content = (response.choices[0].message.content or "").split("</think>")[-1]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except ValueError as e:  # JSONDecodeError
        raise ValueError(
            f"{model} returned no usable JSON.Raw: {content[:200]!r}"
        ) from e

    if not hit:
        db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, content))
        db.commit()

    if _captured is not None:
        _captured.append({"seed": seed, "messages": messages, "response": content})
    return parsed
