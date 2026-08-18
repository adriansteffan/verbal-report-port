"""Shared client for LLM extracted features."""

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

DEFAULT_MODEL = "qwen3.6:27b"

OUTPUT.mkdir(exist_ok=True)
_db = sqlite3.connect(OUTPUT / "llm_cache.sqlite")
# the request is stored next to the reply so a run can be read back afterwards
_db.execute(
    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT,"
    " seed INTEGER, messages TEXT, response TEXT)"
)


# which (config, participant) used which call. Separate from `cache` because
# the same call is shared whenever two configs ask the same question - keying it
# on the content is what makes that sharing possible, and this records the rest.
_db.execute(
    "CREATE TABLE IF NOT EXISTS calls (key TEXT, config TEXT, participant TEXT,"
    " PRIMARY KEY (key, config, participant))"
)


def judge(
    messages: list[dict],
    response_format: dict,
    seed: int = 0,
    model: str = DEFAULT_MODEL,
    config: str = "",
    participant: str = "",
) -> dict:

    key = hashlib.sha256(
        json.dumps([model, messages, response_format, seed], sort_keys=True).encode()
    ).hexdigest()

    if config or participant:
        _db.execute(
            "INSERT OR IGNORE INTO calls VALUES (?, ?, ?)", (key, config, participant)
        )
        _db.commit()  # a cache hit commits nothing else, so this would be lost

    hit = _db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()

    if hit:
        content = hit[0]
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            response_format=response_format,  # pyright: ignore[reportArgumentType]
            max_tokens=8000,  # thinking models need headroom
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
        _db.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?, ?)",
            (key, model, seed, json.dumps(messages), content),
        )
        _db.commit()
    return parsed
