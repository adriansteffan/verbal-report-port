"""Dump the cached LLM calls to CSV, one row per call.

    uv run src/export_calls.py

Joins the content-keyed reply cache with the record of which (config,
participant) asked for it, so a call that two configs shared appears once per
config. Rows with no config are calls made outside an extractor.
"""

import json
import re

import pandas as pd

from utils.llm import _db
from utils.paths import OUTPUT

OUT = OUTPUT / "llm_calls.csv"

rows = _db.execute("""
    SELECT c.config, c.participant, m.model, m.seed, m.messages, m.response
    FROM cache m LEFT JOIN calls c ON c.key = m.key
    ORDER BY c.config, c.participant
""").fetchall()

records = []
for config, participant, model, seed, messages, response in rows:
    messages = json.loads(messages)
    last = messages[-1]["content"]
    # the unit being judged is the last triple-quoted block; the rest of the
    # prompt is the category catalogue, which is identical across calls
    quoted = re.findall(r'"""(.*?)"""', last, re.DOTALL)
    reply = json.loads(response)
    answer = reply.get("applies", reply.get("categories"))
    records.append(
        {
            "config": " ".join((config or "").split()),  # repr wraps lines
            "participant": participant or "",
            "model": model,
            "seed": seed,
            "turns": len(messages),
            "prompt_chars": sum(len(m["content"]) for m in messages),
            "unit": quoted[-1].strip() if quoted else "",
            "reasoning": reply.get("reasoning", ""),
            "answer": ", ".join(answer) if isinstance(answer, list) else answer,
        }
    )

df = pd.DataFrame(records)
df.to_csv(OUT, index=False)
print(f"{len(df)} calls -> {OUT}")
if len(df):
    print(df.groupby(["model", "config"], dropna=False).size().to_string())
