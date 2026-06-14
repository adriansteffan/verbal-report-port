import hashlib
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import participants

HERE = Path(__file__).resolve().parent
RESSOURCES = HERE.parent / "ressources"
OUT = HERE / "output" / "embeddings.parquet"
MODEL = "qwen3-embedding:8b"
N_PARTICIPANTS = None  # limit for testing, None for all


load_dotenv(HERE / ".env")
client = OpenAI(
    base_url=os.environ.get("OPENWEBUI_BASE_URL", "https://ai.cogpsy.fun/api"),
    api_key=os.environ["OPENWEBUI_API_KEY"],
)


def text_hash(text: str) -> str:
    return hashlib.sha256(f"{MODEL}\x00{text}".encode()).hexdigest()[:16]


def embed(texts: list[str]) -> list[list[float]]:
    return [
        d.embedding for d in client.embeddings.create(model=MODEL, input=texts).data
    ]


def texts_to_embed(participant: str) -> list[dict]:
    """Metadata + text for one participant's utterances and four scopes."""
    df = pd.read_csv(RESSOURCES / participant / "transcriptions.csv").sort_values(
        "filename"
    )
    df = df[~df["filename"].str.contains("ruledetection")]
    parsed = df["filename"].str.extract(r"audio_\d+_(?P<phase>.+)_(?P<idx>\d+)\.wav")
    offset = {"acquisition": 0, "transfer": 24}  # global round = idx + offset
    df = df.assign(
        phase=parsed["phase"].values,
        round=parsed["idx"].astype(int).values + parsed["phase"].map(offset).values,
        text=df["text"].fillna("").str.strip(),
    )
    df = df[df["text"].map(participants.is_english)]

    rows = [
        {
            "kind": "utterance",
            "phase": r.phase,
            "round": r.round,
            "key": r.filename,
            "text": r.text,
        }
        for r in df.itertuples()
    ]
    scopes = {
        "full": df["text"],
        "acquisition": df.loc[df["phase"] == "acquisition", "text"],
        "transfer": df.loc[df["phase"] == "transfer", "text"],
        "seqgen": df.loc[df["phase"] == "seqgen", "text"],
    }
    for kind, texts in scopes.items():
        if len(texts):
            rows.append(
                {
                    "kind": kind,
                    "phase": None,
                    "round": None,
                    "key": None,
                    "text": "\n\n".join(texts),
                }
            )
    return rows


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    cache = pd.read_parquet(OUT).to_dict("records") if OUT.exists() else []
    store = {r["text_hash"]: r["embedding"] for r in cache}  # hash -> vector, reused
    done = {(r["participant"], r["kind"], r["key"]) for r in cache}

    folders = sorted(
        p.name for p in RESSOURCES.iterdir() if (p / "transcriptions.csv").exists()
    )
    for pid in folders[:N_PARTICIPANTS]:
        rows = [
            r for r in texts_to_embed(pid) if (pid, r["kind"], r["key"]) not in done
        ]
        if not rows:
            continue
        for r in rows:
            r["text_hash"] = text_hash(r["text"])
        missing = {
            r["text_hash"]: r["text"] for r in rows if r["text_hash"] not in store
        }
        if missing:
            store.update(zip(missing, embed(list(missing.values()))))
        for r in rows:
            cache.append(
                {
                    "participant": pid,
                    "kind": r["kind"],
                    "phase": r["phase"],
                    "round": r["round"],
                    "key": r["key"],
                    "text_hash": r["text_hash"],
                    "embedding": store[r["text_hash"]],
                }
            )
            done.add((pid, r["kind"], r["key"]))
        pd.DataFrame(cache).to_parquet(OUT, index=False)
        n_calls = len(missing)
        print(
            f"{pid}: +{len(rows)} rows ({n_calls} embedded, {len(rows) - n_calls} reused)"
        )

    print(f"done; {OUT}")
