import functools
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # make detection deterministic

HERE = Path(__file__).resolve().parent
RESSOURCES = HERE.parent / "ressources"

# Calibrated on hand-checked cases: lowest confirmed-real participant had ~360
# English chars, highest pure-hallucination case had ~35. 200 sits safely between.
MIN_ENGLISH_CHARS = 200


def all_participants() -> list[str]:
    return sorted(p.name for p in RESSOURCES.iterdir() if p.is_dir())


def is_english(text: str) -> bool:
    """True if langdetect flags the text as English. False for empty/undetectable
    strings (which is how Whisper hallucinations on silence show up)."""
    text = (text or "").strip()
    if not text:
        return False
    try:
        return any(d.lang == "en" and d.prob > 0.5 for d in detect_langs(text))
    except Exception:  # langdetect raises on undetectable strings
        return False


def english_char_count(participant: str) -> int:
    """Characters across recordings whose text is English (pre-debrief only,
    matching the LLM-labeling input in main.py)."""
    f = RESSOURCES / participant / "transcriptions.csv"
    if not f.exists():
        return 0
    df = pd.read_csv(f)
    df = df.loc[~df["filename"].str.contains("ruledetection")]
    return sum(len(t.strip()) for t in df["text"].dropna() if is_english(t))


@functools.cache
def mic_failures() -> frozenset[str]:
    return frozenset(
        p for p in all_participants() if english_char_count(p) < MIN_ENGLISH_CHARS
    )


def excluded() -> frozenset[str]:
    """Participants the labmate dropped for poor performance (NaN id in df_vr)."""
    vr = pd.read_csv(HERE / "df_vr.csv")
    return frozenset(vr.loc[vr["id"].isna(), "participant"])


def analyzable(participants: list[str] | None = None) -> list[str]:
    """Participant ids minus mic-failures and labmate-excluded ones."""
    pool = set(participants) if participants is not None else set(all_participants())
    return sorted(pool - mic_failures() - excluded())


def filter_df(df: pd.DataFrame, col: str = "participant") -> pd.DataFrame:
    """Drop rows whose participant is a mic-failure or labmate-excluded."""
    return df.loc[~df[col].isin(list(mic_failures() | excluded()))]


if __name__ == "__main__":
    mic, exc = mic_failures(), excluded()
    print(f"{len(all_participants())} folders")
    print(
        f"{len(mic)} mic-failures, {len(exc)} labmate-excluded "
        f"({len(mic & exc)} both), {len(analyzable())} analyzable"
    )
    print("\nmic-failure ids:")
    for p in sorted(mic):
        print(f"  {p}  ({english_char_count(p)} English chars)")
