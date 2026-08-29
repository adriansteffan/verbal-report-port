import functools
import re

import pandas as pd
from langdetect import DetectorFactory, detect_langs

from utils.paths import DF_VR, RESSOURCES

DetectorFactory.seed = 0  # langdetect samples n-grams at random; we pin the RNG

PHASE_OFFSET = {"acquisition": 0, "transfer": 24}
LAST_ROUND = 48


def _english_prob(text: str) -> float:
    try:
        return next((d.prob for d in detect_langs(text) if d.lang == "en"), 0.0)
    except Exception:  # langdetect raises on undetectable strings
        return 0.0


def is_artifact(text: str) -> bool:
    """Detect whisper hallucinates on silence, which
    come back either in another script or as a long passage in another language"""
    text = (text or "").strip()
    if not text:
        return True
    if not re.search(r"[A-Za-z]", text):  # No latin characters at all?
        return True
    # Lngdetect is unreliable at small utterances
    MIN_CHARS_FOR_LANGUAGE_ID = 60
    # Drop only when English is essentially ruled out (rather than when some
    # other language just wins). Repetitive real speech flattens the n-gram profile and gets confidently
    # mislabelled Tagalog or Afrikaans, but English keeps some probability mass;
    # genuine hallucination leave it at zero.
    return len(text) >= MIN_CHARS_FOR_LANGUAGE_ID and _english_prob(text) < 0.05


@functools.cache
def utterances(participant: str) -> pd.DataFrame:
    """One row per round of the study, holding what the participant said in
    that round (potentially empty).

    This is the single definition of "what the participant actually said" -
    the cohort filter below measures the same text the extractors will see."""
    grid = pd.RangeIndex(1, LAST_ROUND + 1, name="round")
    f = RESSOURCES / participant / "transcriptions.csv"
    if not f.exists():
        # one folder is empty, happens when someone opened the study and recorded nothing
        return pd.DataFrame({"round": grid, "text": ""})

    df = pd.read_csv(f).sort_values("filename", ignore_index=True)
    # drop post-debriefing recording
    df = df.loc[~df["filename"].str.contains("ruledetection")]
    parsed = df["filename"].str.extract(r"audio_\d+_(?P<phase>.+)_(?P<idx>\d+)\.wav")
    df = df.assign(
        round=parsed["idx"].astype(float).to_numpy()
        + parsed["phase"].map(PHASE_OFFSET).to_numpy(),
        text=df["text"].fillna("").str.strip(),
    )
    df = df.loc[(df["text"] != "") & ~df["text"].map(is_artifact)]
    # the phases we are not intrested in (instructions, practice, seqgen)
    # have no round of their own and drop out here
    df = df.dropna(subset=["round"]).astype({"round": int}).set_index("round")
    return df[["text"]].reindex(grid, fill_value="").reset_index()


@functools.cache
def vr() -> pd.DataFrame:
    """The behavioral table, indexed by participant: `aware` and `time`
    (the round behavior shows they got the rule, 25-48, or 0 for never)."""
    return pd.read_csv(DF_VR).set_index("participant")


@functools.cache
def analyzable_participants() -> list[str]:
    """The cohort: everyone with enough English speech to analyse, minus the
    participants dropped for performance (NaN id in df_vr)."""

    MIN_SPEECH_CHARS = 200

    return [
        p
        for p in sorted(p.name for p in RESSOURCES.iterdir() if p.is_dir())
        if p not in set(vr().index[vr()["id"].isna()])
        # Enough real speech to analyse.
        and utterances(p)["text"].str.len().sum() >= MIN_SPEECH_CHARS
        # Language is a property of the speaker, judged once on their whole
        # artifact-free transcript, where detection is reliable.
        and _english_prob("\n\n".join(t for t in utterances(p)["text"] if t)) > 0.5
    ]
