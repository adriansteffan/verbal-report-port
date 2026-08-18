"""The cluster/category scheme from vr_prompts-examples.xlsx."""

import functools

import pandas as pd

from utils.paths import TAXONOMY

# Residual class. Kept as an answer option so top-k has somewhere to put speech
# that fits nothing, but dropped from the feature vector, where it would just be
# the near-constant complement of everything else.
ESCAPE = "Not otherwise specified"

PROMPT_COLUMN = {
    "utterance": "PROMPT",
    "group": "PROMPT_group",
    "transcript": "PROMPT_transcript",
}

# what the prompt calls the unit, matching the wording of the variant above
UNIT_LABEL = {"utterance": "Utterance", "group": "Passage", "transcript": "Transcript"}


@functools.cache
def _table() -> pd.DataFrame:
    return pd.read_excel(TAXONOMY)


def categories(include_escape: bool = False) -> list[str]:
    names = _table()["Category"].tolist()
    return names if include_escape else [c for c in names if c != ESCAPE]


def clusters() -> dict[str, list[str]]:
    """Cluster -> its categories. Cluster scores are pooled from these"""
    t = _table()
    t = t[t["Category"] != ESCAPE]
    return {c: g["Category"].tolist() for c, g in t.groupby("Cluster", sort=False)}


@functools.cache
def _column(name: str) -> dict[str, str]:
    """category -> value lookup"""
    t = _table().set_index("Category")[name]
    return {k: ("" if pd.isna(v) else str(v)) for k, v in t.items()}


def prompt(category: str, prompt_granularity: str) -> str:
    return _column(PROMPT_COLUMN[prompt_granularity])[category]


def examples(category: str) -> str:
    """Few-shot examples for one category, empty for the residual class."""
    return _column("Most diagnostic examples")[category]
