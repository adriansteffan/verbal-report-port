"""The cluster/category scheme from vr_prompts-v2_1.xlsx."""

import functools
import random
import re

import pandas as pd

from utils.paths import TAXONOMY

# Residual class. Kept as an answer option so top-k has somewhere to put speech
# that fits nothing, but dropped from the feature vector, where it would just be
# the near-constant complement of everything else.
ESCAPE = "Not otherwise specified"

# what the prompt calls the unit
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
    return {
        str(cluster): list(group["Category"])
        for cluster, group in t.groupby("Cluster", sort=False)
    }


@functools.cache
def _column(name: str) -> dict[str, str]:
    """category -> value lookup"""
    t = _table()
    # anything that is not a string is a blank cell, which pandas reads as NaN
    return {
        str(category): value if isinstance(value, str) else ""
        for category, value in zip(t["Category"], t[name])
    }


def prompt(category: str) -> str:
    return _column("PROMPT")[category]


def examples(category: str, rng: random.Random | None = None) -> str:
    """Few-shot examples for one category, potentially shuffled, empty for the residual class."""
    shots = _column("Examples")[category]
    if rng is None:
        return shots
    snippet = r'"[^"]*"'
    found = re.findall(snippet, shots)
    picked = iter(rng.sample(found, len(found)))
    return re.sub(snippet, lambda _: next(picked), shots)
