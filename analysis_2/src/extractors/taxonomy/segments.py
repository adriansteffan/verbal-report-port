"""How the in-scope speech is cut into the units the LLM judges, and how the
per-unit scores are folded back into one number per category."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

POOL = {"max": np.max, "mean": np.mean}


def _labelled(units: pd.DataFrame) -> list[str]:
    """The rounds that were spoken in, each tagged with its number so the model
    can tell what was said when"""
    spoken = units[units["text"] != ""]
    return [f"Round {r}: {t}" for r, t in zip(spoken["round"], spoken["text"])]


class Segmenter(BaseEstimator, ABC):
    prompt_granularity: str

    @abstractmethod
    def split(self, units: pd.DataFrame) -> list[str]: ...

    @abstractmethod
    def pool(self, scores: np.ndarray) -> np.ndarray:
        """(units x categories) -> one score per category."""


class Transcript(Segmenter):
    """Everything at once, one unit."""

    prompt_granularity = "transcript"

    def split(self, units: pd.DataFrame) -> list[str]:
        spoken = _labelled(units)
        return ["\n\n".join(spoken)] if spoken else []

    def pool(self, scores: np.ndarray) -> np.ndarray:
        return scores[0]


class _Chunked(Segmenter, ABC):
    """Many units, so the scores need pooling: max = the behavior occurred at
    all, mean = how much of the participant's speech it made up."""

    def __init__(self, pooling: str = "max"):
        self.pooling = pooling

    def pool(self, scores: np.ndarray) -> np.ndarray:
        return POOL[self.pooling](scores, axis=0)


class Groups(_Chunked):
    """One unit per cycle of `size` rounds."""

    prompt_granularity = "group"

    def __init__(self, size: int = 6, pooling: str = "max"):
        super().__init__(pooling=pooling)
        self.size = size

    def split(self, units: pd.DataFrame) -> list[str]:
        cycles = (units["round"] - 1) // self.size
        return ["\n\n".join(_labelled(g)) for _, g in units.groupby(cycles)]


class Utterances(_Chunked):
    """One unit per recording."""

    prompt_granularity = "utterance"

    def split(self, units: pd.DataFrame) -> list[str]:
        return _labelled(units)
