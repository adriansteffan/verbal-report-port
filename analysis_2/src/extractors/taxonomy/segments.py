"""How the in-scope speech is cut into the units the LLM judges, and how the
per-unit scores are folded back into one number per category."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

POOL = {"max": np.max, "mean": np.mean}


class Segmenter(BaseEstimator, ABC):
    prompt_granularity: str

    @abstractmethod
    def split(self, texts: list[str]) -> list[str]: ...

    @abstractmethod
    def pool(self, scores: np.ndarray) -> np.ndarray:
        """(units x categories) -> one score per category."""


class Transcript(Segmenter):
    """Everything at once, one unit."""

    prompt_granularity = "transcript"

    def split(self, texts: list[str]) -> list[str]:
        return ["\n\n".join(texts)] if texts else []

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
    """Consecutive utterances, chunked."""

    prompt_granularity = "group"

    def __init__(self, size: int = 6, pooling: str = "max"):
        super().__init__(pooling=pooling)
        self.size = size

    def split(self, texts: list[str]) -> list[str]:
        return [
            "\n\n".join(texts[i : i + self.size])
            for i in range(0, len(texts), self.size)
        ]


class Utterances(_Chunked):
    """One unit per recording."""

    prompt_granularity = "utterance"

    def split(self, texts: list[str]) -> list[str]:
        return texts
