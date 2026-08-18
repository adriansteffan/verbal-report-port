"""At what level the feature vector reports: one number per category, or one
per cluster."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from extractors.taxonomy import codebook

POOL = {"max": np.max, "mean": np.mean}


class Level(BaseEstimator, ABC):
    @abstractmethod
    def report(self, per_category: np.ndarray) -> dict[str, float]:
        """One score per category -> the named feature vector."""


class Categories(Level):
    """One number per category."""

    def report(self, per_category: np.ndarray) -> dict[str, float]:
        return dict(zip(codebook.categories(), per_category.tolist()))


class Clusters(Level):
    """One number per cluster, pooled from its categories"""

    def __init__(self, across: str = "max"):
        self.across = across

    def report(self, per_category: np.ndarray) -> dict[str, float]:
        scores = dict(zip(codebook.categories(), per_category.tolist()))
        pool = POOL[self.across]
        return {
            cluster: float(pool([scores[c] for c in members]))
            for cluster, members in codebook.clusters().items()
        }
