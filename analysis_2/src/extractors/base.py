from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# In memory feature vector cache keyed on (extractor config, participant)
_CACHE: dict = {}


class FeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    """Fit our extractors into the sklearn pipeline shape;
    maps participant ids to a feature matrix. X is a column of ids,
    so extraction happens inside the sklearn pipeline but only once per
    participant."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return self.frame(np.ravel(X)).to_numpy(dtype=float)

    def frame(self, participants) -> pd.DataFrame:
        """Named feature matrix, participants x features. transform() drops the
        names for sklearn but this keeps them, for export and for reading."""
        participants = list(participants)
        # via DataFrame so columns align by feature name, not by dict order
        return pd.DataFrame([self._cached(p) for p in participants], index=participants)

    def config(self) -> str:
        """The full nested config, as sklearn renders it. Called through the
        class because type checkers resolve `self.__repr__` against object's,
        which takes no arguments; N_CHAR_MAX defeats the 700-char truncation
        that would let two long configs collide."""
        return BaseEstimator.__repr__(self, N_CHAR_MAX=10**9)

    def _cached(self, participant: str) -> dict[str, float]:
        key = (self.config(), participant)
        if key not in _CACHE:
            _CACHE[key] = self._extract(participant)
        return _CACHE[key]

    @abstractmethod
    def _extract(self, participant: str) -> dict[str, float]: ...
