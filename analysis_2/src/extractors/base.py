from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

# In memory feature vector cache keyed on (extractor config, participant)
_CACHE: dict = {}

# How many participants to extract at once. default value, main.py overrides it
WORKERS = 1


class FeatureExtractor(BaseEstimator, TransformerMixin, ABC):
    """Fit our extractors into the sklearn pipeline shape;
    maps participant ids to a feature matrix. X is a column of ids,
    so extraction happens inside the sklearn pipeline but only once per
    participant."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return self.frame(np.ravel(X)).to_numpy(dtype=float)

    def config(self) -> str:
        with config_context(print_changed_only=False):
            return " ".join(BaseEstimator.__repr__(self, N_CHAR_MAX=10**9).split())

    def frame(self, participants, progress: bool = False) -> pd.DataFrame:
        """Named feature matrix, participants x features. transform() drops the
        names for sklearn but this keeps them, for export and for reading."""
        participants = list(participants)
        config = self.config()

        def extract(participant):
            key = (config, participant)
            if key not in _CACHE:
                _CACHE[key] = self._extract(participant)
            return _CACHE[key]

        with ThreadPoolExecutor(WORKERS) as pool:
            rows = list(
                tqdm(
                    pool.map(extract, participants),
                    desc="  extracting",
                    unit="p",
                    total=len(participants),
                    disable=not progress,
                    # redraw every iteration
                    miniters=1,
                    mininterval=0,
                )
            )
        # via DataFrame so columns align by feature name, not by dict order
        return pd.DataFrame(rows, index=participants)

    @abstractmethod
    def _extract(self, participant: str) -> dict[str, float]: ...
