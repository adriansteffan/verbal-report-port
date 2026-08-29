"""Which of a participant's utterances are in scope."""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator

from utils import data

LAST_ACQUISITION_ROUND = 24


class Scope(BaseEstimator, ABC):
    @abstractmethod
    def select(self, participant: str) -> pd.DataFrame:
        """The in-scope recordings, with the round each was made in."""


class Acquisition(Scope):
    def select(self, participant: str) -> pd.DataFrame:
        df = data.utterances(participant)
        return df[df["round"] <= LAST_ACQUISITION_ROUND]


class UntilDiscovery(Scope):
    """Speech up to the round the participant's behavior shows they got the
    rule. Unaware folks get a fixed scope instead. Might be leaky if not careful, discuss again"""

    def __init__(self, unaware_extra_rounds: int = 12):
        self.unaware_extra_rounds = unaware_extra_rounds

    def select(self, participant: str) -> pd.DataFrame:
        df = data.utterances(participant)
        discovered = float(data.vr()["time"].dropna().get(participant, 0))
        cutoff = (
            discovered
            if discovered > 0
            else LAST_ACQUISITION_ROUND + self.unaware_extra_rounds
        )
        return df[df["round"] <= cutoff]
