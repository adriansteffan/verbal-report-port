"""Which of a participant's utterances are in scope."""

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator

from utils import data

LAST_ACQUISITION_ROUND = 24


class Scope(BaseEstimator, ABC):
    @abstractmethod
    def select(self, participant: str) -> list[str]: ...


class Acquisition(Scope):
    def select(self, participant: str) -> list[str]:
        df = data.utterances(participant)
        return df.loc[df["phase"] == "acquisition", "text"].tolist()


class UntilDiscovery(Scope):
    """Speech up to the round the participant's behavior shows they got the
    rule. Unaware folks get a fixed scope instead. Might be leaky if not careful, discuss again"""

    def __init__(self, unaware_extra_rounds: int = 12):
        self.unaware_extra_rounds = unaware_extra_rounds

    def select(self, participant: str) -> list[str]:
        df = data.utterances(participant)
        discovered = data.vr()["time"].dropna().get(participant, 0)
        cutoff = (
            discovered
            if discovered > 0
            else LAST_ACQUISITION_ROUND + self.unaware_extra_rounds
        )
        return df.loc[df["round"] <= cutoff, "text"].tolist()
