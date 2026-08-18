import random

from extractors.base import FeatureExtractor


class RandomFeatureExtractor(FeatureExtractor):
    def __init__(self, n_features: int = 5):
        self.n_features = n_features

    def _extract(self, participant: str) -> dict[str, float]:
        rng = random.Random(participant)
        return {f"random_{i}": rng.random() for i in range(self.n_features)}
