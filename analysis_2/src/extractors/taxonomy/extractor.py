import numpy as np

from extractors.base import FeatureExtractor
from extractors.taxonomy import codebook
from extractors.taxonomy.coders import Coder
from extractors.taxonomy.levels import Level
from extractors.taxonomy.scopes import Scope
from extractors.taxonomy.segments import Segmenter


class TaxonomyExtractor(FeatureExtractor):
    """What verbal behavior did this participant engage in, as one number per
    category or cluster.
    """

    def __init__(
        self,
        scope: Scope,
        segments: Segmenter,
        coder: Coder,
        level: Level,
    ):
        self.scope = scope
        self.segments = segments
        self.coder = coder
        self.level = level

    def _extract(self, participant: str) -> dict[str, float]:
        units = self.segments.split(self.scope.select(participant))
        if units:
            scores = self.coder.score(
                units,
                self.segments.prompt_granularity,
                self.config(),  # drill down in order to reconstruct from cache later during inspection
                participant,
            )
        else:  # participant said nothing in this scope
            scores = np.zeros((1, len(codebook.categories())))
        return self.level.report(self.segments.pool(scores))
