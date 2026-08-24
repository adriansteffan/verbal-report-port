import numpy as np
import pandas as pd

from extractors.base import FeatureExtractor
from extractors.taxonomy import codebook
from extractors.taxonomy.coders import Coder
from extractors.taxonomy.levels import Level
from extractors.taxonomy.scopes import Scope
from extractors.taxonomy.segments import Segmenter
from utils import llm


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

    def unit_scores(self, participant: str) -> pd.DataFrame:
        """One row per unit, one column per category, values in [0, 1] = the
        fraction of seeds that picked it."""
        units = self.segments.split(self.scope.select(participant))
        if not units:
            return pd.DataFrame(columns=codebook.categories())
        # provenance is set here because this is the one place that knows both
        # the config and the participant; llm.judge reads it
        with llm.provenance(self.config(), participant):
            scores = self.coder.score(units, self.segments.prompt_granularity)
        return pd.DataFrame(
            scores,
            columns=codebook.categories(),
            index=pd.RangeIndex(1, len(units) + 1, name="unit"),
        )

    def _extract(self, participant: str) -> dict[str, float]:
        scores = self.unit_scores(participant).to_numpy()
        if not len(scores):  # said nothing in this scope, so nothing to pool
            scores = np.zeros((1, len(codebook.categories())))
        return self.level.report(self.segments.pool(scores))
