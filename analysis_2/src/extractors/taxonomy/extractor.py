import numpy as np
import pandas as pd

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

    def unit_scores(self, participant: str) -> pd.DataFrame:
        """One row per unit, one column per category, values in [0, 1] = the
        fraction of seeds that picked it."""
        units = self.segments.split(self.scope.select(participant))
        if not units:
            # dtype, or _extract's isnan gets an object array
            return pd.DataFrame(columns=codebook.categories(), dtype=float)
        scores = np.full((len(units), len(codebook.categories())), np.nan)
        spoken = [i for i, unit in enumerate(units) if unit]
        if spoken:
            scores[spoken] = self.coder.score(
                [units[i] for i in spoken], self.segments.prompt_granularity
            )
        return pd.DataFrame(
            scores,
            columns=codebook.categories(),
            index=pd.RangeIndex(1, len(units) + 1, name="unit"),
        )

    def _extract(self, participant: str) -> dict[str, float]:
        scores = self.unit_scores(participant).to_numpy()
        scores = scores[~np.isnan(scores).all(axis=1)]
        if not len(scores):
            # silent participants should have been filterd out earlier
            raise ValueError(f"{participant} said nothing in {self.scope!r}")
        return self.level.report(self.segments.pool(scores))
