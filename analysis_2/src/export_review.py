"""uv run src/export_review.py

Writes output/review/judgements.csv

heavily generated/throwaway code to generate the judgements.csv for manual inspection
"""

import json
import re

import pandas as pd

from extractors.taxonomy import codebook, coders
from extractors.taxonomy.scopes import Acquisition, UntilDiscovery
from extractors.taxonomy.segments import Groups, Transcript, Utterances
from pipeline import cohort, slug
from utils import llm
from utils.paths import OUTPUT

# Everything that can change a prompt
SCOPES = [Acquisition(), UntilDiscovery()]
SEGMENTERS = [Transcript(), Groups(size=6), Utterances()]
KS = [1, 3]

pids, labels = cohort()


def _passages() -> dict[tuple[str, str], tuple[str, int, str]]:
    """(participant, prompt) -> (granularity, passage number, text).

    The cache does not record which passage a call was for, so we run the prompt
    builders over every scope, segmenter and k the project uses and the
    results matched back. Next time I will use a simple csv as a cache again ugh why was I trying to be fancy
    """
    out = {}
    for scope in SCOPES:
        for segments in SEGMENTERS:
            granularity = segments.prompt_granularity
            for pid in pids:
                for i, unit in enumerate(segments.split(scope.select(pid)), start=1):
                    forms = [f'{codebook.UNIT_LABEL[granularity]}:\n"""{unit}"""']
                    forms += [coders._topk_prompt(unit, k, granularity) for k in KS]  # type: ignore
                    for form in forms:
                        out[(pid, form)] = (granularity, i, unit)
    return out


def _param(configs: set[str], name: str) -> str:
    values = set()
    for config in configs:
        if found := re.search(rf"\b{name}=(?:'([^']*)'|(\w+))", config):
            values.add(next(g for g in found.groups() if g is not None))
    return "; ".join(sorted(values))


asked: dict[str, tuple[str, set[str]]] = {}
for key, config, pid in llm.db.execute("SELECT key, config, participant FROM calls"):
    asked.setdefault(key, (pid, set()))[1].add(config)

sent = _passages()
rows, skipped = [], 0
for key, seed, messages, response in llm.db.execute(
    "SELECT key, seed, messages, response FROM cache"
):
    pid, configs = asked.get(key, ("", set()))
    found = sent.get((pid, json.loads(messages)[-1]["content"]))
    if found is None:
        skipped += 1
        continue
    granularity, index, unit = found
    reply = json.loads(response)
    rows.append(
        {
            "participant": pid,
            "aware": labels[pid],
            "scope": _param(configs, "scope"),
            "granularity": granularity,
            "k": _param(configs, "k"),
            "memory": _param(configs, "memory"),
            "model": _param(configs, "model"),
            "seed": seed,
            "passage": index,
            "n_configs": len(configs),
            "configs": "; ".join(
                sorted(slug(c).replace("TaxonomyExtractor-", "") for c in configs)
            ),
            "text": " ".join(unit.split()),
            "categories": "; ".join(reply["categories"]),
            "reasoning": " ".join(reply.get("reasoning", "").split()),
        }
    )

out = OUTPUT / "review"
out.mkdir(parents=True, exist_ok=True)
judgements = pd.DataFrame(rows).sort_values(
    ["scope", "granularity", "k", "memory", "participant", "passage", "seed"]
)
judgements.to_csv(out / "judgements.csv", index=False)
