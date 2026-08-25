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

UNIT_MARKERS = tuple(f'{label}:\n"""' for label in codebook.UNIT_LABEL.values())


def _tail(prompt: str) -> str:
    """
    Ugly hack to work shuffling into the current caching structure, ugh.
    Everything from Passage:/Utterance:/Transcript: onwards,
    dropping the catalogue that comes before it.

    Calls are looked up by their prompt text, and the catalogue is the part of
    that text a shuffling coder reorders on every seed. Cutting it off leaves a
    key that is the same whatever order the categories were listed in."""
    starts = [prompt.index(m) for m in UNIT_MARKERS if m in prompt]
    return prompt[min(starts) :] if starts else prompt


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
                    # built with an empty catalogue, since _tail cuts it off anyway
                    forms += [
                        coders._topk_prompt(unit, k, granularity, "")  # type: ignore
                        for k in KS
                    ]
                    for form in forms:
                        out[(pid, _tail(form))] = (granularity, i, unit)
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
    found = sent.get((pid, _tail(json.loads(messages)[-1]["content"])))
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
            "shuffle": _param(configs, "shuffle"),
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
    ["scope", "granularity", "k", "memory", "shuffle", "participant", "passage", "seed"]
)
judgements.to_csv(out / "judgements.csv", index=False)
