"""How the LLM is asked which categories apply to a unit.

Coders return a (units x categories) matrix of scores in
[0, 1]. With one seed those are 0/1; with several, the fraction of seeds that
said yes."""

import json
import random
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from extractors.taxonomy import codebook
from utils import llm

# How the categories and their examples are shuffled before the model reads
# them. clustered shuffles within cluster and the cluster order, but leaves options within a cluster next to each other
# full shuffles the categories randomly
# examples for a given category are always shuffled
SHUFFLES = ("none", "clustered", "full")

BINARY_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "applies",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "One or two sentences weighing the criteria "
                    "against what was actually said, quoting the decisive phrase.",
                },
                "applies": {"type": "boolean"},
            },
            "required": ["reasoning", "applies"],
            "additionalProperties": False,
        },
    },
}


class Coder(BaseEstimator, ABC):
    def __init__(
        self,
        n_seeds: int = 5,
        memory: bool = False,
        shuffle: str = "none",
        model: str = llm.DEFAULT_MODEL,
    ):
        self.n_seeds = n_seeds
        self.memory = memory
        self.shuffle = shuffle
        self.model = model

    @abstractmethod
    def score(self, units: list[str], prompt_granularity: str) -> np.ndarray: ...

    def _rng(self, seed: int) -> random.Random | None:
        """The randomness `_options` and `codebook.examples` draw their order
        from, or None to leave the codebook in file order."""
        if self.shuffle not in SHUFFLES:
            raise ValueError(f"shuffle must be one of {SHUFFLES}, got {self.shuffle!r}")
        return random.Random(seed) if self.shuffle != "none" else None

    def _options(self, rng: random.Random | None) -> list[str]:
        """The categories in the order they are offered to the model.

        "clustered" shuffles within each cluster and shuffles the clusters
        themselves, so related categories stay adjacent: the grouping tells the
        model that e.g. letter and position hypotheses are the same behavior
        applied to different features."""
        options = codebook.categories(include_escape=True)
        if rng is None:
            return options
        if self.shuffle == "full":
            return rng.sample(options, len(options))
        # shuffle categories within clusters
        clusters = [rng.sample(cs, len(cs)) for cs in codebook.clusters().values()]
        # shuffle clusters
        return [
            c for cluster in rng.sample(clusters, len(clusters)) for c in cluster
        ] + [codebook.ESCAPE]

    def _walk(
        self, units, prompt_granularity, response_format, seed, system
    ) -> list[dict]:
        """Should be called by score(). One pass over the units, the codebook in
        the system message and one unit per turn.
        With memory on, each answer stays in context for the next unit."""
        history: list[dict] = [{"role": "system", "content": system}]
        replies = []
        for unit in units:
            content = f'{codebook.UNIT_LABEL[prompt_granularity]}:\n"""{unit}"""'
            messages = history + [{"role": "user", "content": content}]
            reply = llm.judge(messages, response_format, seed, self.model)
            replies.append(reply)
            if self.memory:
                history = messages + [
                    {"role": "assistant", "content": json.dumps(reply)}
                ]
        return replies


def _binary_system(category: str, prompt_granularity: str, examples: str) -> str:
    shots = f"\nExamples of this category:\n{examples}\n" if examples else ""
    label = codebook.UNIT_LABEL[prompt_granularity].lower()
    return (
        f"{codebook.prompt(category, prompt_granularity)}\n{shots}\n"
        f"You will be sent one {label} at a time, in the order they were spoken. "
        f"For each, answer whether the category applies."
    )


class Binary(Coder):
    """One yes/no question per category. Thorough but costs len(categories)
    calls per unit.

    With memory=True the conversation runs along the units, not along the
    categories: one thread per category walks every unit in order, so the model
    sees how it ruled on *this* category for earlier units and never sees its
    answers for any other category. That keeps a thread to len(units) turns
    instead of len(units) * len(categories), but loses cross-category
    consistency. (might want to change later, but this is too expensive anyways)"""

    def score(self, units: list[str], prompt_granularity: str) -> np.ndarray:
        categories = codebook.categories()
        out = np.zeros((len(units), len(categories)))
        for seed in range(self.n_seeds):
            rng = self._rng(seed)
            for j, category in enumerate(categories):
                examples = codebook.examples(category, rng)
                system = _binary_system(category, prompt_granularity, examples)
                replies = self._walk(
                    units, prompt_granularity, BINARY_FORMAT, seed, system
                )
                out[:, j] += [bool(r["applies"]) for r in replies]
        return out / self.n_seeds


def _catalogue(
    prompt_granularity: str, options: list[str], rng: random.Random | None
) -> str:
    """The category list spelled out, in the order given. The rng reorders the
    examples inside a category; the category order is already decided."""
    return "\n\n".join(
        f"{c}: {codebook.prompt(c, prompt_granularity)}"
        + (f"\nExamples: {examples}" if (examples := codebook.examples(c, rng)) else "")
        for c in options
    )


def _topk_system(k: int, prompt_granularity: str, catalogue: str) -> str:
    label = codebook.UNIT_LABEL[prompt_granularity].lower()
    return (
        f"Below is a catalogue of verbal behavior categories.\n\n"
        f"{catalogue}\n\n"
        f"You will be sent one {label} at a time from a think-aloud experiment, "
        f"in the order it was spoken. For each, name the {k} categories that "
        f"apply most strongly."
    )


class TopK(Coder):
    """One call per unit: name the k categories that fit best."""

    def __init__(
        self,
        k: int = 3,
        n_seeds: int = 5,
        memory: bool = False,
        shuffle: str = "none",
        model: str = llm.DEFAULT_MODEL,
    ):
        super().__init__(n_seeds=n_seeds, memory=memory, shuffle=shuffle, model=model)
        self.k = k

    def _format(self, options: list[str]) -> dict:
        # the escape category is offered here, then dropped from the features
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "top_categories",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "One or two sentences on which "
                            "categories were considered and why these fit best.",
                        },
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": options,
                            },
                            "minItems": self.k,
                            "maxItems": self.k,
                            # CACHE-BOUND: "uniqueItems": True belongs here -
                            # without it the model can name one category twice -
                            # but adding it rewrites every top-k request
                        },
                    },
                    "required": ["reasoning", "categories"],
                    "additionalProperties": False,
                },
            },
        }

    def score(self, units: list[str], prompt_granularity: str) -> np.ndarray:
        categories = codebook.categories()
        index = {c: j for j, c in enumerate(categories)}
        out = np.zeros((len(units), len(categories)))
        for seed in range(self.n_seeds):
            rng = self._rng(seed)
            options = self._options(rng)
            catalogue = _catalogue(prompt_granularity, options, rng)
            system = _topk_system(self.k, prompt_granularity, catalogue)
            replies = self._walk(
                units, prompt_granularity, self._format(options), seed, system
            )
            for i, reply in enumerate(replies):
                # set(): a category named twice is one occurrence, not two.
                # CACHE-BOUND: goes away with the uniqueItems note in _format
                for picked in set(reply["categories"]):
                    if picked in index:  # the escape category falls through here
                        out[i, index[picked]] += 1
        return out / self.n_seeds
