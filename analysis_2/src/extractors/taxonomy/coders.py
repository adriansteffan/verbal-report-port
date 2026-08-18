"""How the LLM is asked which categories apply to a unit.

Coders return a (units x categories) matrix of scores in
[0, 1]. With one seed those are 0/1; with several, the fraction of seeds that
said yes."""

import functools
import json
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from extractors.taxonomy import codebook
from utils import llm

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
        self, n_seeds: int = 5, memory: bool = False, model: str = llm.DEFAULT_MODEL
    ):
        self.n_seeds = n_seeds
        self.memory = memory
        self.model = model

    @abstractmethod
    def score(
        self,
        units: list[str],
        prompt_granularity: str,
        config: str = "",
        participant: str = "",
    ) -> np.ndarray: ...

    def _walk(
        self,
        units: list[str],
        prompt,
        response_format: dict,
        seed: int,
        config: str = "",
        participant: str = "",
    ) -> list[dict]:
        """Should be called by score(). One pass over the units.
        With memory on, each answer stays in context for the next unit"""
        history: list[dict] = []
        replies = []
        for unit in units:
            messages = history + [{"role": "user", "content": prompt(unit)}]
            reply = llm.judge(
                messages, response_format, seed, self.model, config, participant
            )
            replies.append(reply)
            if self.memory:
                history = messages + [
                    {"role": "assistant", "content": json.dumps(reply)}
                ]
        return replies


def _binary_prompt(unit: str, category: str, prompt_granularity: str) -> str:
    examples = codebook.examples(category)
    shots = f"\nExamples of this category:\n{examples}\n" if examples else ""
    return (
        f"{codebook.prompt(category, prompt_granularity)}\n{shots}\n"
        f'{codebook.UNIT_LABEL[prompt_granularity]}:\n"""{unit}"""\n\n'
        f"Does the category apply?"
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

    def score(
        self,
        units: list[str],
        prompt_granularity: str,
        config: str = "",
        participant: str = "",
    ) -> np.ndarray:
        categories = codebook.categories()
        out = np.zeros((len(units), len(categories)))
        for seed in range(self.n_seeds):
            for j, category in enumerate(categories):
                replies = self._walk(
                    units,
                    lambda u, c=category: _binary_prompt(u, c, prompt_granularity),
                    BINARY_FORMAT,
                    seed,
                    config,
                    participant,
                )
                out[:, j] += [bool(r["applies"]) for r in replies]
        return out / self.n_seeds


@functools.cache
def _catalogue(prompt_granularity: str) -> str:
    """The full category list"""
    return "\n\n".join(
        f"{c}: {codebook.prompt(c, prompt_granularity)}"
        + (f"\nExamples: {examples}" if (examples := codebook.examples(c)) else "")
        for c in codebook.categories(include_escape=True)
    )


def _topk_prompt(unit: str, k: int, prompt_granularity: str) -> str:
    catalogue = _catalogue(prompt_granularity)
    return (
        f"Below is a catalogue of verbal behavior categories, then one "
        f"{codebook.UNIT_LABEL[prompt_granularity].lower()} from a think-aloud experiment.\n\n"
        f"{catalogue}\n\n"
        f'{codebook.UNIT_LABEL[prompt_granularity]}:\n"""{unit}"""\n\n'
        f"Name the {k} categories that apply most strongly."
    )


class TopK(Coder):
    """One call per unit: name the k categories that fit best."""

    def __init__(
        self,
        k: int = 3,
        n_seeds: int = 5,
        memory: bool = False,
        model: str = llm.DEFAULT_MODEL,
    ):
        super().__init__(n_seeds=n_seeds, memory=memory, model=model)
        self.k = k

    def _format(self) -> dict:
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
                                "enum": codebook.categories(include_escape=True),
                            },
                            "minItems": self.k,
                            "maxItems": self.k,
                        },
                    },
                    "required": ["reasoning", "categories"],
                    "additionalProperties": False,
                },
            },
        }

    def score(
        self,
        units: list[str],
        prompt_granularity: str,
        config: str = "",
        participant: str = "",
    ) -> np.ndarray:
        categories = codebook.categories()
        index = {c: j for j, c in enumerate(categories)}
        out = np.zeros((len(units), len(categories)))
        for seed in range(self.n_seeds):
            replies = self._walk(
                units,
                lambda u: _topk_prompt(u, self.k, prompt_granularity),
                self._format(),
                seed,
                config,
                participant,
            )
            for i, reply in enumerate(replies):
                for picked in reply["categories"]:
                    if picked in index:  # the escape category falls through here
                        out[i, index[picked]] += 1
        return out / self.n_seeds
