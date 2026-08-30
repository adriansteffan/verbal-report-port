import hashlib
import re

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from utils import data, llm
from utils.paths import OUTPUT

PROGRESS = False  # main.py turns this on


def cohort(limit: int | None = None) -> tuple[list[str], pd.Series]:
    """Analyzable participants that also carry a behavioral label."""
    labels = data.vr()["aware"].dropna().astype(int)
    return [p for p in data.analyzable_participants() if p in labels.index][
        :limit
    ], labels


def slug(config: str) -> str:
    """Readable one-line form of a config."""
    config = re.sub(r"\b\w+=(?=\w+\()", "", config)
    return re.sub(r"[^0-9A-Za-z]+", "-", config).strip("-")[:200]


def _filename(config: str) -> str:
    return f"{slug(config)}-{hashlib.sha256(config.encode()).hexdigest()[:8]}.csv"


def features(extractor, limit: int | None = None) -> pd.DataFrame:
    """Write one participants x features CSV per config, for reading without
    rerunning anything."""
    pids, labels = cohort(limit)
    if PROGRESS:
        print(" ".join(extractor.config().split()))
    df = extractor.frame(pids, progress=PROGRESS)
    df.insert(0, "aware", labels[pids].to_numpy())
    df.index.name = "participant"

    config = extractor.config()
    path = OUTPUT / "features" / _filename(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"{len(df)} participants x {df.shape[1] - 1} features -> {path.name}")
    return df


def export_calls(extractor, limit: int | None = None) -> pd.DataFrame:
    """Every request this config sends and the reply it gets, for reading by
    hand. Every call is a cache hit once features() has been through, so free"""
    pids, labels = cohort(limit)
    rows = []
    for pid in pids:
        with llm.capture() as sent:
            scores = extractor.unit_scores(pid)
        # a unit the participant was silent through is never sent and scores
        # NaN, so the rows that survive name the units the calls were about
        spoken = scores.dropna(how="all").index.tolist()
        for n, call in enumerate(sent):
            messages = call["messages"]
            rows.append(
                {
                    "participant": pid,
                    "aware": labels[pid],
                    # each seed makes one pass over the units, in order
                    "unit": spoken[n % len(spoken)],
                    "seed": call["seed"],
                    "system": next(
                        (m["content"] for m in messages if m["role"] == "system"), ""
                    ),
                    "conversation": "\n\n".join(
                        f"[{m['role']}]\n{m['content']}"
                        for m in messages
                        if m["role"] != "system"
                    ),
                    "response": call["response"],
                }
            )
    df = pd.DataFrame(rows).sort_values(["participant", "unit", "seed"])
    path = OUTPUT / "review" / _filename(extractor.config())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{len(df)} calls -> review/{path.name}")
    return df


def _name(model) -> str:
    """Extractors spell out their whole config: in a sweep every line otherwise
    reads TaxonomyExtractor and there is no telling the runs apart."""
    steps = getattr(model, "steps", None)
    if steps is not None:
        return " + ".join(_name(step) for _, step in steps)
    config = getattr(model, "config", None)
    return " ".join(config().split()) if callable(config) else type(model).__name__  # type: ignore


def _loo_scores(model, X, y) -> np.ndarray:
    """One out-of-sample score per participant, from a full leave-one-out pass."""
    # SVC and friends have no predict_proba unless asked for it
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    scores = cross_val_predict(model, X, y, cv=LeaveOneOut(), method=method)
    # predict_proba columns follow np.unique(y), so find class 1, decision_function is already 1-D
    return scores[:, list(np.unique(y)).index(1)] if scores.ndim > 1 else scores


def unit_features(extractor, limit: int | None = None) -> pd.DataFrame:
    """Per-unit scores for a chunked segmenter, in the same shape as features()
    but keyed by (participant, unit)

    Written to a subfolder, not used in our models, just for external analyses"""
    pids, labels = cohort(limit)
    rows = [
        {
            "participant": pid,
            "unit": unit,
            "aware": labels[pid],
            **scores,
            "text": " ".join(text.split()),
        }
        for pid in pids
        for (unit, scores), text in zip(
            extractor.unit_scores(pid).iterrows(),
            extractor.segments.split(extractor.scope.select(pid)),
        )
    ]
    df = pd.DataFrame(rows)

    with config_context(print_changed_only=False):  # spell out every parameter
        config = " ".join(
            repr(o) for o in (extractor.coder, extractor.scope, extractor.segments)
        )
    config = re.sub(r"pooling='\w+', ?", "", config)

    path = OUTPUT / "features" / "per-unit" / _filename(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"{len(df)} units x {len(df.columns) - 4} categories -> per-unit/{path.name}")
    return df


def _write_permutations(model: str, auc: float, null: np.ndarray) -> None:
    """The null the p-value is read against, one row per run. Permutation 0 is
    the real labels, the rest are shuffled."""
    df = pd.DataFrame(
        {
            "permutation": range(len(null) + 1),
            "auc": np.concatenate([[auc], null]),
        }
    )
    path = OUTPUT / "permutations" / _filename(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def evaluate(model, limit: int | None = None, n_permutations: int = 0, seed: int = 0):
    """Leave-one-out AUC, with an optional permutation p-value."""
    pids, labels = cohort(limit)
    X = np.array(pids).reshape(-1, 1)
    y = labels[pids].to_numpy()

    auc = float(roc_auc_score(y, _loo_scores(model, X, y)))
    result = {
        "model": _name(model),
        "n": len(y),
        "pos_rate": float(y.mean()),
        "auc": auc,
    }

    if n_permutations:
        rng = np.random.default_rng(seed)
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y)
            null[i] = roc_auc_score(y_perm, _loo_scores(model, X, y_perm))
        # + 1 in formula to prevent zero p-value
        result["p"] = float((np.sum(null >= auc) + 1) / (n_permutations + 1))
        _write_permutations(result["model"], auc, null)

    print(
        f"{result['model']}: n={result['n']} pos_rate={result['pos_rate']:.2f} "
        f"auc={result['auc']:.3f}"
        + (
            f" p={result['p']:.3f} ({n_permutations} permutations)"
            if n_permutations
            else ""
        )
    )
    return result
