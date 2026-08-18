import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from utils import data
from utils.paths import OUTPUT

PROGRESS = False  # main.py turns this on


def cohort(limit: int | None = None) -> tuple[list[str], pd.Series]:
    """Analyzable participants that also carry a behavioral label."""
    labels = data.vr()["aware"].dropna().astype(int)
    return [p for p in data.analyzable_participants() if p in labels.index][
        :limit
    ], labels


def features(extractor, limit: int | None = None) -> pd.DataFrame:
    """Write one participants x features CSV per config, for reading without
    rerunning anything. The config is in the filename, hashed so two long
    configs cannot land on the same file."""
    pids, labels = cohort(limit)
    if PROGRESS:
        print(" ".join(extractor.config().split()))
    df = extractor.frame(pids, progress=PROGRESS)
    df.insert(0, "aware", labels[pids].to_numpy())
    df.index.name = "participant"

    config = extractor.config()
    slug = re.sub(r"[^0-9A-Za-z]+", "-", config).strip("-")[:80]
    digest = hashlib.sha256(config.encode()).hexdigest()[:8]
    path = OUTPUT / "features" / f"{slug}-{digest}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"{len(df)} participants x {df.shape[1] - 1} features -> {path.name}")
    return df


def _name(model) -> str:
    steps = getattr(model, "steps", None)
    return (
        type(model).__name__
        if steps is None
        else " + ".join(_name(step) for _, step in steps)
    )


def _loo_scores(model, X, y) -> np.ndarray:
    """One out-of-sample score per participant, from a full leave-one-out pass."""
    # SVC and friends have no predict_proba unless asked for it
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    scores = cross_val_predict(model, X, y, cv=LeaveOneOut(), method=method)
    # predict_proba columns follow np.unique(y), so find class 1, decision_function is already 1-D
    return scores[:, list(np.unique(y)).index(1)] if scores.ndim > 1 else scores


def evaluate(model, limit: int | None = None, n_permutations: int = 0, seed: int = 0):
    """Leave-one-out AUC, with an optional permutation p-value."""
    pids, labels = cohort(limit)
    X = np.array(pids).reshape(-1, 1)
    y = labels[pids].to_numpy()

    auc = roc_auc_score(y, _loo_scores(model, X, y))
    result = {
        "model": _name(model),
        "n": len(y),
        "pos_rate": float(y.mean()),
        "auc": float(auc),
    }

    if n_permutations:
        rng = np.random.default_rng(seed)
        null = np.empty(n_permutations)
        for i in range(n_permutations):
            y_perm = rng.permutation(y)
            null[i] = roc_auc_score(y_perm, _loo_scores(model, X, y_perm))
        # + 1 in formula to prevent zero p-value
        result["p"] = float((np.sum(null >= auc) + 1) / (n_permutations + 1))

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
