import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from utils import data
from utils.paths import OUTPUT


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
    df = extractor.frame(pids)
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


def evaluate(model, limit: int | None = None) -> float:
    """Leave-one-out AUC. `model` is a sklearn pipeline taking participant ids."""
    pids, labels = cohort(limit)
    X = np.array(pids).reshape(-1, 1)
    y = labels[pids].to_numpy()

    # SVC and friends have no predict_proba unless asked for it
    method = "predict_proba" if hasattr(model, "predict_proba") else "decision_function"
    scores = cross_val_predict(model, X, y, cv=LeaveOneOut(), method=method)
    # predict_proba columns follow np.unique(y), so find class 1, decision_function is already 1-D
    scores = scores[:, list(np.unique(y)).index(1)] if scores.ndim > 1 else scores

    auc = roc_auc_score(y, scores)
    steps = getattr(model, "steps", [(None, model)])
    name = " + ".join(type(step).__name__ for _, step in steps)
    print(f"{name}: n={len(y)} pos_rate={y.mean():.2f} auc={auc:.3f}")
    return float(auc)
