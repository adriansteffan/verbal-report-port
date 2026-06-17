from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import participants

HERE = Path(__file__).resolve().parent
vr = pd.read_csv(HERE / "df_vr.csv")
labels = pd.read_csv(HERE / "output" / "letter_rule.csv")

SCORE = {"none": 0.0, "partial": 0.5, "explicit": 1.0}
LABEL = {0.0: "none", 0.5: "partial", 1.0: "explicit"}


def bootstrap_ci(stat, n_units, n=2000, level=0.95, seed=0):
    """Percentile CI from resampling unit indices; stat(idx) -> value."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        try:
            v = stat(rng.integers(0, n_units, n_units))
        except ValueError:
            continue  # e.g. a single-class resample makes AUC undefined
        if v is not None and not np.isnan(v):
            vals.append(v)
    return tuple(np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


if __name__ == "__main__":
    keep = participants.analyzable()
    scored = labels.assign(score=labels["rule_evidence"].map(SCORE))
    scored = scored[scored["participant"].isin(keep)]

    agg = scored.groupby("participant")["score"].agg(["mean", "median"])
    df = agg.join(vr.set_index("participant")["aware"]).dropna(subset=["aware"])
    print(
        f"{labels['participant'].nunique()} labeled, {len(agg)} analyzable, {len(df)} with aware"
    )

    # self-consistency across the 5 seeds
    seeds = scored.pivot(index="seed", columns="participant", values="score").to_numpy()
    alpha = krippendorff.alpha(reliability_data=seeds, level_of_measurement="ordinal")
    lo, hi = bootstrap_ci(
        lambda i: krippendorff.alpha(
            reliability_data=seeds[:, i], level_of_measurement="ordinal"
        ),
        seeds.shape[1],
    )
    print(f"alpha {alpha:.3f} [{lo:.3f}, {hi:.3f}]")

    # agreement with behavior
    table = pd.crosstab(df["aware"], df["median"].map(LABEL))
    print(table.reindex(columns=["none", "partial", "explicit"], fill_value=0))
    aware, score = df["aware"].to_numpy(), df["mean"].to_numpy()
    auc = roc_auc_score(aware, score)
    lo, hi = bootstrap_ci(lambda i: roc_auc_score(aware[i], score[i]), len(df))
    print(f"auc {auc:.3f} [{lo:.3f}, {hi:.3f}]")
