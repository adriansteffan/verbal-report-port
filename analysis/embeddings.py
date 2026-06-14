import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import participants

warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn 1.10/1.11 churn

HERE = Path(__file__).resolve().parent
RESSOURCES = HERE.parent / "ressources"
EMB = pd.read_parquet(HERE / "output" / "embeddings.parquet")
vr = pd.read_csv(HERE / "df_vr.csv")

SCOPES = ["transfer", "acquisition", "seqgen", "full"]
DIMS = [256, 512, 1024, 2048, 4096]  # full MRL sweep, only with --sweep
DEFAULT_DIM = 512  # AUC is flat across dims, so default to this single one
K_NEIGHBORS = 5
N_PERM = 200
RNG = np.random.default_rng(0)


def l2norm(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def scope_matrix(scope: str):
    """X (n x 4096), y (aware) for analyzable participants having this scope."""
    keep = set(participants.analyzable())
    df = EMB[(EMB["kind"] == scope) & (EMB["participant"].isin(keep))]
    df = df.merge(vr[["participant", "aware"]], on="participant").dropna(
        subset=["aware"]
    )
    return np.vstack(df["embedding"].to_numpy()), df["aware"].to_numpy().astype(int)


def prototype_loo(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out cosine-prototype scores: sim to aware mean - sim to unaware."""
    Xn = l2norm(X)
    s1, s0 = Xn[y == 1].sum(0), Xn[y == 0].sum(0)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    scores = np.empty(len(y))
    for i in range(len(y)):
        m1 = (s1 - Xn[i]) / (n1 - 1) if y[i] == 1 else s1 / n1
        m0 = (s0 - Xn[i]) / (n0 - 1) if y[i] == 0 else s0 / n0
        scores[i] = Xn[i] @ m1 / np.linalg.norm(m1) - Xn[i] @ m0 / np.linalg.norm(m0)
    return scores


def loo_scores(clf, X, y, method="predict_proba") -> np.ndarray:
    out = cross_val_predict(clf, X, y, cv=LeaveOneOut(), method=method, n_jobs=-1)
    return out if out.ndim == 1 else out[:, 1]


# standardize first: embeddings are unit-norm so each feature has tiny scale,
# which otherwise conditions the linear models badly.
def logistic_loo(X, y):  # L2 strength tuned by internal CV per fold
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=10, cv=5, max_iter=2000, scoring="neg_log_loss"),
    )
    return loo_scores(clf, X, y)


def svm_loo(X, y):  # linear SVM, margin as the score
    return loo_scores(
        make_pipeline(StandardScaler(), SVC(kernel="linear")), X, y, "decision_function"
    )


def lda_loo(X, y):  # shrinkage LDA (Ledoit-Wolf), the p >> n decoding standard
    clf = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    )
    return loo_scores(clf, X, y)


def knn_loo(X, y):  # cosine kNN; X is already L2-normalized
    return loo_scores(
        KNeighborsClassifier(n_neighbors=K_NEIGHBORS, metric="cosine"), X, y
    )


def bootstrap_ci(stat, n_units, n=2000, level=0.95):
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, n_units, n_units)
        try:
            v = stat(idx)
        except ValueError:
            continue
        if not np.isnan(v):
            vals.append(v)
    return tuple(np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]))


def permutation_p(score_fn, X, y, observed, n_perm=N_PERM):
    """Fraction of label-permuted LOO-AUCs >= the observed AUC."""
    hits = sum(
        roc_auc_score(yp := RNG.permutation(y), score_fn(X, yp)) >= observed
        for _ in range(n_perm)
    )
    return (hits + 1) / (n_perm + 1)


if __name__ == "__main__":
    import sys

    dims = DIMS if "--sweep" in sys.argv else [DEFAULT_DIM]
    rows = []
    for scope in SCOPES:
        X, y = scope_matrix(scope)
        for dim in dims:
            Xd = l2norm(X[:, :dim])
            for name, score_fn, do_perm, max_dim in [
                ("prototype", prototype_loo, True, None),
                ("logistic", logistic_loo, False, None),
                ("svm", svm_loo, False, None),
                ("lda", lda_loo, False, 512),
                ("knn", knn_loo, False, None),
            ]:
                if max_dim and dim > max_dim:
                    continue
                scores = score_fn(Xd, y)
                auc = roc_auc_score(y, scores)
                lo, hi = bootstrap_ci(
                    lambda idx: roc_auc_score(y[idx], scores[idx]), len(y)
                )
                p = permutation_p(score_fn, Xd, y, auc) if do_perm else np.nan
                rows.append(
                    {
                        "scope": scope,
                        "dim": dim,
                        "model": name,
                        "n": len(y),
                        "auc": auc,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "perm_p": p,
                    }
                )

    out = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(out.to_string(index=False))
