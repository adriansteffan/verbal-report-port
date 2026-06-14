import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import participants

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
RESSOURCES = HERE.parent / "ressources"
vr = pd.read_csv(HERE / "df_vr.csv")
SCOPES = ["transfer", "acquisition", "full"]
RNG = np.random.default_rng(0)

LETTERS = {c: i for i, c in enumerate("abcdef")}
LEXICON = {
    "rule": [
        "pattern",
        "rule",
        "order",
        "alphabet",
        "sequence",
        "match",
        "always",
        "in order",
        "next letter",
    ],
    "insight": [
        "oh",
        "wait",
        "i see",
        "got it",
        "realiz",
        "figured",
        "aha",
        "notice",
        "makes sense",
    ],
    "tentative": [
        "maybe",
        "i think",
        "guess",
        "not sure",
        "probably",
        "perhaps",
        "might",
    ],
    "certain": ["sure", "definitely", "confident", "certain", "obviously", "clearly"],
    "causal": ["because", "therefore", "since", "means", "so it"],
}


def scope_utterances(participant: str, scope: str) -> list[str]:
    df = pd.read_csv(RESSOURCES / participant / "transcriptions.csv").sort_values(
        "filename"
    )
    df = df[~df["filename"].str.contains("ruledetection")]
    if scope != "full":
        phase = df["filename"].str.extract(r"audio_\d+_(?P<phase>.+)_\d+\.wav")["phase"]
        df = df[phase.values == scope]
    return [t for t in df["text"].fillna("").str.strip() if participants.is_english(t)]


def longest_alpha_run(utterances: list[str]) -> int:
    """Longest run of adjacent single-letter tokens ascending a->f (robust to the
    article 'a' because it requires adjacency, e.g. 'a b c d')."""
    best = 0
    for u in utterances:
        run = best_u = 0
        prev = None
        for t in re.findall(r"[a-z]+", u.lower()):
            v = LETTERS.get(t) if len(t) == 1 else None
            run = (
                run + 1
                if (v is not None and prev == (v - 1))
                else (1 if v is not None else 0)
            )
            best_u = max(best_u, run)
            prev = v
        best = max(best, best_u)
    return best


def features(utterances: list[str]) -> dict:
    text = " ".join(utterances).lower()
    words = re.findall(r"[a-z']+", text)
    nw = max(len(words), 1)
    sec, let = (
        re.compile(r"\b(section|one|two|three|four|five|six|[1-6])\b"),
        re.compile(r"\b[b-f]\b"),
    )
    feat = {
        "n_words": len(words),
        "n_utterances": len(utterances),
        "words_per_utt": len(words) / max(len(utterances), 1),
        "ttr": len(set(words)) / nw,
        "alpha_run": longest_alpha_run(utterances),
        "linkage": sum(
            bool(sec.search(u.lower())) and bool(let.search(u.lower()))
            for u in utterances
        ),
    }
    for cat, terms in LEXICON.items():
        feat[f"{cat}_rate"] = (
            sum(len(re.findall(r"\b" + t, text)) for t in terms) / nw * 100
        )
    return feat


def feature_frame(scope: str) -> pd.DataFrame:
    recs = []
    for p in participants.analyzable():
        utt = scope_utterances(p, scope)
        if utt:
            recs.append({"participant": p, **features(utt)})
    df = (
        pd.DataFrame(recs)
        .merge(vr[["participant", "aware"]], on="participant")
        .dropna(subset=["aware"])
    )
    return df


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def loo(X, y):
    return cross_val_predict(model(), X, y, cv=LeaveOneOut(), method="predict_proba")[
        :, 1
    ]


def bootstrap_ci(y, scores, n=2000):
    aucs = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        try:
            aucs.append(roc_auc_score(y[idx], scores[idx]))
        except ValueError:
            pass
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def permutation_p(X, y, observed, n=200):
    hits = sum(
        roc_auc_score(yp := RNG.permutation(y), loo(X, yp)) >= observed
        for _ in range(n)
    )
    return (hits + 1) / (n + 1)


def contrastive_words(scope, top=12, min_count=8):
    aware_w, unaware_w = Counter(), Counter()
    awareness = vr.set_index("participant")["aware"].to_dict()
    for p in participants.analyzable():
        a = awareness.get(p)
        if a != a:  # NaN
            continue
        words = re.findall(r"[a-z']+", " ".join(scope_utterances(p, scope)).lower())
        (aware_w if a == 1 else unaware_w).update(words)
    ta, tu = sum(aware_w.values()), sum(unaware_w.values())
    vocab = [
        w
        for w in set(aware_w) | set(unaware_w)
        if aware_w[w] + unaware_w[w] >= min_count
    ]
    logodds = {
        w: np.log((aware_w[w] + 1) / (ta - aware_w[w] + 1))
        - np.log((unaware_w[w] + 1) / (tu - unaware_w[w] + 1))
        for w in vocab
    }
    ranked = sorted(logodds, key=logodds.get)
    return [(w, round(logodds[w], 2)) for w in ranked[-top:][::-1]], [
        (w, round(logodds[w], 2)) for w in ranked[:top]
    ]


if __name__ == "__main__":
    print("== LOO AUC by scope (standardized logistic) ==")
    frames = {s: feature_frame(s) for s in SCOPES}
    cols = [c for c in frames["full"].columns if c not in ("participant", "aware")]
    for scope, df in frames.items():
        X, y = df[cols].to_numpy(), df["aware"].to_numpy().astype(int)
        scores = loo(X, y)
        auc = roc_auc_score(y, scores)
        lo, hi = bootstrap_ci(y, scores)
        print(
            f"{scope:11s} n={len(y)} auc={auc:.3f} [{lo:.3f}, {hi:.3f}] perm_p={permutation_p(X, y, auc):.3f}"
        )

    for scope in ("transfer", "acquisition"):
        df = frames[scope]
        X, y = df[cols].to_numpy(), df["aware"].to_numpy().astype(int)
        clf = model().fit(X, y)
        coef = sorted(
            zip(cols, clf.named_steps["logisticregression"].coef_[0]),
            key=lambda kv: -abs(kv[1]),
        )
        print(f"\n== {scope}: logistic coefficients (per SD, + => predicts aware) ==")
        for name, c in coef:
            print(f"  {name:16s} {c:+.2f}")

    aware_top, unaware_top = contrastive_words("full")
    print("\n== contrastive words (full scope, log-odds) ==")
    print("  aware:  ", ", ".join(f"{w}({s:+.1f})" for w, s in aware_top))
    print("  unaware:", ", ".join(f"{w}({s:+.1f})" for w, s in unaware_top))
