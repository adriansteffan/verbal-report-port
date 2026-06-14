import sys
from pathlib import Path

import numpy as np
import pandas as pd

import participants

HERE = Path(__file__).resolve().parent
RESSOURCES = HERE.parent / "ressources"
EMB = pd.read_parquet(HERE / "output" / "embeddings.parquet")
vr = pd.read_csv(HERE / "df_vr.csv")


def l2norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def utterance_text(participant, filename):
    df = pd.read_csv(RESSOURCES / participant / "transcriptions.csv")
    m = df.loc[df["filename"] == filename, "text"]
    return m.iloc[0] if len(m) and isinstance(m.iloc[0], str) else ""


def exemplars(scope="transfer", dim=512, n=15):
    keep = set(participants.analyzable())
    aware = vr.set_index("participant")["aware"].to_dict()
    utt = EMB[(EMB["kind"] == "utterance") & EMB["participant"].isin(keep)].reset_index(
        drop=True
    )
    utt = utt.assign(aware=utt["participant"].map(aware))
    U = l2norm(np.vstack(utt["embedding"].to_numpy())[:, :dim])

    phase_mask = True if scope == "full" else (utt["phase"] == scope)
    in_scope = (utt["aware"].notna() & phase_mask).to_numpy()
    lab = utt.loc[in_scope, "aware"].to_numpy()
    direction = U[in_scope][lab == 1].mean(0) - U[in_scope][lab == 0].mean(0)
    direction /= np.linalg.norm(direction)

    ranked = utt.assign(proj=U @ direction).sort_values("proj", ascending=False)

    def show(block, label):
        print(
            f"\n--- most {label}-like utterances (direction from {scope}, dim={dim}) ---"
        )
        for r in block.itertuples():
            rnd = f"{int(r.round)}" if r.round == r.round else "-"
            text = " ".join(utterance_text(r.participant, r.key).split())
            print(f"{r.proj:+.3f}  aware={r.aware}  {r.phase}/{rnd}: {text[:130]}")

    show(ranked.head(n), "aware")
    show(ranked.tail(n)[::-1], "unaware")


if __name__ == "__main__":
    exemplars(sys.argv[1] if len(sys.argv) > 1 else "transfer")
