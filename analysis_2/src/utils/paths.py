from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "analysis_2"
RESSOURCES = REPO / "ressources"
DF_VR = ROOT / "df_vr.csv"
TAXONOMY = ROOT / "vr_prompts-v2_1.xlsx"
OUTPUT = ROOT / "output"
