import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import participants

HERE = Path(__file__).resolve().parent

load_dotenv(HERE / ".env")
client = OpenAI(
    base_url=os.environ.get("OPENWEBUI_BASE_URL", "https://ai.cogpsy.fun/api"),
    api_key=os.environ["OPENWEBUI_API_KEY"],
)

MODEL = "qwen3.6:27b"
N_PARTICIPANTS_TO_PROCESS = None  # How many participants to run, None for all
N_RUNS = 5
RESSOURCES = HERE.parent / "ressources"
OUT = HERE / "output" / "predict_learning.csv"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "learning_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "will_learn": {
                    "type": "string",
                    "enum": ["unlikely", "uncertain", "likely"],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["will_learn", "reasoning"],
            "additionalProperties": False,
        },
    },
}

PROMPT = """Below is the think-aloud transcript of one participant during the \
FIRST half of a decision-making experiment (the "acquisition" phase). In every \
round one of six sections contained the winning option, and the winning section \
cycled predictably: Section 1 in round 1, Section 2 in round 2, ... Section 6 \
in round 6, then repeating. During this first half the winning letters were \
arbitrary -- there was no rule to be found yet. Later, in a second half (not \
shown), a hidden rule governing the winning option is introduced.

Do NOT look for rule discovery here -- there is no rule to find yet, so the \
absence of rule talk means nothing. Instead, judge the participant's general \
problem-solving disposition from how they think aloud: do they actively generate \
and test hypotheses, search for patterns and regularities, track and compare \
outcomes across rounds, and reason systematically -- or do they choose \
passively, guess, or disengage? Based on that disposition, predict whether this \
person will later uncover the hidden rule:
- "likely": an active, systematic hypothesis-tester / pattern-seeker -- the kind \
of reasoner who tends to find hidden structure.
- "uncertain": mixed or insufficient signal.
- "unlikely": passive, guessing, or disengaged.

Briefly justify your prediction by describing their exploration style.

Transcript:
{transcript}"""


def acquisition_text(csv: Path) -> str:
    df = pd.read_csv(csv).sort_values("filename")
    phase = df["filename"].str.extract(r"audio_\d+_(?P<phase>.+)_\d+\.wav")["phase"]
    texts = df["text"][phase.values == "acquisition"].dropna()  # column first -> Series
    return "\n\n".join(
        t for t in (str(x).strip() for x in texts) if participants.is_english(t)
    )


OUT.parent.mkdir(exist_ok=True)
rows = pd.read_csv(OUT).to_dict("records") if OUT.exists() else []  # resume
done = {(row["participant"], row["seed"]) for row in rows}
for csv in sorted(RESSOURCES.glob("*/transcriptions.csv"))[:N_PARTICIPANTS_TO_PROCESS]:
    transcript = acquisition_text(csv)
    if len(transcript) < 200:  # too little acquisition speech to predict from
        print(f"skipping {csv.parent.name}: only {len(transcript)} acquisition chars")
        continue
    for seed in range(N_RUNS):
        if (csv.parent.name, seed) in done:
            continue
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": PROMPT.format(transcript=transcript)}
            ],
            response_format=RESPONSE_FORMAT,  # pyright: ignore[reportArgumentType]
            max_tokens=8000,  # thinking models need headroom
            seed=seed,
            temperature=0.6,
        )
        content = (response.choices[0].message.content or "").split("</think>")[-1]
        prediction = json.loads(content[content.find("{") : content.rfind("}") + 1])
        rows.append({"participant": csv.parent.name, "seed": seed, **prediction})
        print(
            f"[{len(rows)}] {csv.parent.name} seed={seed}: will_learn={prediction['will_learn']}"
        )
        pd.DataFrame(rows).to_csv(OUT, index=False)  # save progress each step
