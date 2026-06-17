import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent

load_dotenv(HERE / ".env")
client = OpenAI(
    base_url=os.environ.get("OPENWEBUI_BASE_URL", "https://ai.cogpsy.fun/api"),
    api_key=os.environ["OPENWEBUI_API_KEY"],
)

MODEL = "qwen3.6:27b"
N_PARTICIPANTS_TO_PROCESS = None  # How many participants to run, None for all
N_RUNS = 5  # judgements per participant, one per seed (long format: N_RUNS rows each)
RESSOURCES = HERE.parent / "ressources"
OUT = HERE / "output" / "letter_rule.csv"

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "letter_rule_judgement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rule_evidence": {
                    "type": "string",
                    "enum": ["explicit", "partial", "none"],
                },
                "evidence": {"type": "string"},
            },
            "required": ["rule_evidence", "evidence"],
            "additionalProperties": False,
        },
    },
}

PROMPT = """Below is the full think-aloud transcript of one participant in a \
decision-making experiment. In each round, one of six sections contained the \
winning option. The winning section always cycled predictably: Section 1 in \
round 1, Section 2 in round 2, ... Section 6 in round 6, then Section 1 \
again, and so on. Noticing this section cycle alone is NOT what we are \
looking for.

In addition, there was a hidden letter rule: when Section 1 contained the \
winner, option A won; Section 2 -> B; Section 3 -> C; Section 4 -> D; \
Section 5 -> E; Section 6 -> F. This rule held during only one half of the \
experiment (which half varied between participants); in the other half, the \
winning letters varied irregularly.

Did the participant discover this letter rule at any point? Remarks that the \
letter rule started working or stopped working (e.g. "now the letters match \
the sections" or "the letter thing doesn't work anymore") also count as \
evidence of having found it. Rate the strongest evidence in the transcript \
on this scale:
- "explicit": states the section-number-to-letter mapping as a rule \
(e.g. "the winning letter matches the section number").
- "partial": uses the mapping correctly in the moment (e.g. "section 3, so it \
must be C") multiple times without articulating it as a rule.
- "none": no linkage between section number and letter; merely guessing or \
naming letters without tying them to the section number does not count.

Quote the decisive passage as evidence (or the closest near miss if the \
rating is "none").

Transcript:
{transcript}"""


OUT.parent.mkdir(exist_ok=True)
rows = (
    pd.read_csv(OUT).to_dict("records") if OUT.exists() else []
)  # resume previous run
done = {(row["participant"], row["seed"]) for row in rows}
for csv in sorted(RESSOURCES.glob("*/transcriptions.csv"))[:N_PARTICIPANTS_TO_PROCESS]:
    df = pd.read_csv(csv).sort_values("filename", ignore_index=True)
    # only judge speech from before the rule was revealed on screen,
    # i.e. up to the end of the sequence generation task
    revealed = df.index[df["filename"].str.contains("ruledetection")]
    if len(revealed):
        df = df.iloc[: revealed[0]]
    transcript = "\n\n".join(df["text"].dropna())
    if len(transcript) < 200:  # mic failure, nothing to judge
        print(
            f"skipping {csv.parent.name}: transcript has only {len(transcript)} chars"
        )
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
        judgement = json.loads(content[content.find("{") : content.rfind("}") + 1])
        rows.append({"participant": csv.parent.name, "seed": seed, **judgement})
        print(
            f"[{len(rows)}] {csv.parent.name} seed={seed}: "
            f"rule_evidence={judgement['rule_evidence']}"
        )
        pd.DataFrame(rows).to_csv(OUT, index=False)  # save progress each step
