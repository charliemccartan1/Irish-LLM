"""
Converts UCCIX's real multichoice_cloze.json (the actual Cloze-gle data
-- confirmed real, already used successfully in the UCCIX eval run) into
the format evaluate_cloze.py expects.

Real, confirmed schema per item:
    {"label": 0, "sentence1": "...", "sentence2": "...", "sentence3": "..."}
label is a 0-indexed pointer to which sentence is grammatically correct.

Run this directly on Hypatia -- the source data is already local (part
of the UCCIX clone), no download/internet needed.

Usage:
    python prepare_cloze.py
"""
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = SCRIPT_DIR / "UCCIX" / "eval" / "lm_eval" / "tasks" / "irish_cloze" / "multichoice_cloze.json"

with open(SOURCE_PATH, encoding="utf-8") as f:
    raw_items = json.load(f)

print(f"loaded {len(raw_items)} raw items from {SOURCE_PATH}")

converted = []
for row in raw_items:
    converted.append({
        "candidates": [row["sentence1"], row["sentence2"], row["sentence3"]],
        "correct_index": row["label"],
    })

with open(SCRIPT_DIR / "cloze_test.jsonl", "w", encoding="utf-8") as f:
    for item in converted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"wrote {len(converted)} items to cloze_test.jsonl")

# Sanity check against the real example already confirmed by hand:
# "An í Eilís an bainisteoir?" with label=0 -- í (feminine) correct for
# a female name, matching the grammatical pattern used throughout
sample = converted[0]
assert sample["candidates"][sample["correct_index"]] == "An í Eilís an bainisteoir?"
print("sanity check passed: first item's correct_index correctly points to the í (feminine) variant")
