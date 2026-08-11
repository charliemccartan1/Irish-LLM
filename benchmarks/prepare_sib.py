"""
Downloads SIB-Ga (Davlan/sib200, gle_Latn config -- confirmed real
source) and converts to the format evaluate_sib.py expects. Pulls the 7
real Irish topic labels directly from the dataset's own ClassLabel
metadata, rather than guessing translations -- verified below against
Qomhra's own confirmed examples (eolaiocht/teicneolaiocht, polaitiocht).

Usage:
    python prepare_sib.py
"""
import json
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
N_SHOT = 10

ds = load_dataset("Davlan/sib200", "gle_Latn")
print(f"splits found: {list(ds.keys())}")

label_feature = ds["train"].features["category"] if "category" in ds["train"].features else ds["train"].features["label"]
label_col = "category" if "category" in ds["train"].features else "label"
print(f"label column: {label_col}, sample raw value: {ds['train'][0][label_col]!r}")

# Confirm whether labels are already strings (Irish category names) or
# integer IDs needing a names lookup
if isinstance(ds["train"][0][label_col], str):
    label_names = sorted(set(ds["train"][label_col]))
else:
    label_names = label_feature.names

print(f"resolved {len(label_names)} label names: {label_names}")
assert len(label_names) == 7, f"expected 7 categories, found {len(label_names)} -- check label resolution above"

# Sanity check against Qomhra's own confirmed examples
science_present = any("eolaíocht" in n or "eolaiocht" in n for n in label_names)
politics_present = any("polaitíocht" in n or "polaitiocht" in n for n in label_names)
print(f"sanity check -- science/tech label found: {science_present}, politics label found: {politics_present}")
if not (science_present and politics_present):
    print("WARNING: could not confirm label text against Qomhra's own examples -- verify label_names above manually")


def row_to_example(row):
    label_value = row[label_col]
    label_index = label_names.index(label_value) if isinstance(label_value, str) else label_value
    return {"text": row["text"], "label_index": label_index}


fewshot = [row_to_example(row) for row in ds["train"].select(range(N_SHOT))]
with open(SCRIPT_DIR / "sib_fewshot.json", "w", encoding="utf-8") as f:
    json.dump({"labels": label_names, "examples": fewshot}, f, ensure_ascii=False, indent=2)
print(f"wrote {len(fewshot)} few-shot examples")

eval_split = "test" if "test" in ds else [s for s in ds if s != "train"][0]
with open(SCRIPT_DIR / "sib_test.jsonl", "w", encoding="utf-8") as f:
    for row in ds[eval_split]:
        f.write(json.dumps(row_to_example(row), ensure_ascii=False) + "\n")
print(f"wrote {len(ds[eval_split])} test rows (split: {eval_split})")

with open(SCRIPT_DIR / "sib_labels.json", "w", encoding="utf-8") as f:
    json.dump(label_names, f, ensure_ascii=False, indent=2)
