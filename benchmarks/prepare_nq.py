"""
Downloads and converts NQ-open into the JSONL format evaluate_nq.py
expects, plus a 5-shot example file. Real source confirmed: nq_open (not
the full 45GB natural_questions -- nq_open's simpler question+answer-list
structure matches Qomhra's own Table 8 example exactly).

Usage:
    python prepare_nq.py
"""
import json
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
N_SHOT = 5

ds = load_dataset("google-research-datasets/nq_open")
print(f"splits found: {list(ds.keys())}")

# Held-out few-shot examples from train, separate from the eval set
fewshot = [{"question": row["question"], "answer": row["answer"]} for row in ds["train"].select(range(N_SHOT))]
with open(SCRIPT_DIR / "nq_fewshot.json", "w", encoding="utf-8") as f:
    json.dump(fewshot, f, ensure_ascii=False, indent=2)
print(f"wrote {len(fewshot)} few-shot examples to nq_fewshot.json")

eval_split = "validation" if "validation" in ds else [s for s in ds if s != "train"][0]
with open(SCRIPT_DIR / "nq_eng.jsonl", "w", encoding="utf-8") as f:
    for row in ds[eval_split]:
        f.write(json.dumps({"question": row["question"], "answer": row["answer"]}, ensure_ascii=False) + "\n")
print(f"wrote {len(ds[eval_split])} rows to nq_eng.jsonl (split: {eval_split})")
