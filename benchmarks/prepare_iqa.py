"""
Downloads and converts IrishQA (both en and ga) into the JSONL format
evaluate_iqa.py expects, plus ready-to-use few-shot example lists. Run
once; evaluate_iqa.py handles all 8 baseline/custom x pre/post-SFT x
en/ga combinations afterward.

Usage:
    python prepare_iqa.py
"""
import json
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).resolve().parent


def convert_row(row):
    choices = [row["mc_answer1"], row["mc_answer2"], row["mc_answer3"], row["mc_answer4"]]
    answer_index = row["correct_answer_num"] - 1  # dataset is 1-indexed
    return {"question": row["question"], "context": row["context"], "choices": choices, "answer_index": answer_index}


for lang in ["en", "ga"]:
    ds = load_dataset("ReliableAI/IrishQA", lang)
    main_split = "test" if "test" in ds else [s for s in ds if s != "few_shot"][0]

    with open(SCRIPT_DIR / f"iqa_{lang}.jsonl", "w", encoding="utf-8") as f:
        for row in ds[main_split]:
            f.write(json.dumps(convert_row(row), ensure_ascii=False) + "\n")
    print(f"[{lang}] wrote {len(ds[main_split])} rows (split: {main_split})")

    fewshot = [convert_row(row) for row in ds["few_shot"]]
    with open(SCRIPT_DIR / f"iqa_{lang}_fewshot.json", "w", encoding="utf-8") as f:
        json.dump(fewshot, f, ensure_ascii=False, indent=2)
    print(f"[{lang}] wrote {len(fewshot)} few-shot examples")
