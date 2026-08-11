"""
Downloads gaHealth (Lankford et al. 2022) -- real, confirmed source (exact
match to Qomhra's stated upper-bound BLEU: 57.6 gle2eng, 37.6 eng2gle).
Plain, line-aligned text files, not an HF dataset.

Usage:
    python prepare_bleu.py
"""
import json
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = "https://raw.githubusercontent.com/seamusl/gaHealth/main/"
N_SHOT = 5

FILES = ["test.en2ga.en", "test.en2ga.ga", "test.ga2en.en", "test.ga2en.ga", "dev.en", "dev.ga"]

for fname in FILES:
    out_path = SCRIPT_DIR / fname
    print(f"downloading {fname}...")
    urllib.request.urlretrieve(BASE_URL + fname, out_path)

with open(SCRIPT_DIR / "dev.en", encoding="utf-8") as f:
    dev_en = [line.strip() for line in f]
with open(SCRIPT_DIR / "dev.ga", encoding="utf-8") as f:
    dev_ga = [line.strip() for line in f]
assert len(dev_en) == len(dev_ga), "dev.en/dev.ga line count mismatch -- files may be corrupted"

for direction, src_lines, tgt_lines in [("en2ga", dev_en, dev_ga), ("ga2en", dev_ga, dev_en)]:
    fewshot = [{"source": s, "target": t} for s, t in zip(src_lines[:N_SHOT], tgt_lines[:N_SHOT])]
    with open(SCRIPT_DIR / f"bleu_{direction}_fewshot.json", "w", encoding="utf-8") as f:
        json.dump(fewshot, f, ensure_ascii=False, indent=2)
    print(f"[{direction}] wrote {len(fewshot)} few-shot examples")

for direction, src_file, tgt_file in [("en2ga", "test.en2ga.en", "test.en2ga.ga"),
                                        ("ga2en", "test.ga2en.ga", "test.ga2en.en")]:
    with open(SCRIPT_DIR / src_file, encoding="utf-8") as f:
        sources = [line.strip() for line in f]
    with open(SCRIPT_DIR / tgt_file, encoding="utf-8") as f:
        references = [line.strip() for line in f]
    assert len(sources) == len(references), f"[{direction}] line count mismatch -- files may be corrupted"
    pairs = [{"source": s, "reference": r} for s, r in zip(sources, references)]
    with open(SCRIPT_DIR / f"bleu_{direction}_test.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"[{direction}] wrote {len(pairs)} test pairs")
