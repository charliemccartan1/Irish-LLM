#!/usr/bin/env python3
"""
Compares tokenisation fertility (tokens per word, and characters per token)
between Qwen2.5's default tokeniser and a custom one (e.g. tokenizer/,
from train_tokeniser_bpe.py), per domain and overall.

Measured on output/splits/{domain}_valid.txt -- the VALID split, not train --
so this reports honest, held-out fertility rather than a number inflated by
measuring a BPE tokeniser on the exact text its merges were learned from.

Usage:
    python measure_tokeniser_bpe.py
    python measure_tokeniser_bpe.py --custom_tokenizer tokenizer \\
        --baseline_tokenizer /sharedscratch/$USER/dissertation/qwen2.5-1.5b
"""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from prepare_cpt_data import SPLIT_DIR, iter_documents

# Only the two domains this project actually cares about comparing --
# math/code/instruct fertility isn't the point of building this tokeniser.
DOMAINS_TO_CHECK = ["irish", "english"]


def measure(tokenizer, path: Path):
    """Returns (word_count, char_count, token_count) for one domain's valid
    split, streamed doc-by-doc so this stays memory-flat regardless of file
    size. Word count is a simple whitespace split -- matches the "tokens per
    word" framing fertility is usually reported in. Character count enables
    the complementary "characters per token" metric."""
    n_words = 0
    n_chars = 0
    n_tokens = 0
    for doc in iter_documents(path):
        n_words += len(doc.split())
        n_chars += len(doc)
        n_tokens += len(tokenizer(doc, add_special_tokens=False)["input_ids"])
    return n_words, n_chars, n_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--custom_tokenizer", default="tokenizer",
                    help="path to the custom tokeniser dir (default: tokenizer, "
                         "matching train_tokeniser_bpe.py's OUT_DIR)")
    p.add_argument("--baseline_tokenizer", default="Qwen/Qwen2.5-1.5B")
    args = p.parse_args()

    baseline = AutoTokenizer.from_pretrained(args.baseline_tokenizer)
    custom = AutoTokenizer.from_pretrained(args.custom_tokenizer)
    print(f"[measure_tokeniser_bpe] baseline vocab={baseline.vocab_size:,}  "
          f"custom vocab={custom.vocab_size:,}\n")

    header = (f"{'domain':10s} {'words':>12s} "
              f"{'baseline tok/word':>18s} {'custom tok/word':>16s} "
              f"{'baseline chr/tok':>17s} {'custom chr/tok':>15s}")
    print(header)
    print("-" * len(header))

    totals = {"words": 0, "chars": 0, "baseline_tokens": 0, "custom_tokens": 0}

    for domain in DOMAINS_TO_CHECK:
        path = SPLIT_DIR / f"{domain}_valid.txt"
        if not path.exists():
            print(f"[measure_tokeniser_bpe] skipping '{domain}' -- {path} not found")
            continue

        words, chars, baseline_tokens = measure(baseline, path)
        _, _, custom_tokens = measure(custom, path)  # words/chars are tokenizer-independent

        totals["words"] += words
        totals["chars"] += chars
        totals["baseline_tokens"] += baseline_tokens
        totals["custom_tokens"] += custom_tokens

        print(f"{domain:10s} {words:>12,d} "
              f"{baseline_tokens / words:>18.2f} {custom_tokens / words:>16.2f} "
              f"{chars / baseline_tokens:>17.2f} {chars / custom_tokens:>15.2f}")

    print("-" * len(header))
    if totals["words"] > 0:
        print(f"{'overall':10s} {totals['words']:>12,d} "
              f"{totals['baseline_tokens'] / totals['words']:>18.2f} "
              f"{totals['custom_tokens'] / totals['words']:>16.2f} "
              f"{totals['chars'] / totals['baseline_tokens']:>17.2f} "
              f"{totals['chars'] / totals['custom_tokens']:>15.2f}")


if __name__ == "__main__":
    main()
