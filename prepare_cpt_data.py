#!/usr/bin/env python3
"""
CPT data preparation pipeline for bilingual Qwen2.5-1.5B continued pretraining.

Output layout (processed_data/):
    splits/                      shared across every tokenizer -- train/valid/test
                                  assignment is a content hash on raw text, so it
                                  doesn't depend on tokenization at all.
    <tag>/tokenized/, <tag>/packed/
                                  one pair per tokenizer, e.g. baseline/ (Qwen's own
                                  tokenizer, the default) and custom/ --
                                  set via --tag, so more than one tokenizer's output
                                  can live side by side without overwriting anything.

Stages (run in order; each is independently re-runnable and resumes from disk,
not RAM, so it's safe on a laptop even with multi-GB inputs):

    split      Stream each domain's raw .txt, split on <|endoftext|>, and
               deterministically assign each document to train/valid/test
               (94/3/3) via a content hash. Writes {domain}_{split}.txt files
               to processed_data/splits/ -- unaffected by --tag/--tokenizer_name.

    tokenize   Stream each split file doc-by-doc through --tokenizer_name, append
               the eos/<|endoftext|> id after each doc, and write a flat uint32
               token array (.bin) + per-doc offsets (.npy) per domain/split, to
               processed_data/<tag>/tokenized/.

    pack       Build a pointer list over (domain, doc) pairs -- repeating
               Irish train docs 4x, everything else 1x -- deterministically
               shuffle the pointers, then walk them and concatenate their
               token spans into a single stream, truncated to a whole number
               of 2048-token blocks, to processed_data/<tag>/packed/. This is
               what the base model actually trains on.

Usage:
    python prepare_cpt_data.py split
    python prepare_cpt_data.py tokenize                # --tag baseline (default), Qwen tokenizer
    python prepare_cpt_data.py pack                     # --tag baseline (default)
    python prepare_cpt_data.py all

    # Same split/, separate tokenized+packed for a custom tokenizer:
    python prepare_cpt_data.py tokenize --tokenizer_name tokenizer --tag custom
    python prepare_cpt_data.py pack     --tokenizer_name tokenizer --tag custom
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Config -- edit these to match your setup
# --------------------------------------------------------------------------

END_TOKEN = "<|endoftext|>"
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B"
BLOCK_SIZE = 2048
SEED = 2000 # First run was 1337

TRAIN_FRAC = 0.94
VALID_FRAC = 0.03
# TEST_FRAC is whatever's left (~0.03)

DOMAINS = {
    "irish":    Path("data/irish/irish_data.txt"),
    "english":  Path("data/english/english_data.txt"),
    "math":     Path("data/math/math_data.txt"),
    "code":     Path("data/code/code_data.txt"),
    "instruct": Path("data/instruct/instruct_data.txt"),
}

# how many times each domain's TRAIN documents are repeated when building the
# final packed mix. valid/test are always repeat=1 so eval stays honest and
# uncontaminated by duplicated Irish docs.
TRAIN_REPEATS = {
    "irish": 4,
    "english": 1,
    "math": 1,
    "code": 1,
    "instruct": 1,
}

# The combined valid/test pack below is for comparing against the train loss
# curve (to monitor training stability / overfitting) -- that comparison is
# only meaningful if valid is drawn from the same mixture as train, so this
# defaults to mirroring TRAIN_REPEATS. Set False to instead use each domain's
# natural, un-repeated proportion in valid/test.
MIRROR_TRAIN_MIX_IN_EVAL = True

# per-domain valid/test packs (see stage_pack_per_domain_eval) are only built
# for these domains -- math/code/instruct still train normally as part of the
# mixture, they just don't get their own tracked eval slice.
EVAL_DOMAINS = ["irish", "english"]

OUT_DIR = Path("processed_data")
SPLIT_DIR = OUT_DIR / f"splits_seed{SEED}"   # shared across tokenizers for THIS seed --
                                              # edit SEED above and rerun to get a second,
                                              # independent split without touching this one
# TOK_DIR / PACK_DIR nest under a --tag subfolder (e.g. "baseline" or
# "custom") so more than one tokenizer's tokenized/packed output can
# live side by side without overwriting each other. These module-level values
# are just the default ("baseline") for anything that imports this module
# directly (e.g. train_tokeniser_bpe.py importing SPLIT_DIR) without going
# through main() -- when running this file's own CLI, main() overwrites them
# from --tag before any stage actually runs.
TOK_DIR = OUT_DIR / "baseline" / f"tokenized_seed{SEED}"
PACK_DIR = OUT_DIR / "baseline" / f"packed_seed{SEED}"

CHUNK_SIZE = 8 * 1024 * 1024  # 8MB read chunks -> memory stays flat regardless of file size


# --------------------------------------------------------------------------
# Shared streaming document reader
# --------------------------------------------------------------------------

def iter_documents(path: Path, chunk_size: int = CHUNK_SIZE):
    """Yield documents from a large text file split on END_TOKEN, reading in
    fixed-size chunks so memory use doesn't scale with file size."""
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            parts = buffer.split(END_TOKEN)
            for doc in parts[:-1]:
                doc = doc.strip()
                if doc:
                    yield doc
            buffer = parts[-1]
    doc = buffer.strip()
    if doc:
        yield doc


def doc_split_name(doc_text: str, seed: int = SEED) -> str:
    """Deterministic train/valid/test assignment from a hash of the document
    content (+ seed). Same doc content always lands in the same split, and
    this needs no prior knowledge of corpus size, so it's a single streaming
    pass per file. It also means if a doc is later duplicated (e.g. Irish
    upsampling), all copies land in the same split -- no train/test leakage."""
    h = hashlib.md5(f"{seed}:{doc_text}".encode("utf-8")).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    if frac < TRAIN_FRAC:
        return "train"
    elif frac < TRAIN_FRAC + VALID_FRAC:
        return "valid"
    return "test"


# --------------------------------------------------------------------------
# Stage 1: split
# --------------------------------------------------------------------------

def stage_split():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for domain, path in DOMAINS.items():
        if not path.exists():
            raise FileNotFoundError(f"missing input file for domain '{domain}': {path}")

        counts = {"train": 0, "valid": 0, "test": 0}
        handles = {
            split: open(SPLIT_DIR / f"{domain}_{split}.txt", "w", encoding="utf-8")
            for split in ("train", "valid", "test")
        }
        try:
            for doc in iter_documents(path):
                split = doc_split_name(doc)
                handles[split].write(doc)
                handles[split].write(END_TOKEN)
                counts[split] += 1
        finally:
            for h in handles.values():
                h.close()

        total = sum(counts.values())
        if total == 0:
            print(f"[split] {domain}: WARNING - no documents found in {path}")
            continue
        print(f"[split] {domain:10s} total={total:>9,d}  "
              f"train={counts['train']:>9,d} ({counts['train']/total:.1%})  "
              f"valid={counts['valid']:>9,d} ({counts['valid']/total:.1%})  "
              f"test={counts['test']:>9,d} ({counts['test']/total:.1%})")


# --------------------------------------------------------------------------
# Stage 2: tokenize
# --------------------------------------------------------------------------

def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    eos_id = tok.convert_tokens_to_ids(END_TOKEN)
    if eos_id is None or eos_id == tok.unk_token_id:
        eos_id = tok.eos_token_id
    if eos_id is None:
        raise ValueError(
            f"couldn't resolve a token id for {END_TOKEN!r} in the "
            f"{TOKENIZER_NAME} tokenizer -- check its special tokens."
        )
    return tok, eos_id


def stage_tokenize(batch_size: int = 1000):
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, eos_id = load_tokenizer()
    print(f"[tokenize] eos/<|endoftext|> token id = {eos_id}")

    for domain in DOMAINS:
        for split in ("train", "valid", "test"):
            src = SPLIT_DIR / f"{domain}_{split}.txt"
            if not src.exists():
                continue
            bin_path = TOK_DIR / f"{domain}_{split}.bin"
            off_path = TOK_DIR / f"{domain}_{split}_offsets.npy"

            offsets = [0]
            cursor = 0
            batch = []
            n_seen = 0

            with open(bin_path, "wb") as bin_f:
                def flush(batch):
                    nonlocal cursor
                    if not batch:
                        return
                    enc = tokenizer(batch, add_special_tokens=False)["input_ids"]
                    for ids in enc:
                        ids.append(eos_id)
                        arr = np.array(ids, dtype=np.uint32)
                        bin_f.write(arr.tobytes())
                        cursor += len(arr)
                        offsets.append(cursor)

                for doc in iter_documents(src):
                    batch.append(doc)
                    n_seen += 1
                    if len(batch) >= batch_size:
                        flush(batch)
                        batch = []
                        if n_seen % 50000 < batch_size:
                            print(f"[tokenize]   {domain}_{split} ... {n_seen:,} docs")
                flush(batch)

            np.save(off_path, np.array(offsets, dtype=np.int64))
            n_docs = len(offsets) - 1
            print(f"[tokenize] {domain}_{split:8s}  docs={n_docs:>9,d}  tokens={cursor:>13,d}")


# --------------------------------------------------------------------------
# Stage 3: mix + shuffle + pack
# --------------------------------------------------------------------------

def stage_pack(block_size: int = BLOCK_SIZE, seed: int = SEED):
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    domain_names = list(DOMAINS.keys())

    for split in ("train", "valid", "test"):
        offsets, bins = {}, {}
        for domain in domain_names:
            off_path = TOK_DIR / f"{domain}_{split}_offsets.npy"
            bin_path = TOK_DIR / f"{domain}_{split}.bin"
            if not off_path.exists():
                continue
            offsets[domain] = np.load(off_path)
            bins[domain] = np.memmap(bin_path, dtype=np.uint32, mode="r")

        active_domains = [d for d in domain_names if d in offsets]

        # flatten every domain's docs into shared lookup arrays: global doc i
        # belongs to domain global_domain_id[i], with tokens at
        # bins[domain][global_start[i]:global_end[i]]
        g_domain_id, g_start, g_end = [], [], []
        base_range = {}
        cur = 0
        for d_id, domain in enumerate(active_domains):
            offs = offsets[domain]
            n_docs = len(offs) - 1
            g_domain_id.append(np.full(n_docs, d_id, dtype=np.uint8))
            g_start.append(offs[:-1])
            g_end.append(offs[1:])
            base_range[domain] = (cur, n_docs)
            cur += n_docs

        g_domain_id = np.concatenate(g_domain_id)
        g_start = np.concatenate(g_start)
        g_end = np.concatenate(g_end)
        g_len = g_end - g_start

        # pointer list into the global arrays, repeated per TRAIN_REPEATS.
        # valid/test mirror the train repeats too (when MIRROR_TRAIN_MIX_IN_EVAL,
        # the default) so this combined pack is drawn from the same mixture as
        # train and its loss is directly comparable to the train loss curve.
        apply_repeats = split == "train" or MIRROR_TRAIN_MIX_IN_EVAL
        ptr_chunks = []
        for domain in active_domains:
            base_off, n_docs = base_range[domain]
            repeat = TRAIN_REPEATS.get(domain, 1) if apply_repeats else 1
            local_idx = np.arange(n_docs, dtype=np.int64) + base_off
            ptr_chunks.append(np.tile(local_idx, repeat))
        ptrs = np.concatenate(ptr_chunks)

        rng = np.random.default_rng(seed if split == "train" else seed + 1)
        ptrs = rng.permutation(ptrs)

        total_tokens = int(g_len[ptrs].sum())
        num_blocks = total_tokens // block_size
        out_len = num_blocks * block_size
        out_path = PACK_DIR / f"{split}.bin"
        out = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(out_len,))

        cursor = 0
        for i, p in enumerate(ptrs):
            if cursor >= out_len:
                break
            domain = active_domains[g_domain_id[p]]
            start, end = g_start[p], g_end[p]
            tokens = bins[domain][start:end]
            n = len(tokens)
            if cursor + n <= out_len:
                out[cursor:cursor + n] = tokens
                cursor += n
            else:
                remaining = out_len - cursor
                out[cursor:cursor + remaining] = tokens[:remaining]
                cursor += remaining
                break
            if i % 500000 == 0 and i > 0:
                print(f"[pack]   {split} ... {i:,}/{len(ptrs):,} docs placed")

        out.flush()
        print(f"[pack] {split:5s}  blocks={num_blocks:>9,d}  "
              f"tokens={out_len:>13,d}  -> {out_path}")

        meta = {
            "split": split,
            "block_size": block_size,
            "num_blocks": int(num_blocks),
            "num_tokens": int(out_len),
            "seed": int(seed if split == "train" else seed + 1),
            "domains": active_domains,
            "train_repeats": {d: (TRAIN_REPEATS.get(d, 1) if apply_repeats else 1)
                               for d in active_domains},
        }
        with open(PACK_DIR / f"{split}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    stage_pack_per_domain_eval(block_size=block_size)


def stage_pack_per_domain_eval(block_size: int = BLOCK_SIZE, seed: int = SEED + 2):
    """Additionally pack each of EVAL_DOMAINS' valid/test tokens on their own
    -- no mixing with other domains, no repeats -- so you can compute
    perplexity per domain (e.g. is English perplexity creeping up while Irish
    improves). The combined valid/test pack above answers a different
    question (does the blended loss track the train loss curve); this
    answers "which domain is actually driving that number," for whichever
    domains you've listed in EVAL_DOMAINS.
    """
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    for domain in EVAL_DOMAINS:
        for split in ("valid", "test"):
            off_path = TOK_DIR / f"{domain}_{split}_offsets.npy"
            bin_path = TOK_DIR / f"{domain}_{split}.bin"
            if not off_path.exists():
                continue
            offsets = np.load(off_path)
            tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
            n_docs = len(offsets) - 1

            rng = np.random.default_rng(seed)
            order = rng.permutation(n_docs)

            total_tokens = int((offsets[1:] - offsets[:-1]).sum())
            num_blocks = total_tokens // block_size
            out_len = num_blocks * block_size
            out_path = PACK_DIR / f"{domain}_{split}.bin"
            out = np.memmap(out_path, dtype=np.uint32, mode="w+", shape=(out_len,))

            cursor = 0
            for doc_idx in order:
                if cursor >= out_len:
                    break
                start, end = offsets[doc_idx], offsets[doc_idx + 1]
                doc_tokens = tokens[start:end]
                n = len(doc_tokens)
                if cursor + n <= out_len:
                    out[cursor:cursor + n] = doc_tokens
                    cursor += n
                else:
                    remaining = out_len - cursor
                    out[cursor:cursor + remaining] = doc_tokens[:remaining]
                    cursor += remaining
                    break
            out.flush()
            print(f"[pack-domain] {domain}_{split:8s}  blocks={num_blocks:>7,d}  "
                  f"tokens={out_len:>10,d}  -> {out_path}")


# --------------------------------------------------------------------------

def main():
    global TOKENIZER_NAME, TOK_DIR, PACK_DIR

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["split", "tokenize", "pack", "all"])
    parser.add_argument("--tokenizer_name", default=TOKENIZER_NAME,
                         help="HF tokenizer name or local path for the tokenize/pack stages. "
                              "Defaults to the Qwen baseline. Point this at a custom "
                              "tokenizer's directory to build a second, parallel packed "
                              "dataset -- pair with --tag so it lands in its own subfolder.")
    parser.add_argument("--tag", default="baseline",
                         help="subfolder under processed_data/ for this tokenizer's "
                              "tokenized/ and packed/ output, e.g. 'baseline' (default) or "
                              "'custom'. processed_data/splits/ is always shared "
                              "regardless of --tag -- split assignment doesn't depend on "
                              "tokenization, so there's no reason to duplicate it.")
    args = parser.parse_args()

    TOKENIZER_NAME = args.tokenizer_name
    TOK_DIR = OUT_DIR / args.tag / f"tokenized_seed{SEED}"
    PACK_DIR = OUT_DIR / args.tag / f"packed_seed{SEED}"

    if args.stage in ("split", "all"):
        stage_split()
    if args.stage in ("tokenize", "all"):
        stage_tokenize()
    if args.stage in ("pack", "all"):
        stage_pack()


if __name__ == "__main__":
    main()