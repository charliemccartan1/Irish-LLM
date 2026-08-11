#!/usr/bin/env python3
"""
Evaluate a saved CPT checkpoint against the held-out test set. Deliberately
separate from train_cpt.py so you can run this against any epoch checkpoint
whenever you're ready, rather than it happening automatically right after
training finishes.

Reads test_{block}.bin (combined, mixture-mirrored) and {domain}_test_{block}.bin
(per-domain, unmixed) -- these need to be uploaded to Hypatia before running this
(see README.md; they aren't needed for training itself, only for this step).

Writes a single JSON of all metrics, by default to <checkpoint_dir>/test_metrics.json
-- matching the path analyze_tokeniser_comparison.py looks for when pointed at
--metrics_path final/test_metrics.json (i.e. run this with --checkpoint_dir
pointing at a run's final/ directory).

Usage:
    python evaluate_test_set.py \
        --checkpoint_dir /sharedscratch/$USER/dissertation/output/runs/original/seed1/final \
        --packed_dir /sharedscratch/$USER/dissertation/processed_data/baseline/packed
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from train_cpt import PackedDataset, make_compute_metrics, preprocess_logits_for_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--packed_dir", type=Path, required=True)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--domains", nargs="+", default=["irish", "english"])
    p.add_argument("--attn_implementation", default="sdpa")
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--out", type=Path, default=None,
                    help="defaults to <checkpoint_dir>/test_metrics.json")
    p.add_argument("--adapter_path", type=Path, default=None,
                    help="LoRA adapter dir for POST-sft evaluation. Omit for PRE-sft "
                         "(CPT-only). Loaded on top of --checkpoint_dir's base weights, "
                         "then merged into the model before evaluation.")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="dtype to load model weights in. Default bfloat16 matches the real "
                         "GPU run; use float32 for a CPU-only local check.")
    p.add_argument("--no_bf16_training", action="store_true",
                    help="disable TrainingArguments(bf16=True). Set this for CPU-only local "
                         "checks. Leave unset on the real GPU run.")
    p.add_argument("--dataloader_num_workers", type=int, default=0,
                    help="0 (default) matches HF's own default and avoids Windows "
                         "multiprocessing quirks for local checks; bump up on Hypatia if wanted.")
    p.add_argument("--eval_max_blocks", type=int, default=None,
                    help="if set, caps every test set (combined + each domain) to at most this "
                         "many blocks. The real test.bin/{domain}_test.bin files are as large "
                         "as the valid.bin equivalents (tens of thousands of blocks) -- fine on "
                         "an A100, but a full pass can take hours/days on a laptop CPU. Leave "
                         "unset when you want the real, full-test-set number.")
    args = p.parse_args()
    out_path = args.out or (args.checkpoint_dir / "test_metrics.json")

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir, dtype=dtype_map[args.torch_dtype], attn_implementation=args.attn_implementation,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir, torch_dtype=dtype_map[args.torch_dtype], attn_implementation=args.attn_implementation,
        )

    if args.adapter_path is not None:
        from peft import PeftModel
        print(f"[evaluate_test_set] loading LoRA adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    test_ds = {"combined": PackedDataset(args.packed_dir / "test.bin", args.block_size)}
    for domain in args.domains:
        path = args.packed_dir / f"{domain}_test.bin"
        if path.exists():
            test_ds[domain] = PackedDataset(path, args.block_size)
        else:
            print(f"[evaluate_test_set] skipping '{domain}' -- {path} not found")

    if args.eval_max_blocks is not None:
        for name, ds in test_ds.items():
            n = min(len(ds), args.eval_max_blocks)
            print(f"[evaluate_test_set] --eval_max_blocks: capping '{name}' test set to "
                  f"{n} blocks (was {len(ds)})")
            test_ds[name] = Subset(ds, range(n))

    eval_args = TrainingArguments(
        output_dir=str(args.checkpoint_dir),   # scratch dir for Trainer's own bookkeeping only
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=not args.no_bf16_training,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    all_metrics = {}
    for name, ds in test_ds.items():
        prefix = "test" if name == "combined" else f"test_{name}"
        metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix=prefix)
        print(f"[evaluate_test_set] {name}: {metrics}")
        all_metrics.update(metrics)

    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[evaluate_test_set] wrote {out_path}")


if __name__ == "__main__":
    main()