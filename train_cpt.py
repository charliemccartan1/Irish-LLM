#!/usr/bin/env python3
"""
CPT training script for Qwen2.5-1.5B on the packed dataset produced by
prepare_cpt_data.py.

Reads train_{block}.bin for training, valid_{block}.bin (the mixture-mirrored
combined pack) plus {domain}_valid_{block}.bin (per-domain, unmixed) for eval,
so you get both a train/valid stability comparison and per-domain perplexity
in the same Trainer run (HF Trainer accepts eval_dataset as a dict).

Usage (single node, multi-GPU):
    accelerate launch --num_processes=4 --mixed_precision=bf16 train_cpt.py \
        --packed_dir processed_data/baseline/packed --block_size 2048 \
        --output_dir /sharedscratch/$USER/dissertation/output/runs/original/seed1
"""
import argparse
import inspect
import json
import os
import sys
from pathlib import Path

# Both real job scripts (train_baseline_cpt.slurm/train_custom_cpt.slurm) are
# already submitted and can't be resubmitted, so neither of the two fixes
# below can be added at the shell/.slurm level -- they're applied here
# instead, as early as possible.
#
# PYTHONUNBUFFERED can't be set via os.environ from inside the script --
# it controls the interpreter's own stdout buffering mode, decided at
# Python startup, before this code ever runs. Reconfiguring stdout directly
# achieves the same practical effect: real progress won't sit invisible in
# a buffer the way it did earlier in this project (job 4334143 looked
# hung for ~20 minutes before a manual GPU check revealed it was actually
# still running).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# Unlike PYTHONUNBUFFERED, this one CAN be set from os.environ here, since
# PyTorch's CUDA allocator reads it lazily on first CUDA use, not at
# interpreter startup -- as long as this runs before any CUDA operation
# (it does; nothing above touches the GPU yet). Confirmed "not supported
# on this platform" on L40S in earlier testing; never tested on A100,
# where it may actually take effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint


class PackedDataset(Dataset):
    """Wraps a flat uint32 .bin of packed token blocks (from prepare_cpt_data.py)."""

    def __init__(self, bin_path: Path, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint32, mode="r")
        self.block_size = block_size
        self.n_blocks = len(self.data) // block_size

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        block = self.data[start:start + self.block_size].astype(np.int64)
        ids = torch.from_numpy(block)
        return {"input_ids": ids, "labels": ids.clone()}


def preprocess_logits_for_metrics(logits, labels):
    """Runs per eval batch, before Trainer accumulates results across the whole
    eval set. Keeping raw logits (batch, seq, ~150k vocab) in float32 across an
    entire eval pass would be many GB and likely OOM, so this reduces each
    batch down to just what compute_metrics actually needs: per-token
    negative log-likelihood (for loss/perplexity) and the top-5 predicted
    token ids (for accuracy) -- 7 numbers per position instead of ~150,000."""
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = logits[:, :-1, :]                    # position t predicts token t+1
    shifted_labels = labels[:, 1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)  # (batch, seq-1)
    top5 = torch.topk(logits, k=5, dim=-1).indices.float()                # (batch, seq-1, 5)
    return torch.cat([nll.unsqueeze(-1), top5], dim=-1)                   # (batch, seq-1, 6)


def make_compute_metrics(tokenizer):
    """Factory, not a bare function -- bits-per-byte needs the tokenizer to
    decode label ids back to text, so it can't be a plain module-level
    function the way the original loss/perplexity/accuracy metrics were."""
    special_ids = set(tokenizer.all_special_ids)

    def compute_metrics(eval_pred):
        """Consumes the reduced (nll, top5_ids) output of preprocess_logits_for_metrics.
        Deliberately does NOT return a 'loss' key -- Trainer already computes its own
        '{prefix}_loss' from the model's internal forward pass, and returning a
        second one here would just create a same-named key to collide with it."""
        preds, labels = eval_pred
        nll = preds[..., 0]                # (batch, seq-1), NATS (natural log)
        top5 = preds[..., 1:6]
        shifted_labels = labels[:, 1:]      # (batch, seq-1) -- same positions as nll/top5

        loss = nll.mean()
        top1_acc = (top5[..., 0] == shifted_labels).mean()
        top5_acc = (top5 == shifted_labels[..., None]).any(axis=-1).mean()

        # Bits-per-byte: normalizes loss to raw UTF-8 bytes of the underlying
        # text instead of per-token, which is what makes it comparable ACROSS
        # tokenizer configs with different vocabularies/fertility -- unlike
        # loss/perplexity/accuracy above, which aren't (see
        # analyze_tokeniser_comparison.py's docstring caveat). Special/
        # separator tokens (<|endoftext|> etc.) are excluded from BOTH the
        # bit sum and the byte count, so numerator and denominator cover
        # exactly the same underlying content. Needs no repacked data --
        # decoding is lossless/reversible (byte-level BPE, no Unicode
        # normalization), so this just uses the tokenizer already loaded
        # for training.
        is_special = np.isin(shifted_labels, list(special_ids))
        total_bits = float(nll[~is_special].sum() / np.log(2))   # nats -> bits

        texts = tokenizer.batch_decode(shifted_labels.tolist(), skip_special_tokens=True)
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        bits_per_byte = total_bits / total_bytes if total_bytes > 0 else float("nan")

        return {
            "perplexity": float(np.exp(loss)),
            "top1_accuracy": float(top1_acc),
            "top5_accuracy": float(top5_acc),
            "bits_per_byte": bits_per_byte,
        }

    return compute_metrics


class EvalCacheClearCallback(TrainerCallback):
    """Clears the CUDA memory cache once after each eval pass completes.

    Deliberately NOT called every training step -- torch.cuda.empty_cache()
    forces a CUDA sync and discards PyTorch's warmed memory cache, so the
    next allocation has to go back to the (slow) CUDA driver instead of
    reusing an already-available block. PyTorch's own docs warn against
    calling it routinely during training for exactly this reason -- doing
    so every step would be a real, measurable throughput cost, not a free
    safety net.

    Eval is different: it happens rarely (--eval_steps, roughly 9-10 eval
    EVENTS across a full run -- though each event may trigger this callback
    more than once internally, since eval_dataset is a dict evaluated per-
    domain; the exact count is uncertain from reading the code alone, which
    is why _n_calls below exists -- check the log for the real number)
    and represents a genuine memory-usage-
    pattern shift (full-vocab logits computation, different batch size --
    see make_compute_metrics), the same class of transition that caused
    the original eval-time OOM earlier in this project. A cache clear here
    costs one sync every few thousand steps, not every step.
    """

    def __init__(self):
        self._n_calls = 0

    def on_evaluate(self, args, state, control, **kwargs):
        self._n_calls += 1
        torch.cuda.empty_cache()
        print(f"[train] EvalCacheClearCallback: on_evaluate call #{self._n_calls} "
              f"(step {state.global_step})")
        return control


class UnfreezeCallback(TrainerCallback):
    """Unfreezes all non-embedding parameters once training reaches
    --freeze_non_embedding_steps -- the other half of the freeze block above,
    which only freezes and (until now) had no mechanism to undo it.

    This implements the tokenizer-adaptation method of de Vries & Nissim
    (2021): after swapping in a re-embedded model (new/averaged embedding
    rows sitting next to a pretrained transformer body that's never seen
    them), freeze everything except the embedding matrix for the first N
    steps so those rows can align before their initially large, essentially
    random-relative-to-the-rest-of-the-model gradients are allowed to touch
    the pretrained body -- avoiding the loss-spike/catastrophic-forgetting
    failure mode that motivates this whole stage.
    """

    def __init__(self, model, unfreeze_at_step: int):
        self.model = model
        self.unfreeze_at_step = unfreeze_at_step
        self._unfrozen = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._unfrozen and state.global_step >= self.unfreeze_at_step:
            n = 0
            for name, param in self.model.named_parameters():
                if "embed_tokens" not in name and "lm_head" not in name:
                    param.requires_grad_(True)
                    n += 1
            self._unfrozen = True
            # Unfreezing suddenly needs gradient (and soon, once AdamW's
            # lazily-allocated momentum/variance buffers kick in on their
            # first step) optimizer storage for ~1.3B previously-frozen
            # parameters -- a real memory spike. Gradient checkpointing is
            # now on from the very start of training (see main()), so it's
            # already absorbing most of this; the cache clear is cheap,
            # low-frequency insurance for what's left.
            torch.cuda.empty_cache()
            print(f"[train] step {state.global_step}: unfroze {n} non-embedding "
                  f"parameter tensors")
        return control


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packed_dir", type=Path, required=True)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2,
                    help="separate from --per_device_train_batch_size on purpose. Eval's "
                         "preprocess_logits_for_metrics upcasts the full (batch, seq, vocab) "
                         "logits to fp32 to compute log-softmax -- with a ~150K vocab this is "
                         "far more memory-hungry per example than training. Without this flag "
                         "eval silently used HF's default of 8, which combined with "
                         "still-resident optimizer state from training (AdamW: ~2x params in "
                         "fp32) caused a real CUDA OOM during a smoke test. Lower further "
                         "(e.g. 1) if OOMs persist; the eval batch size doesn't need to match "
                         "training's for correctness, only for consistency of eval timing.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=2.0)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--attn_implementation", default="sdpa",
                    help="'sdpa' works out of the box; 'flash_attention_2' is faster "
                         "and supports document masking, but must be installed/compiled separately")
    p.add_argument("--domains", nargs="+",
                    default=["irish", "english", "math", "code", "instruct"])
    p.add_argument("--seed", type=int, default=1,
                    help="training seed -- also the seed used for the run's output dir "
                         "naming convention (runs/<tokenizer>/seed<N>/), for later "
                         "compatibility with a multi-seed/multi-tokenizer comparison")
    p.add_argument("--freeze_non_embedding_steps", type=int, default=0,
                    help="if >0, only train embed_tokens/lm_head for this many steps before "
                         "unfreezing the rest of the model -- use this for the custom-tokenizer "
                         "variant, where embeddings are freshly (re)initialized and need to settle "
                         "before their gradients get to touch the pretrained transformer body. "
                         "300 (~1.5% of total steps) is what both real job scripts use, following "
                         "Dobler and de Melo (2024)'s proportional approach for a comparably "
                         "constrained compute budget -- de Vries & Nissim (2021) themselves don't "
                         "specify a duration. Unfreezing is handled by UnfreezeCallback (see above); "
                         "leave at 0 for the baseline-tokenizer run, which has no re-embedded "
                         "rows that need this.")
    p.add_argument("--resume_from_checkpoint", default=None,
                    help="path to a specific checkpoint dir to resume from, or the literal "
                         "'auto' to resume from the latest checkpoint found under --output_dir "
                         "if one exists. Safe to pass 'auto' on every submission -- it's a "
                         "no-op (trains from scratch) on a fresh/empty output_dir, and picks "
                         "up where a previous, e.g. walltime-killed, run of this SAME "
                         "--output_dir left off otherwise.")
    p.add_argument("--save_strategy", default="epoch", choices=["steps", "epoch"],
                    help="'epoch' (default) saves one checkpoint per epoch -- matches the "
                         "original behavior, but means --resume_from_checkpoint=auto has "
                         "nothing to pick up if the job is killed before a full epoch "
                         "finishes. 'steps' saves every --save_steps instead, which bounds "
                         "how much progress a mid-epoch kill can lose.")
    p.add_argument("--save_steps", type=int, default=500,
                    help="only used when --save_strategy=steps")
    p.add_argument("--save_total_limit", type=int, default=3,
                    help="max checkpoints kept on disk (oldest deleted first). With "
                         "--save_strategy=steps this stops checkpoints from filling scratch; "
                         "with --save_strategy=epoch you can set this higher (e.g. to your "
                         "--num_train_epochs) to keep one per epoch as before.")
    p.add_argument("--max_steps", type=int, default=-1,
                    help="if >0, overrides --num_train_epochs and stops after this many "
                         "optimizer steps regardless of dataset size -- mainly useful for "
                         "smoke tests against a tiny dataset. Leave at -1 for real runs.")
    p.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="dtype to load model weights in. Default bfloat16 matches the real "
                         "GPU run; use float32 for a CPU-only smoke test -- unaccelerated "
                         "bf16 on CPU is very slow and can be numerically flaky.")
    p.add_argument("--no_bf16_training", action="store_true",
                    help="disable TrainingArguments(bf16=True). Set this for CPU-only smoke "
                         "tests -- CPU bf16 mixed precision isn't accelerated the way it is "
                         "on the A100s. Leave unset for the real GPU run.")
    p.add_argument("--dataloader_num_workers", type=int, default=4,
                    help="lower to 0 for local/Windows smoke tests to avoid multiprocessing "
                         "worker overhead/quirks on a small dataset; 4 is fine on Hypatia.")
    p.add_argument("--logging_steps", type=int, default=10,
                    help="console/wandb log frequency in steps. Default 10 matches the real "
                         "run; set to 1 for smoke tests (esp. with a small --max_steps) so you "
                         "get a print every step instead of one line at the very end.")
    p.add_argument("--eval_max_blocks", type=int, default=None,
                    help="if set, caps EVERY eval dataset (combined + each domain) to at most "
                         "this many blocks. --max_steps only limits training -- eval always "
                         "runs over the FULL valid/test .bin regardless, which is fine for the "
                         "real run but means a smoke test against real (non-tiny) packed data "
                         "can take hours/days on eval alone. Leave unset for the real run.")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                    help="disable gradient checkpointing. Checkpointing trades ~25% extra "
                         "compute (recomputing forward activations during backward) for lower "
                         "memory use. Testing showed this IS needed even on 80GB A100s: "
                         "unfreezing (see UnfreezeCallback) suddenly needs gradient/optimizer "
                         "storage for ~1.3B previously-frozen parameters, which OOM'd on L40S "
                         "without checkpointing on. Passing this flag no longer fully disables "
                         "checkpointing for a real full run (max_steps==-1) -- see the override "
                         "in main() below -- only for smoke/preview tests, which set --max_steps "
                         "explicitly and are unaffected.")
    args = p.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    try:
        # newer transformers: torch_dtype= was renamed to dtype=
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=dtype_map[args.torch_dtype], attn_implementation=args.attn_implementation,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype_map[args.torch_dtype], attn_implementation=args.attn_implementation,
        )
    # max_steps==-1 (its default) only happens on a real full run driven by
    # --num_train_epochs -- every smoke/preview test in this project always
    # sets --max_steps explicitly. train_baseline_cpt.slurm/
    # train_custom_cpt.slurm are already submitted and can't be resubmitted
    # to drop --no_gradient_checkpointing, so this forces checkpointing on
    # for exactly the runs that need it (avoiding the unfreeze-transition
    # OOM found in testing) without changing behaviour for any test script,
    # which will still respect the flag normally.
    is_full_run = args.max_steps == -1
    if is_full_run and args.no_gradient_checkpointing:
        print("[train] overriding --no_gradient_checkpointing: forcing gradient "
              "checkpointing ON for this full run to avoid the unfreeze-transition "
              "OOM found in earlier testing (see chat)")
    if not args.no_gradient_checkpointing or is_full_run:
        model.gradient_checkpointing_enable()

    unfreeze_callback = None
    if args.freeze_non_embedding_steps > 0:
        for name, param in model.named_parameters():
            if "embed_tokens" not in name and "lm_head" not in name:
                param.requires_grad_(False)
        print(f"[train] froze all non-embedding params for the first "
              f"{args.freeze_non_embedding_steps} steps")
        unfreeze_callback = UnfreezeCallback(model, args.freeze_non_embedding_steps)

    train_ds = PackedDataset(args.packed_dir / "train.bin", args.block_size)

    eval_ds = {"combined": PackedDataset(args.packed_dir / "valid.bin", args.block_size)}
    for domain in args.domains:
        path = args.packed_dir / f"{domain}_valid.bin"
        if path.exists():
            eval_ds[domain] = PackedDataset(path, args.block_size)

    if args.eval_max_blocks is not None:
        for name, ds in eval_ds.items():
            n = min(len(ds), args.eval_max_blocks)
            print(f"[train] --eval_max_blocks: capping '{name}' eval set to {n} blocks "
                  f"(was {len(ds)})")
            eval_ds[name] = Subset(ds, range(n))

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,      # -1 = disabled, defers to num_train_epochs (HF default)
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=not args.no_bf16_training,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,               # ignored by Trainer when save_strategy="epoch"
        save_total_limit=args.save_total_limit,
        report_to=["wandb"],  # set WANDB_MODE=offline if compute nodes lack internet, sync later
        dataloader_num_workers=args.dataloader_num_workers,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=make_compute_metrics(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    # Passing the tokenizer makes every checkpoint -- not just final/ -- save it
    # alongside the model, so evaluate_test_set.py can load any epoch/step
    # checkpoint directly. The kwarg was renamed tokenizer -> processing_class in
    # newer transformers; this picks whichever the installed version expects.
    tokenizer_kwarg = ("processing_class"
                        if "processing_class" in inspect.signature(Trainer.__init__).parameters
                        else "tokenizer")
    trainer_kwargs[tokenizer_kwarg] = tokenizer
    callbacks = [EvalCacheClearCallback()]
    if unfreeze_callback is not None:
        callbacks.append(unfreeze_callback)
    trainer_kwargs["callbacks"] = callbacks

    trainer = Trainer(**trainer_kwargs)

    resume = args.resume_from_checkpoint
    if resume == "auto":
        # Guard against output_dir not existing yet on a fresh run --
        # get_last_checkpoint() calls os.listdir() internally and throws
        # FileNotFoundError otherwise. Checking .is_dir() first avoids a
        # race condition where the main process's Trainer(...) init has
        # created the directory but other ranks reach this line before
        # that's visible to them (hit in practice: 3 of 4 ranks crashed
        # here on a completely fresh output_dir, job 4334354).
        resume = get_last_checkpoint(str(args.output_dir)) if args.output_dir.is_dir() else None
        if resume:
            print(f"[train] --resume_from_checkpoint=auto: resuming from {resume}")
        else:
            print(f"[train] --resume_from_checkpoint=auto: no checkpoint found under "
                  f"{args.output_dir}, starting fresh")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))

    # one clean final validation pass (periodic evals during training already covered
    # this, but a single final_eval_metrics.json is what analyze_tokeniser_comparison.py
    # expects to find in each run dir by default)
    final_metrics = trainer.evaluate()
    with open(args.output_dir / "final_eval_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[train] wrote {args.output_dir / 'final_eval_metrics.json'}")
    print("[train] run evaluate_test_set.py separately when you're ready for a held-out "
          "test number -- it isn't produced automatically here.")


if __name__ == "__main__":
    main()