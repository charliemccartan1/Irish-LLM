#!/usr/bin/env python3
"""
Re-embeds a base model's (tied) embedding matrix for a new, differently-sized
tokenizer vocabulary -- Stage 2 of the tokenizer-swap pipeline, following the
adaptation method of de Vries & Nissim (2021), the same method the
NorMistral/Norwegian-languages paper (cross-referenced earlier in this
project) uses for their own tokenizer swap.

Method:
  - For every token in the NEW vocabulary whose exact string form also
    exists in the OLD vocabulary: copy that token's original embedding
    directly.
  - For every genuinely new token: decompose its string into OLD-vocabulary
    sub-tokens (re-tokenizing with the OLD tokenizer) and initialize its
    embedding as the AVERAGE of those sub-tokens' original embeddings.

Only ONE embedding matrix needs touching, not two -- Qwen2.5-1.5B ties its
input embeddings and lm_head (confirmed earlier via the resume-test's
"missing keys: lm_head.weight" message and matching post-resume perplexity),
so there's no separate output layer to align.

A subtlety worth knowing about, specific to byte-level BPE: vocabulary
entries aren't real text -- they're byte-level-mapped strings (e.g. a
leading space renders as "\u0120", not " "). Matching on those raw strings
directly is fine (both tokenizers share the same fixed byte-to-unicode
table), but DECOMPOSING with the old tokenizer needs the token converted
back to real text first via convert_tokens_to_string() -- feeding the raw
byte-mapped string straight into old_tokenizer(...) would double-apply the
byte mapping and silently produce garbage. This script does that conversion;
it's the one place in this whole process that's easy to get wrong.

Usage:
    python reembed_tokenizer.py \\
        --base_model qwen2.5-1.5b \\
        --new_tokenizer tokenizer \\
        --out_dir qwen2.5-1.5b-custom
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):   # tqdm is a nice-to-have, not a hard dependency
        return iterable


def build_new_embedding_matrix(old_embeddings, old_tokenizer, new_tokenizer):
    """Returns a [len(new_tokenizer), hidden_size] tensor: exact-string
    matches copied directly from old_embeddings, everything else
    initialized as the mean of its de Vries & Nissim decomposition under
    the OLD tokenizer."""
    hidden_size = old_embeddings.shape[1]
    new_vocab_size = len(new_tokenizer)   # NOT .vocab_size -- that can exclude
                                            # added/special tokens depending on
                                            # tokenizer type; len() is the true
                                            # total, matching what
                                            # resize_token_embeddings() needs.
    new_embeddings = torch.empty(new_vocab_size, hidden_size, dtype=old_embeddings.dtype)

    old_vocab = old_tokenizer.get_vocab()   # raw vocab string -> id
    new_vocab = new_tokenizer.get_vocab()

    n_direct = 0
    n_averaged = 0
    n_fallback = 0

    for token_str, new_id in tqdm(new_vocab.items(), desc="re-embedding", total=len(new_vocab)):
        old_id = old_vocab.get(token_str)
        if old_id is not None:
            # Exact byte-level string match -- same fixed byte-to-unicode
            # table on both sides, so this is a safe direct comparison
            # without needing to decode anything.
            new_embeddings[new_id] = old_embeddings[old_id]
            n_direct += 1
            continue

        # Not present verbatim in the old vocab. Convert the byte-level
        # vocab string back into real text (see module docstring for why
        # this step matters), then let the OLD tokenizer decompose that
        # text through its own full pipeline.
        decoded_text = new_tokenizer.convert_tokens_to_string([token_str])
        sub_ids = old_tokenizer(decoded_text, add_special_tokens=False)["input_ids"]

        if sub_ids:
            new_embeddings[new_id] = old_embeddings[sub_ids].mean(dim=0)
            n_averaged += 1
        else:
            # Only reachable for a token whose decoded text is empty or
            # otherwise unencodable -- shouldn't happen in practice, but
            # fall back to the old tokenizer's own unk/pad embedding rather
            # than leave a row of uninitialized memory.
            fallback_id = old_tokenizer.unk_token_id
            if fallback_id is None:
                fallback_id = old_tokenizer.pad_token_id or 0
            new_embeddings[new_id] = old_embeddings[fallback_id]
            n_fallback += 1

    print(f"[reembed] {n_direct:,} direct copies, {n_averaged:,} averaged from "
          f"sub-tokens, {n_fallback:,} fallback (should be 0, or very close to it)")
    return new_embeddings


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True, help="path to the original model, e.g. qwen2.5-1.5b")
    p.add_argument("--new_tokenizer", required=True, help="path to the new tokenizer dir, e.g. tokenizer")
    p.add_argument("--out_dir", required=True, help="where to write the re-embedded model + tokenizer")
    args = p.parse_args()

    print(f"[reembed] loading base model + its ORIGINAL tokenizer from {args.base_model}")
    old_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    print(f"[reembed] loading new tokenizer from {args.new_tokenizer}")
    new_tokenizer = AutoTokenizer.from_pretrained(args.new_tokenizer)

    old_embeddings = model.get_input_embeddings().weight.detach().clone()
    print(f"[reembed] old embedding matrix: {tuple(old_embeddings.shape)}")

    new_embeddings = build_new_embedding_matrix(old_embeddings, old_tokenizer, new_tokenizer)
    print(f"[reembed] new embedding matrix: {tuple(new_embeddings.shape)}")

    # Resize first -- for Qwen2.5-1.5B (tie_word_embeddings=True) this also
    # handles lm_head automatically, since it's the same tensor. THEN
    # overwrite the freshly (randomly) initialized rows resize_token_embeddings
    # creates with our actually-computed values.
    model.resize_token_embeddings(len(new_tokenizer))
    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(new_embeddings)

    # Keep the model config's special-token ids in sync with the new
    # tokenizer -- otherwise eos/pad/bos handling would still point at OLD
    # vocab ids that mean something different (or are flat-out out of range)
    # under the new one. Deliberately sets None too, not just real values --
    # leaving a stale old-vocab id in place (e.g. bos_token_id from a
    # tokenizer that never defined a bos token) is exactly the bug this is
    # here to prevent.
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        new_val = getattr(new_tokenizer, attr, None)
        setattr(model.config, attr, new_val)
        setattr(model.generation_config, attr, new_val)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    new_tokenizer.save_pretrained(str(out_dir))
    print(f"[reembed] wrote re-embedded model + tokenizer to {out_dir}")
    print(f"[reembed] point train_cpt.py --model_name at {out_dir} for the custom-tokenizer run, "
          f"with --freeze_non_embedding_steps set (1000 is a reasonable starting point).")


if __name__ == "__main__":
    main()
