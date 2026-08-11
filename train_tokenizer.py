#!/usr/bin/env python3
"""
Trains a byte-level BPE tokeniser (same algorithm family as Qwen2.5's own
tokeniser, unlike the existing SentencePiece-based train_tokeniser.py) on the
EXACT weighted training mixture prepare_cpt_data.py packs into train.bin --
same domains, same TRAIN_REPEATS upsampling (Irish 4x), read from the same
output/splits/{domain}_train.txt files that stage_split() produces.

Run `python prepare_cpt_data.py split` first if output/splits/ doesn't exist
yet -- this script deliberately does NOT re-implement the split/repeat logic,
it imports DOMAINS/TRAIN_REPEATS/iter_documents directly from
prepare_cpt_data.py so the tokeniser's training mixture can never silently
drift out of sync with what the model actually trains on.

Byte-level BPE (the GPT-2/Qwen family, as opposed to SentencePiece Unigram)
means every possible input string is representable without an <unk> token --
raw bytes are the fallback for anything not covered by a learned merge. That
matters for a tokeniser meant to sit inside a model that must never emit an
"unknown" token during generation, and keeps the algorithm family consistent
with Qwen2.5's own tokeniser for the later re-embedding step.

Usage:
    python train_tokeniser_bpe.py
"""
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from prepare_cpt_data import DOMAINS, SPLIT_DIR, TRAIN_REPEATS

# --------------------------------------------------------------------------
# Config -- edit these to match your setup (mirrors prepare_cpt_data.py's style)
# --------------------------------------------------------------------------

VOCAB_SIZE = 32_000   # UCCIX and similar bilingual low-resource-language LLM
                       # projects typically land in the 32K-50K range -- Qwen's
                       # ~152K is sized for broad multilingual coverage this
                       # project doesn't need, and every extra vocab slot is
                       # another embedding row that starts poorly-initialised
                       # and needs training to become useful.

# Qwen2.5's own special tokens this project actually needs: <|endoftext|> is
# the CPT document separator/eos (hardcoded throughout prepare_cpt_data.py
# and train_cpt.py); <|im_start|>/<|im_end|> are needed for the planned
# instruction-tuning stage's chat template. Add more of Qwen's special tokens
# only if something downstream actually needs them -- an unused special
# token is just a dead vocab slot.
SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

OUT_DIR = Path("tokenizer")


def weighted_file_list():
    """Builds the file list tokenizer.train() reads directly with native Rust
    file I/O -- repeating each domain's path TRAIN_REPEATS[domain] times
    achieves the same upsampling weighting as before, without needing to
    materialize the corpus through a slow, single-threaded Python generator
    first (train_from_iterator() with a Python iterator pays that cost;
    tokenizer.train(files=...) doesn't).

    Trade-off worth knowing: this splits each file by LINE, not by the
    <|endoftext|> document boundary iter_documents() respects. For a
    byte-level BPE trainer this shouldn't meaningfully change which merges
    get learned -- it cares about local subword co-occurrence within
    pre-tokenized "words", not document-level context -- but it's a real
    behavioral difference from the corpus measure_tokeniser_bpe.py reads."""
    files = []
    for domain in DOMAINS:
        path = SPLIT_DIR / f"{domain}_train.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run `python prepare_cpt_data.py split` first."
            )
        repeat = TRAIN_REPEATS.get(domain, 1)
        files.extend([str(path)] * repeat)
    return files


def main():
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        # Ensures all 256 raw byte values are base vocab entries, matching
        # Qwen/GPT-2-style byte-level BPE and guaranteeing byte-fallback
        # coverage regardless of what characters show up (Irish fada
        # diacritics included) -- see module docstring.
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        show_progress=True,
    )

    print(f"[train_tokeniser_bpe] training BPE, target vocab={VOCAB_SIZE:,} ...")
    tokenizer.train(files=weighted_file_list(), trainer=trainer)

    # Wrap as a HF-loadable tokenizer directory (tokenizer.json,
    # tokenizer_config.json, special_tokens_map.json, ...) -- same shape as
    # qwen2.5-1.5b/, so train_cpt.py's --model_name can point straight at
    # whatever re-embedded model directory you build to pair with this.
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=None,
        eos_token="<|endoftext|>",
        # Matches the planned SFT stage's design ("pre-training document
        # separator acting as the pad token") -- keep that consistent here
        # rather than picking a different pad token later.
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(str(OUT_DIR))
    print(f"[train_tokeniser_bpe] wrote {OUT_DIR} (vocab_size={fast_tokenizer.vocab_size:,})")


if __name__ == "__main__":
    main()
