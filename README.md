# Irish-LLM

Continued pretraining and tokenizer adaptation of Qwen2.5-1.5B for Irish, comparing two strategies: retaining the original tokenizer versus replacing it with a custom, from-scratch Irish/English BPE vocabulary. Developed as part of an MSc dissertation at the University of St Andrews.

## Overview

This repository contains the training, tokenizer-construction, and evaluation pipeline used to:
- Continue-pretrain Qwen2.5-1.5B on a bilingual Irish/English corpus, using either the model's original ~152K-token vocabulary or a custom, from-scratch 32K-token vocabulary
- Measure tokenizer efficiency (fertility, characters-per-token) across both approaches
- Evaluate resulting models on Irish-language benchmarks (Cloze-Ga, SIB-Ga, IrishQA) alongside general-capability benchmarks (BLEU translation, Natural Questions)

## Repository structure

- `train_cpt.py` — continued pretraining
- `train_tokenizer.py` — custom BPE tokenizer construction
- `reembed_tokenizer.py` — embedding initialization for the custom vocabulary
- `prepare_cpt_data.py` — corpus preparation and packing
- `measure_tokenizer.py` — tokenizer efficiency metrics (fertility, characters/token)
- `evaluate_test_set.py` — held-out perplexity, bits-per-byte, and accuracy evaluation
- `generate.py` — model inference/generation utilities
- `inspect_bins.py` — packed training data inspection
- `benchmarks/` — benchmark datasets and evaluation harnesses
- `load_data/` — data loading utilities

## Requirements

See `requirements.txt` *(not yet present — see note below)*.

## Usage

*(Add specific commands for each pipeline stage — tokenizer training, CPT, evaluation.)*
