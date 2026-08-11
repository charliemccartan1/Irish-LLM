"""
Evaluates Cloze-gle via whole-sentence log-likelihood scoring, zero-shot,
matching Qomhra's confirmed methodology exactly: candidates differ by a
single pronoun (e used masculine, i feminine, iad plural); the sentence
with the highest raw (unnormalized) LL is the model's prediction.

Data format TBD -- build_candidates() below is the one function that will
need adjusting once the real multichoice_cloze.json structure is known.
Everything else (scoring, selection) is generic and already verified
against Qomhra's own Table 4 example (see test below).

Usage:
    python evaluate_cloze.py --model_path output/runs/original/seed1/final --model_tag baseline
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def load_model_and_tokenizer(model_path, adapter_path=None, tokenizer_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return tokenizer, model


def sentence_nll(sentence, tokenizer, model, device):
    """Raw, UNNORMALIZED total negative log-likelihood of the full
    sentence -- matches Table 4 exactly (candidates of equal token
    length compare directly; no per-token averaging)."""
    ids = tokenizer(sentence, return_tensors="pt")["input_ids"].to(device)
    n_tokens = ids.shape[1] - 1
    if n_tokens <= 0:
        return 0.0
    with torch.no_grad():
        out = model(ids, labels=ids)
    return out.loss.item() * n_tokens


def predict(candidates, tokenizer, model, device):
    """candidates: list of full candidate sentence strings. Returns the
    index of the one with the highest (least negative) LL, i.e. -NLL."""
    scores = [-sentence_nll(c, tokenizer, model, device) for c in candidates]
    return scores.index(max(scores)), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter for POST-sft. Omit for PRE-sft.")
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(SCRIPT_DIR / "cloze_test.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.limit:
        items = items[: args.limit]

    stage = "POST-SFT" if args.adapter_path else "PRE-SFT"
    tag = f"{args.model_tag}_{'post' if args.adapter_path else 'pre'}sft"

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.adapter_path, args.tokenizer_name)
    device = next(model.parameters()).device

    n_correct = 0
    for item in items:
        predicted_index, scores = predict(item["candidates"], tokenizer, model, device)
        n_correct += predicted_index == item["correct_index"]

    accuracy = n_correct / len(items)
    print(f"[{args.model_tag}, {stage}, zero-shot] accuracy: {n_correct}/{len(items)} = {accuracy:.4f}")

    out_path = SCRIPT_DIR / f"cloze_{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump({"model_tag": args.model_tag, "stage": stage,
                   "accuracy": accuracy, "n_correct": n_correct, "n_total": len(items)}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
