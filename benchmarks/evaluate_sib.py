"""
Evaluates SIB-Ga via label-continuation likelihood scoring (10-shot),
matching Qomhra's confirmed methodology exactly: "The model scores each
topic label as a continuation; the label with the highest probability is
selected." Same core mechanism as evaluate_iqa.py, verified against
Qomhra's own Table 6 example there -- adapted here for 7 fixed topic
labels instead of 4 per-question choices.

Usage:
    python evaluate_sib.py --model_path output/runs/original/seed1/final --model_tag baseline
    python evaluate_sib.py --model_path output/runs/original/seed1/final --adapter_path output/runs/sft_baseline/final --model_tag baseline
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


def build_text_prompt(text, label=None):
    block = f"Téacs: {text}\nÁbhar:"
    if label is not None:
        block += f" {label}"
    return block


def build_prefix(fewshot_examples, label_names):
    blocks = [build_text_prompt(ex["text"], label_names[ex["label_index"]]) for ex in fewshot_examples]
    return "\n\n".join(blocks) + "\n\n"


def prompt_nll(prefix, text, tokenizer, model, device):
    prompt_ids = tokenizer(prefix + build_text_prompt(text), return_tensors="pt")["input_ids"].to(device)
    n_prompt = prompt_ids.shape[1] - 1
    if n_prompt == 0:
        return 0.0
    with torch.no_grad():
        prompt_out = model(prompt_ids, labels=prompt_ids)
    return prompt_out.loss.item() * n_prompt


def label_logprob(prefix, text, label, tokenizer, model, device, total_nll_prompt):
    full_ids = tokenizer(prefix + build_text_prompt(text, label), return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        full_out = model(full_ids, labels=full_ids)
    n_full = full_ids.shape[1] - 1
    total_nll_full = full_out.loss.item() * n_full
    return -(total_nll_full - total_nll_prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter for POST-sft. Omit for PRE-sft.")
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--n_shot", type=int, default=10, help="Qomhra's paper confirms SIB-Ga is 10-shot")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(SCRIPT_DIR / "sib_labels.json", encoding="utf-8") as f:
        label_names = json.load(f)
    with open(SCRIPT_DIR / "sib_fewshot.json", encoding="utf-8") as f:
        fewshot_data = json.load(f)
        assert fewshot_data["labels"] == label_names, "label order mismatch between sib_labels.json and sib_fewshot.json"
        fewshot_examples = fewshot_data["examples"][: args.n_shot]
    with open(SCRIPT_DIR / "sib_test.jsonl", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    if args.limit:
        items = items[: args.limit]

    stage = "POST-SFT" if args.adapter_path else "PRE-SFT"
    tag = f"{args.model_tag}_{'post' if args.adapter_path else 'pre'}sft"

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.adapter_path, args.tokenizer_name)
    device = next(model.parameters()).device
    prefix = build_prefix(fewshot_examples, label_names)

    n_correct = 0
    for item in items:
        total_nll_prompt = prompt_nll(prefix, item["text"], tokenizer, model, device)
        scores = [label_logprob(prefix, item["text"], label, tokenizer, model, device, total_nll_prompt)
                  for label in label_names]
        n_correct += scores.index(max(scores)) == item["label_index"]

    accuracy = n_correct / len(items)
    print(f"[{args.model_tag}, {stage}, {args.n_shot}-shot] accuracy: {n_correct}/{len(items)} = {accuracy:.4f}")

    out_path = SCRIPT_DIR / f"sib_{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump({"model_tag": args.model_tag, "stage": stage, "n_shot": args.n_shot,
                   "accuracy": accuracy, "n_correct": n_correct, "n_total": len(items)}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
