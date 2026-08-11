"""
Evaluates IrishQA (en/ga) via likelihood scoring, for any of the 8
baseline/custom x pre/post-SFT x en/ga combinations, via CLI flags.
Self-contained -- no separate model_loading.py needed, works if uploaded
on its own alongside prepare_iqa.py's output.

Examples (see run_all_iqa.sh for all 8 at once):
    python evaluate_iqa.py --lang en --model_path output/runs/original/seed1/final
    python evaluate_iqa.py --lang en --model_path output/runs/original/seed1/final --adapter_path output/runs/sft_baseline/final
    python evaluate_iqa.py --lang ga --model_path output/runs/custom/seed1/final
    python evaluate_iqa.py --lang ga --model_path output/runs/custom/seed1/final --adapter_path output/runs/sft_custom/final
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


def build_question_prompt(context, question, answer_text=None):
    """No choices listed here at all -- matches the paper's description
    precisely: each candidate answer's likelihood is scored as a direct
    continuation of just the question+context, never shown alongside the
    other three options."""
    block = f"{context}\n\nQuestion: {question}\nAnswer:"
    if answer_text is not None:
        block += f" {answer_text}"
    return block


def build_prefix(fewshot_examples):
    if not fewshot_examples:
        return ""
    blocks = []
    for ex in fewshot_examples:
        correct_answer = ex["choices"][ex["answer_index"]]
        blocks.append(build_question_prompt(ex["context"], ex["question"], answer_text=correct_answer))
    return "\n\n".join(blocks) + "\n\n"


def prompt_nll(prefix, context, question, tokenizer, model, device):
    prompt_ids = tokenizer(prefix + build_question_prompt(context, question), return_tensors="pt")["input_ids"].to(device)
    n_prompt = prompt_ids.shape[1] - 1
    if n_prompt == 0:
        return 0.0
    with torch.no_grad():
        prompt_out = model(prompt_ids, labels=prompt_ids)
    return prompt_out.loss.item() * n_prompt


def choice_logprob(prefix, context, question, answer_text, tokenizer, model, device, total_nll_prompt):
    full_ids = tokenizer(prefix + build_question_prompt(context, question, answer_text),
                          return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        full_out = model(full_ids, labels=full_ids)
    n_full = full_ids.shape[1] - 1
    total_nll_full = full_out.loss.item() * n_full
    return -(total_nll_full - total_nll_prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=["en", "ga"], required=True)
    parser.add_argument("--model_path", required=True, help="baseline or custom CPT checkpoint")
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter for POST-sft. Omit for PRE-sft.")
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--model_tag", required=True, help="e.g. 'baseline' or 'custom' -- keeps output filenames distinct")
    parser.add_argument("--n_shot", type=int, default=5, help="Qomhra's paper confirms IQA is 5-shot")
    args = parser.parse_args()

    data_path = SCRIPT_DIR / f"iqa_{args.lang}.jsonl"
    fewshot_path = SCRIPT_DIR / f"iqa_{args.lang}_fewshot.json"
    with open(fewshot_path, encoding="utf-8") as f:
        fewshot_examples = json.load(f)[: args.n_shot]

    stage = "POST-SFT" if args.adapter_path else "PRE-SFT"
    tag = f"{args.model_tag}_{args.lang}_{'post' if args.adapter_path else 'pre'}sft"
    out_path = SCRIPT_DIR / f"iqa_{tag}_results.json"

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.adapter_path, args.tokenizer_name)
    device = next(model.parameters()).device
    prefix = build_prefix(fewshot_examples)

    with open(data_path, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    n_correct = 0
    for q in questions:
        context, question, choices = q.get("context", ""), q["question"], q["choices"]
        total_nll_prompt = prompt_nll(prefix, context, question, tokenizer, model, device)
        scores = [choice_logprob(prefix, context, question, choice_text, tokenizer, model, device, total_nll_prompt)
                  for choice_text in choices]
        n_correct += scores.index(max(scores)) == q["answer_index"]

    accuracy = n_correct / len(questions)
    print(f"[{args.model_tag}, {args.lang}, {stage}, {args.n_shot}-shot] accuracy: {n_correct}/{len(questions)} = {accuracy:.4f}")

    with open(out_path, "w") as f:
        json.dump({"model_tag": args.model_tag, "lang": args.lang, "stage": stage, "n_shot": args.n_shot,
                   "accuracy": accuracy, "n_correct": n_correct, "n_total": len(questions)}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
