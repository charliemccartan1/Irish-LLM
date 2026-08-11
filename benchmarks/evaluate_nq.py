"""
Evaluates NQ-eng via open-ended generation (5-shot), scored by exact
match against a list of acceptable answers -- matching Qomhra's own
described methodology and Table 8 example exactly. Self-contained, no
separate model_loading.py needed.

Usage:
    python evaluate_nq.py --model_path output/runs/original/seed1/final --model_tag baseline
    python evaluate_nq.py --model_path output/runs/original/seed1/final --model_tag baseline --adapter_path output/runs/sft_baseline/final
    python evaluate_nq.py --model_path output/runs/custom/seed1/final --model_tag custom
    python evaluate_nq.py --model_path output/runs/custom/seed1/final --model_tag custom --adapter_path output/runs/sft_custom/final
"""
import argparse
import json
import re
import string
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent


def load_model_and_tokenizer(model_path, adapter_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return tokenizer, model


def normalize(text):
    """Standard SQuAD/NQ-style EM normalization: lowercase, strip
    punctuation, remove articles, collapse whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(response, references):
    norm_response = normalize(response)
    return any(norm_response == normalize(ref) for ref in references)


def build_prefix(fewshot_examples):
    blocks = [f"Question: {ex['question']}\nAnswer: {ex['answer'][0]}" for ex in fewshot_examples]
    return "\n\n".join(blocks) + "\n\n"


def generate_answer(prefix, question, tokenizer, model, device):
    prompt = prefix + f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=32, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # NOTE: repetition_penalty/no_repeat_ngram_size deliberately
            # NOT used -- same reasoning as evaluate_bleu.py. Qomhra's own
            # paper confirms NQ suffers this identical rambling problem
            # and reports it unmitigated; adding this here would give an
            # advantage their reported methodology didn't have.
        )
    raw_generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    # Cut at the FIRST occurrence of the model regenerating a fake new
    # turn (confirmed real pattern from actual output: "\n\nQuestion:" or
    # "\nAnswer:"), not at any bare newline -- a bare-newline cut would
    # silently truncate a genuinely multi-line correct answer with no way
    # to tell the difference from rambling. This only cuts when the
    # specific leakage pattern is actually present.
    processed = raw_generated
    for marker in ["\nQuestion:", "\nAnswer:"]:
        idx = processed.find(marker)
        if idx != -1:
            processed = processed[:idx]
    processed = processed.strip()
    return processed, raw_generated


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter for POST-sft. Omit for PRE-sft.")
    parser.add_argument("--model_tag", required=True, help="e.g. 'baseline' or 'custom' -- keeps output filenames distinct")
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="cap the number of eval questions, for a quick check")
    args = parser.parse_args()

    with open(SCRIPT_DIR / "nq_fewshot.json", encoding="utf-8") as f:
        fewshot_examples = json.load(f)[: args.n_shot]
    prefix = build_prefix(fewshot_examples)

    with open(SCRIPT_DIR / "nq_eng.jsonl", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]
    if args.limit:
        questions = questions[: args.limit]

    stage = "POST-SFT" if args.adapter_path else "PRE-SFT"
    tag = f"{args.model_tag}_{'post' if args.adapter_path else 'pre'}sft"

    tokenizer, model = load_model_and_tokenizer(args.model_path, args.adapter_path)
    device = next(model.parameters()).device

    n_correct = 0
    samples = []
    for q in questions:
        response, raw_response = generate_answer(prefix, q["question"], tokenizer, model, device)
        is_correct = exact_match(response, q["answer"])
        n_correct += is_correct
        if len(samples) < 10:
            samples.append({"question": q["question"], "response": response, "raw_response": raw_response,
                             "references": q["answer"], "correct": is_correct})

    accuracy = n_correct / len(questions)
    print(f"[{args.model_tag}, {stage}, {args.n_shot}-shot] exact match: {n_correct}/{len(questions)} = {accuracy:.4f}")

    out_path = SCRIPT_DIR / f"nq_{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump({"model_tag": args.model_tag, "stage": stage, "n_shot": args.n_shot,
                   "exact_match": accuracy, "n_correct": n_correct, "n_total": len(questions),
                   "samples": samples}, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
