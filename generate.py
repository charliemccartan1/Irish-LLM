"""Prompt a trained model from this project and see what it generates.

This is plain text completion (the base Qwen2.5-1.5B was never instruction-
tuned, and neither are these continued-pretraining runs), not a chat
assistant - the model will continue your prompt as text, not "answer" it
the way a chatbot would.

Usage:
    # one-off prompt
    python generate.py --model_path output/runs/original/seed1/final \\
        --prompt "Dia duit,"

    # interactive loop - keep typing prompts, Ctrl+C / 'exit' to quit
    python generate.py --model_path output/runs/custom/seed1/final --interactive

On Hypatia, run this inside a GPU allocation (interactive srun, same pattern
as train_cpt.py), and export HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
first -- this script doesn't set them itself, unlike train_cpt.py/
evaluate_test_set1.py, so compute nodes without internet access need it set
in the shell before running this.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate(prompt, tokenizer, model, device, max_new_tokens, temperature, top_p,
             repetition_penalty=1.3, no_repeat_ngram_size=3):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True, help="Path to a <run>/final directory.")
    parser.add_argument("--prompt", default=None, help="Single prompt to generate from. Omit for --interactive.")
    parser.add_argument("--interactive", action="store_true", help="Keep prompting in a loop instead of a single run.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 = greedy/deterministic.")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.3,
                         help="1.0 = off. >1 discourages repeating already-generated tokens -- added "
                              "after observing repetition loops even at greedy decoding.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3,
                         help="0 = off. Blocks any 3-token sequence from repeating verbatim.")
    parser.add_argument("--device", default=None, help="cuda / cpu. Auto-detected if omitted.")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model_path} onto {device}...")
    tokenizer, model = load(args.model_path, device)

    if args.interactive:
        print("Interactive mode. Type a prompt and press enter ('exit' or Ctrl+C to quit).")
        while True:
            try:
                prompt = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            if not prompt.strip():
                continue
            text = generate(prompt, tokenizer, model, device, args.max_new_tokens, args.temperature, args.top_p, args.repetition_penalty, args.no_repeat_ngram_size)
            print(text)
    else:
        if not args.prompt:
            parser.error("--prompt is required unless --interactive is set.")
        text = generate(args.prompt, tokenizer, model, device, args.max_new_tokens, args.temperature, args.top_p, args.repetition_penalty, args.no_repeat_ngram_size)
        print(text)


if __name__ == "__main__":
    main()
