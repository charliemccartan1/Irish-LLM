"""
Evaluates BLEU (gle<->eng, Lankford et al. 2022 / gaHealth), 5-shot,
generation-based, scored via sacrebleu (real, standard BLEU-4). Uses
tokenizer.apply_chat_template() -- matching how the model was actually
SFT'd (chat-formatted Dolly-V2-gle), NOT raw continuation prompting.
This is the deliberate fix for the rambling/non-stopping problem seen in
UCCIX's gaHealth task, which uses plain "Bearla: ...\\nGaeilge:" style
prompting with no chat structure at all.

Usage:
    python evaluate_bleu.py --direction en2ga --model_path output/runs/original/seed1/final --model_tag baseline
    python evaluate_bleu.py --direction ga2en --model_path output/runs/original/seed1/final --model_tag baseline --adapter_path output/runs/sft_baseline/final
"""
import argparse
import json
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

SCRIPT_DIR = Path(__file__).resolve().parent

DIRECTION_LABELS = {"en2ga": ("English", "Irish"), "ga2en": ("Irish", "English")}


def load_model_and_tokenizer(model_path, adapter_path=None, tokenizer_path=None):
    is_post_sft = adapter_path is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    if is_post_sft and tokenizer.chat_template is None:
        raise ValueError(
            f"{tokenizer_path or model_path}'s tokenizer has no chat_template -- needed for "
            "POST-sft evaluation. For custom's post-SFT checkpoint specifically, pass "
            "--tokenizer_name pointing at the known-good qwen2.5-1.5b-custom copy."
        )

    eos_ids = [tokenizer.eos_token_id]
    if is_post_sft:
        # Real bug found by inspecting actual generated output: without
        # this, generate() has no way to recognize the chat template's
        # real end-of-turn marker (often different from the generic
        # eos_token_id), so it ran to max_new_tokens every time --
        # producing correct translations followed by rambling into new
        # fake turns, repeated tokens, and garbage. <|im_end|> is
        # Qwen2.5's real turn-boundary token.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id not in eos_ids:
            eos_ids.append(im_end_id)
    print(f"[evaluate_bleu] generation will stop at token ids: {eos_ids}")

    return tokenizer, model, eos_ids


def build_messages(fewshot_examples, source_text, src_lang, tgt_lang):
    """Chat-formatted version, for POST-SFT models only -- matches the
    turn structure the SFT data actually used."""
    messages = []
    for ex in fewshot_examples:
        messages.append({"role": "user", "content": f"Translate this {src_lang} text to {tgt_lang}: {ex['source']}"})
        messages.append({"role": "assistant", "content": ex["target"]})
    messages.append({"role": "user", "content": f"Translate this {src_lang} text to {tgt_lang}: {source_text}"})
    return messages


class StopOnNewline(StoppingCriteria):
    """Real, generation-time stopping criterion -- halts model.generate()
    the INSTANT a newline appears in the newly-generated text, checked
    after every single token. This directly mirrors lm-eval-harness's own
    'until: ["\\n"]' mechanism (confirmed from UCCIX's actual gaHealth
    task YAML), which is a hard external stop condition, NOT dependent on
    the model correctly predicting its own eos/im_end token. Given the
    real, confirmed finding that this model's SFT training data is
    single-turn only, it likely never learned to reliably predict a stop
    token in this evaluation's multi-turn few-shot context -- this
    stopping criterion doesn't need it to."""

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0][self.prompt_len:]
        if len(generated_ids) == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return "\n" in text


def build_raw_prompt(fewshot_examples, source_text, src_lang, tgt_lang):
    """Plain continuation format, for PRE-SFT (CPT-only) models --
    matches every other benchmark script in this project (IQA, SIB,
    Cloze, NQ). A CPT-only model was never trained on chat-formatted
    turns, so using apply_chat_template() here would test it in a format
    it has no learned behavior for at all -- a genuine structural
    mismatch, not just the earlier missing-stop-token bug."""
    blocks = [f"{src_lang}: {ex['source']}\n{tgt_lang}: {ex['target']}" for ex in fewshot_examples]
    prefix = "\n\n".join(blocks) + "\n\n"
    return prefix + f"{src_lang}: {source_text}\n{tgt_lang}:"


def translate(fewshot_examples, source_text, src_lang, tgt_lang, tokenizer, model, device, eos_ids, is_post_sft):
    if is_post_sft:
        messages = build_messages(fewshot_examples, source_text, src_lang, tgt_lang)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = build_raw_prompt(fewshot_examples, source_text, src_lang, tgt_lang)

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    stopping_criteria = StoppingCriteriaList([StopOnNewline(tokenizer, input_ids.shape[1])])
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
            stopping_criteria=stopping_criteria,
            # NOTE: repetition_penalty/no_repeat_ngram_size deliberately
            # NOT used here -- neither UCCIX's confirmed generation config
            # (do_sample=false, temperature=0.0, no repetition mitigation)
            # nor Qomhra's own methodology (their paper reports the
            # unmitigated rambling as a genuine, honestly-reported
            # limitation) includes this. Adding it would give this
            # model's generation an intervention the comparison targets
            # don't receive, invalidating the comparison.
        )
    generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Direct, unambiguous live diagnostic: if n_tokens is consistently
    # near 128, the stopping mechanism (eos_ids AND StopOnNewline) is NOT
    # firing -- generation is running to the full cap every time. If
    # n_tokens is small and varies example to example, something is
    # genuinely causing early stops.
    n_tokens = output_ids.shape[1] - input_ids.shape[1]
    stop_status = "HIT MAX CAP" if n_tokens >= 128 else "stopped early"
    print(f"    [gen] {n_tokens}/128 tokens ({stop_status})", flush=True)

    if not is_post_sft:
        # Raw continuation risks drifting into a new source/target block
        # the same way NQ's raw prompting does -- same mitigation applied
        generated = generated.split("\n")[0]
    return generated.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=["en2ga", "ga2en"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="LoRA adapter for POST-sft. Omit for PRE-sft.")
    parser.add_argument("--tokenizer_name", default=None,
                         help="override tokenizer path -- needed for custom pre-SFT (see chat_template check)")
    parser.add_argument("--model_tag", required=True, help="e.g. 'baseline' or 'custom'")
    parser.add_argument("--n_shot", type=int, default=5, help="Qomhra's paper confirms BLEU is 5-shot")
    parser.add_argument("--limit", type=int, default=None, help="cap eval examples, for a quick check")
    args = parser.parse_args()

    src_lang, tgt_lang = DIRECTION_LABELS[args.direction]

    with open(SCRIPT_DIR / f"bleu_{args.direction}_fewshot.json", encoding="utf-8") as f:
        fewshot_examples = json.load(f)[: args.n_shot]
    with open(SCRIPT_DIR / f"bleu_{args.direction}_test.json", encoding="utf-8") as f:
        pairs = json.load(f)
    if args.limit:
        pairs = pairs[: args.limit]

    stage = "POST-SFT" if args.adapter_path else "PRE-SFT"
    tag = f"{args.model_tag}_{args.direction}_{'post' if args.adapter_path else 'pre'}sft"

    tokenizer, model, eos_ids = load_model_and_tokenizer(args.model_path, args.adapter_path, args.tokenizer_name)
    device = next(model.parameters()).device

    hypotheses, references = [], []
    for i, pair in enumerate(pairs):
        response = translate(fewshot_examples, pair["source"], src_lang, tgt_lang, tokenizer, model, device, eos_ids, args.adapter_path is not None)
        hypotheses.append(response)
        references.append(pair["reference"])

        # Live check, visible in tail -f: without an explicit flush, SLURM's
        # output redirection buffers stdout, so nothing shows up live at all
        # until the buffer fills or the job ends -- this print is useless
        # for real-time monitoring without flush=True.
        resp_words, ref_words = len(response.split()), max(len(pair["reference"].split()), 1)
        flag = " [RAMBLING?]" if resp_words > ref_words * 3 and resp_words > 15 else ""
        print(f"  [{i+1}/{len(pairs)}] response={resp_words}w ref={ref_words}w{flag}", flush=True)

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"[{args.model_tag}, {args.direction}, {stage}, {args.n_shot}-shot] BLEU: {bleu.score:.2f}")

    out_path = SCRIPT_DIR / f"bleu_{tag}_results.json"
    with open(out_path, "w") as f:
        json.dump({"model_tag": args.model_tag, "direction": args.direction, "stage": stage,
                   "n_shot": args.n_shot, "bleu": bleu.score, "n_total": len(pairs),
                   "sample_hypotheses": hypotheses[:5], "sample_references": references[:5]}, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
