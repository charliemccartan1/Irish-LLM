from datasets import load_dataset
from pathlib import Path
import re

# ----------------------------
# Load datasets
# ----------------------------

irish_ds = load_dataset(
    "ReliableAI/Irish-Text-Collection",
    split="train",
)

bitext_ds = load_dataset(
    "ReliableAI/Irish-English-Parallel-Collection",
    split="train",
)

# ----------------------------
# Output
# ----------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
Path(SCRIPT_DIR.parent / "data" / "irish").mkdir(parents=True, exist_ok=True)
out_file = SCRIPT_DIR.parent / "data" / "irish" / "irish_data.txt"

documents = 0
total_chars = 0

# Remove anything contained in square brackets, for example:
# [GA], [EN], [citation needed], [note]
square_brackets = re.compile(r"\[[^\]]*\]")

def clean_text(text: str) -> str:
    text = text.strip()

    # Remove all square-bracketed content
    text = square_brackets.sub("", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


with out_file.open("w", encoding="utf-8") as f:

    # ----------------------------
    # Monolingual Irish
    # ----------------------------

    for row in irish_ds:
        text = clean_text(row["text"])

        if not text:
            continue

        f.write(text)
        f.write("<|endoftext|>\n")

        documents += 1
        total_chars += len(text)

    # ----------------------------
    # Irish-English bitext
    # ----------------------------

    for row in bitext_ds:
        ga = clean_text(row["text"])
        en = clean_text(row["eng_text"])

        if not ga or not en:
            continue

        sample = f"[ga] {ga}\n[en] {en}"

        f.write(sample)
        f.write("<|endoftext|>\n")

        documents += 1
        total_chars += len(sample)

print(f"Documents : {documents:,}")
print(f"Characters: {total_chars:,}")
print(f"Saved to  : {out_file.resolve()}")