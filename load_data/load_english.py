from datasets import load_dataset
from pathlib import Path

# -------------------------
# Settings
# -------------------------

NUM_DOCUMENTS = 1000000

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR.parent / "data" / "english" / "english_data.txt"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load the 10BT sample
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True,
)

documents = 0
characters = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as out:

    for sample in dataset:

        text = sample["text"].strip()

        if not text:
            continue

        out.write(text)
        out.write("<|endoftext|>\n")

        documents += 1
        characters += len(text)

        if documents >= NUM_DOCUMENTS:
            break

print("\nFinished")
print(f"Documents : {documents:,}")
print(f"Characters: {characters:,}")
print(f"Saved to  : {OUTPUT_FILE.resolve()}")