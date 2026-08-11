from datasets import load_dataset
from pathlib import Path


NUM_FILES = 300000
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR.parent / "data" / "code" / "code_data.txt"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


ds = load_dataset(
    "TempestTeam/dataset-the-stack-v2-dedup-sub",
    name="Python",
    split="train",
    streaming=True,
)

documents = 0
characters = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for sample in ds:

        code = sample["content"].strip()

        if not code:
            continue

        out.write(code)
        out.write("<|endoftext|>\n")

        documents += 1
        characters += len(code)

        if documents % 1000 == 0:
            print(
                f"Documents: {documents:,} | "
                f"Characters: {characters:,}"
            )

        if documents >= NUM_FILES:
            break

print("\nFinished")
print("---------------------")
print(f"Documents: {documents:,}")
print(f"Characters: {characters:,}")
print(f"Output: {OUTPUT_FILE.resolve()}")