from datasets import load_dataset
import os

# Load a small sample (first 100 examples)
data = load_dataset(
    "HuggingFaceTB/finemath",
    "finemath-4plus",
    split="train[:200000]"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, "..", "data", "math"), exist_ok=True)
output_file = os.path.join(SCRIPT_DIR, "..", "data", "math", "math_data.txt")

with open(output_file, "w", encoding="utf-8") as out:
    for example in data:
        out.write(example["text"])
        out.write("<|endoftext|>\n")

# Print total character count
with open(output_file, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Saved to: {output_file}")
print(f"Total characters: {len(text):,}")