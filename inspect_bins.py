import random
import numpy as np
from transformers import AutoTokenizer

BIN_FILE = "processed_data/custom/packed/train.bin"
BLOCK_SIZE = 2048
NUM_BLOCKS_TO_SHOW = 5

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

tokens = np.memmap(BIN_FILE, dtype=np.uint32, mode="r")

total_blocks = len(tokens) // BLOCK_SIZE

print(f"Total tokens: {len(tokens):,}")
print(f"Total blocks: {total_blocks:,}\n")

random.seed(47)

blocks = random.sample(range(total_blocks), min(NUM_BLOCKS_TO_SHOW, total_blocks))

for block in blocks:
    start = block * BLOCK_SIZE
    end = start + BLOCK_SIZE

    block_tokens = tokens[start:end]

    print("=" * 100)
    print(f"Block {block:,}")
    print("=" * 100)
    print(tokenizer.decode(block_tokens.tolist()))
    print()