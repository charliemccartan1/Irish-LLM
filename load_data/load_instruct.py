import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO = "instruction-pretrain/general-instruction-augmented-corpora"
MAX_DOCUMENTS = 700_000
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SCRIPT_DIR.parent / "data" / "instruct" / "instruct_data.txt"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_text(value):
    """Extract all text strings from a JSON record."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if isinstance(value, list):
        return [text for item in value for text in get_text(item)]

    if isinstance(value, dict):
        return [text for item in value.values() for text in get_text(item)]

    return []


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

api = HfApi()

shards = sorted(
    filename
    for filename in api.list_repo_files(REPO, repo_type="dataset")
    if filename.endswith(".txt") and "/shard/" in filename
)

documents_written = 0

with OUTPUT_FILE.open("w", encoding="utf-8") as output:
    for shard in shards:
        if documents_written >= MAX_DOCUMENTS:
            break

        print(f"Downloading {shard}")

        shard_path = Path(
            hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                filename=shard,
            )
        )

        with shard_path.open("r", encoding="utf-8") as source:
            for line in source:
                if documents_written >= MAX_DOCUMENTS:
                    break

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = "\n".join(get_text(record)).strip()

                if not text:
                    continue

                output.write(text)
                output.write("<|endoftext|>\n")

                documents_written += 1

        print(f"Documents written: {documents_written:,}")

character_count = len(OUTPUT_FILE.read_text(encoding="utf-8"))

print("\nFinished")
print(f"Documents written: {documents_written:,}")
print(f"Total characters: {character_count:,}")
print(f"Output file: {OUTPUT_FILE.resolve()}")