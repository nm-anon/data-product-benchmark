import json
import uuid
import os

def generate_uid(name):
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(namespace, name))

def is_valid_data_row(row):
    return len(row) > 1 and any(cell.strip() for cell in row[1:])

def load_convfinqa_corpus(input_path):

    with open(input_path, "r") as f:
        data = json.load(f)

    corpus = []

    for item in data:
        table_id = item["id"]
        raw_table = item["table"]

        if not raw_table or len(raw_table) < 2:
            continue

        # build header from row headers (first column), and values from the rest
        data_rows = [row for row in raw_table if is_valid_data_row(row)]
        header = [row[0] for row in data_rows if len(row) > 1]
        value_rows = [row[1:] for row in data_rows if len(row) > 1]
        value = list(map(list, zip(*value_rows)))  # transpose

        pre_text = item.get("pre_text", [])
        post_text = item.get("post_text", [])

        # Use first sentence of pre_text as title
        title = pre_text[0].strip() if pre_text else ""

        # remaining pre_text (except title) and post_text become text entries
        text_blocks = []
        for sentence in pre_text[1:]:
            if sentence.strip():
                text_blocks.append({
                    "uid": generate_uid(sentence),
                    "value": sentence.strip()
                })
        for sentence in post_text:
            if sentence.strip():
                text_blocks.append({
                    "uid": generate_uid(sentence),
                    "value": sentence.strip()
                })

        entry = {
            "table": {
                "uid": generate_uid(table_id),
                "table_id": table_id,
                "title": title,
                "header": header,
                "value": value
            },
            "text": text_blocks,
            "questions": pre_text  # entire pre_text as a list of questions
        }

        corpus.append(entry)

    return corpus

def prepare_split(split):

    if split == "test":
        input_path = f"benchmark_framework/data/raw/ConvFinQA/data/test_turn_private.json"
    else:
        input_path = f"benchmark_framework/data/raw/ConvFinQA/data/{split}.json"
    output_path = f"benchmark_framework/data/output/ConvFinQA/{split}/ConvFinQA_{split}_corpus.json"

    corpus = load_convfinqa_corpus(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Saved {len(corpus)} entries to {output_path}")

if __name__ == "__main__":
    print("Creating ConvFinQA corpus...")
    splits = ['train', 'dev', 'test']
    for split in splits:
        print(f"   Split: ConvFinQA {split}")
        prepare_split(split)

