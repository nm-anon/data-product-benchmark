import json
import os


def prepare_split(split_name: str):
    # Input and output paths
    input_path = f"benchmark_framework/data/raw/TAT-QA/dataset_raw/tatqa_dataset_{split}.json"
    output_path = f"benchmark_framework/data/output/TATQA/{split_name}/TATQA_{split_name}_corpus.json"

    print("Creating corpus from TATQA...")
    corpus = []

    with open(input_path, "r") as f:
        data = json.load(f)

    for item in data:
        raw_table = item["table"]["table"]
        table_uid = item["table"]["uid"]

        # Default title is empty
        title = ""

        # Skip malformed tables
        if len(raw_table) < 3:
            continue

        # Extract title row if second row is metadata (non-empty first cell, rest empty)
        if raw_table[1][0] and all(cell.strip() == "" for cell in raw_table[1][1:]):
            title = raw_table[1][0].strip()
            data_rows = raw_table[2:]
        else:
            data_rows = raw_table[1:]

        # Transpose logic: row headers become headers, rest become values
        header = [row[0] for row in data_rows if len(row) > 1]
        value_rows = [row[1:] for row in data_rows if len(row) > 1]
        value = list(map(list, zip(*value_rows)))  # transpose values

        # Build entry with table, text, and questions
        entry = {
            "table": {
                "uid": table_uid,
                "table_id": table_uid,
                "title": title,
                "header": header,
                "value": value
            },
            "text": [
                {
                    "uid": para.get("uid", table_uid),
                    "value": para.get("text", "")
                }
                for para in item.get("paragraphs", [])
            ],
            "questions": [
                q["question"] for q in item.get("questions", []) if "question" in q
            ]
        }

        corpus.append(entry)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Saved {len(corpus)} entries to {output_path}")


if __name__ == "__main__":
    print("Creating corpus from TAT-QA")
    splits = ['train', 'dev', 'test']
    for split in splits:
        print(f"   Split: TAT-QA {split}")
        prepare_split(split)
