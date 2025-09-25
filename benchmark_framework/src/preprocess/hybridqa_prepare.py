import os
import json
import uuid
from tqdm import tqdm

WTABLES_DIR  = "benchmark_framework/data/raw/WikiTables-WithLinks/tables_tok"
REQUESTS_DIR = "benchmark_framework/data/raw/WikiTables-WithLinks/request_tok"

MAX_CHUNK_LEN = 1000

def generate_uid(name):
    namespace = uuid.UUID("12345678-1234-5678-1234-567812345678")
    return str(uuid.uuid5(namespace, name))

def format_wiki_table(table_id):
    path = os.path.join(WTABLES_DIR, f"{table_id}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    header = [h[0] for h in data["header"]]
    values = [[[cell[0],cell[1]] for cell in row] for row in data["data"]]

    return {
        "uid": generate_uid(table_id),
        "table_id": table_id,
        "title": data["title"],
        "header": header,
        "value": values
    }

def load_request_passages(request_id):
    path = os.path.join(REQUESTS_DIR, f"{request_id}.json")
    if not os.path.exists(path):
        return []

    with open(path, encoding="utf-8") as f:
        url_map = json.load(f)

    #sentences = url_map.values() 
    outputs = [
        {"uid": generate_uid(p),
         "key": key,
         "value": p}
            for key, p in url_map.items()
        ]
    # return chunk_text(all_paragraphs, MAX_CHUNK_LEN)
    return outputs

def prepare_split(split_name: str):
    split_path   = f"benchmark_framework/data/raw/HybridQA/released_data/{split_name}.json"
    output       = f"benchmark_framework/data/output/HybridQA/{split_name}/HybridQA_{split_name}_corpus.json"

    with open(split_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # group questions by table_id
    questions_by_table = {}
    for ex in split_data:
        tid = ex["table_id"]
        questions_by_table.setdefault(tid, []).append(ex["question"])

    # valid table_ids
    table_ids = sorted(set(questions_by_table.keys()))
    valid_ids = [
        tid for tid in table_ids
        if os.path.exists(os.path.join(WTABLES_DIR, f"{tid}.json"))
        and os.path.exists(os.path.join(REQUESTS_DIR, f"{tid}.json"))
    ]

    print(f"Total valid tables with both .json files: {len(valid_ids)}")

    corpus = []
    for tid in tqdm(valid_ids):
        entry = {
            "table": format_wiki_table(tid),
            "text": load_request_passages(tid),
            "questions": questions_by_table.get(tid, [])
        }
        corpus.append(entry)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(corpus)} entries to {output}")


if __name__ == "__main__":
    print("Creating corpus from HybridQA")
    splits = ['train', 'dev', 'test']
    for split in splits:
        print(f"   Split: HybridQA {split}")
        prepare_split(split)

