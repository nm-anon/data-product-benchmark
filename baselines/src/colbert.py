import json
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient, DataType

# Requires: pip install ragatouille pymilvus
try:
    from ragatouille import RAGPretrainedModel
except ImportError:
    print("Error: Please install ragatouille using 'pip install ragatouille'")

def ingest_colbert_collection(client, data, collection_name, dim=128, remove_if_exists=False):
    """
    Ingests data into Milvus. 
    Note: Standard ColBERT (v2) uses 128-dimensional vectors.
    """
    if client.has_collection(collection_name=collection_name):
        if remove_if_exists:
            client.drop_collection(collection_name=collection_name)
        else:
            print(f"Collection {collection_name} exists — skipping ingestion.")
            return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="uuid", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

    client.create_collection(collection_name=collection_name, schema=schema)
    print(f"Created collection: {collection_name}")

def get_table_representation(table_data):
    title = table_data.get("title", "")
    headers = table_data.get("header", [])
    rows = table_data.get("rows", [])
    # Flatten table into a string for ColBERT tokenization
    tbl_rep = f"Title: {title} Headers: {' | '.join(headers)} "
    for row in rows[:5]: # Limit rows for baseline performance
        tbl_rep += " | ".join([str(cell) for cell in row]) + " "
    return tbl_rep

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # initialize ColBERT Model via RAGatouille
    # This downloads the colbertv2.0 checkpoint (approx 450MB)
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    with open(args.corpus, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    table_contents, table_uuids = [], []
    text_contents, text_uuids = [], []
    seen_text = set()

    for entry in tqdm(corpus, desc="Parsing Corpus"):
        t_uid = entry["table"]["uid"]
        if t_uid not in table_uuids:
            table_contents.append(get_table_representation(entry["table"]))
            table_uuids.append(t_uid)
        
        for passg in entry["synth_text"]:
            if passg['uid'] not in seen_text:
                text_contents.append(f"{passg['title']}: {passg['text']}")
                text_uuids.append(passg['uid'])
                seen_text.add(passg['uid'])

    # ColBERT doesn't use standard flat vectors; it builds a 'PLAID' index.
    print("Indexing Text...")
    text_index_path = RAG.index(
        index_name="text_colbert",
        collection=text_contents,
        document_ids=text_uuids,
        use_faiss=True
    )

    print("Indexing Tables...")
    table_index_path = RAG.index(
        index_name="table_colbert",
        collection=table_contents,
        document_ids=table_uuids,
        use_faiss=True
    )

    with open(args.dpr, "r", encoding="utf-8") as f:
        dprs = [json.loads(line) for line in f if line.strip()]

    query_results = []
    for dpr in tqdm(dprs, desc="Searching"):
        query_text = dpr["DPR"]
        
        # ColBERT Search returns (content, rank, score, doc_id)
        raw_text_res = RAG.search(query=query_text, index_name="text_colbert", k=args.top_text)
        raw_table_res = RAG.search(query=query_text, index_name="table_colbert", k=args.top_table)

        # Reformat to match baseline results structure: [(uuid, score), ...]
        text_rank = [(res['document_id'], res['score']) for res in raw_text_res]
        table_rank = [(res['document_id'], res['score']) for res in raw_table_res]

        query_results.append({
            "dpr_id": dpr["dpr_id"],
            "dpr": query_text,
            "results": {
                "colbert": {
                    "text": text_rank,
                    "table": table_rank
                }
            }
        })

    # 5. Save Results
    output = {
        "dataset": args.dataset,
        "model": "colbertv2.0",
        "results": query_results
    }
    
    with open(args.output_path, "w") as out_f:
        json.dump(output, out_f, indent=1)
    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=str, default="data/TATQA_corpus.json")
    p.add_argument("--dpr", type=str, default="data/TATQA_test.jsonl")
    p.add_argument("--dataset", type=str, default="TATQA")
    p.add_argument("--output_path", type=str, default="data/colbert_results.json")
    p.add_argument("--top_text", type=int, default=100)
    p.add_argument("--top_table", type=int, default=100)
    args = p.parse_args()
    main(args)