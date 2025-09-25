import os
import json
from collections import defaultdict
from math import ceil
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    with open(args.raw_cluster_path, "r") as f:
        raw_clusters = json.load(f)
    with open(args.corpus_path, "r") as f:
        corpus = json.load(f)

    table_meta = {}
    for entry in corpus:
        if "table" not in entry:
            continue
        t = entry["table"]
        tid_full = t["table_id"]
        title = t.get("title", "")
        columns = t.get("header", [])
        text = ""
        if "text" in entry and isinstance(entry["text"], list):
            text = " ".join(seg.get("value", "") for seg in entry["text"])
        embed_input = f"{title}. Columns: {' | '.join(columns)}." + (f" Context: {text}" if text else "")
        table_meta[tid_full] = {"title": title, "columns": columns, "embed_input": embed_input}
        base_id = tid_full.rsplit("_", 1)[0]
        if base_id not in table_meta:
            table_meta[base_id] = table_meta[tid_full]
    
    # embedding model
    model = SentenceTransformer(args.embedding_model)

    interim = []
    missing = 0
    for raw_cid, tables in tqdm(raw_clusters.items(), desc="Revising clusters"):
        if raw_cid == '-1':
            continue
        embed_texts = []
        entries = []
        for tbl in tables:
            raw_tid = tbl["table_id"]
            meta_key = raw_tid if raw_tid in table_meta else raw_tid.rsplit("_", 1)[0]
            meta = table_meta.get(meta_key)
            if not meta:
                missing += 1
                continue
            embed_texts.append(meta["embed_input"])
            entries.append({"table_id": raw_tid, "meta_key": meta_key, "questions": tbl["questions"]})
        if not entries:
            continue

        embeds = model.encode(embed_texts, show_progress_bar=False)
        n_tables = len(entries)

        if n_tables <= args.max_tables_per_cluster:
            labels = np.zeros(n_tables, dtype=int)
            n_sub = 1
        else: # larger than threshold
            n_sub = ceil(n_tables / args.max_tables_per_cluster)
            km = KMeans(n_clusters=n_sub, random_state=42, n_init="auto")
            labels = km.fit_predict(embeds)

        groups = defaultdict(list)
        for lbl, ent in zip(labels, entries):
            groups[lbl].append(ent)

        for lbl, group in groups.items():
            key = raw_cid if n_sub == 1 else f"{raw_cid}_sub{lbl}"
            tables_out = []
            for ent in group:
                meta = table_meta[ent["meta_key"]]
                tables_out.append({
                    "table_id": ent["table_id"],
                    "table_title": meta["title"],
                    "table_columns": meta["columns"],
                    "questions": ent["questions"]
                })
            interim.append((key, tables_out))

    output_clusters = []
    for idx, (subkey, tables_out) in enumerate(interim, start=1):
        output_clusters.append({
            "dpr_id": str(idx),
            "cluster_key": subkey,
            "tables": tables_out
        })


    with open(args.output_path, "w") as f:
        json.dump(output_clusters, f, indent=2)

    print(f"Saved {len(output_clusters)} DPR clusters to {args.output_path}")
    print(f"Skipped {missing} tables not found in corpus metadata")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_cluster_path", type=str, default="data/output/TATQA_table_level_q_clusters.json")
    parser.add_argument("--corpus_path", type=str, default="data/output/TATQA_corpus.json")
    parser.add_argument("--output_path", type=str, default="data/output/TATQA_filtered_clusters.json")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--max_tables_per_cluster", type=int, default=30)
    args = parser.parse_args()

    main(args)
