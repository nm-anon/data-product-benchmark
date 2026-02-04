#!/usr/bin/env python3
"""
Compute dataset-level and corpus-level characteristics for the benchmark.
Produces JSON and Markdown summaries, including a basic topic coverage view.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","in","is","it","its","of","on","or","that","the","to","was","were","will","with",
    "about","above","after","again","against","all","am","any","because","been","before","being","below","between","both","but","could","did","do","does","doing","down",
    "during","each","few","further","had","having","he","her","here","hers","herself","him","himself","his","how","i","if","into","itself","just","more","most",
    "my","myself","no","nor","not","now","off","once","only","other","our","ours","ourselves","out","over","own","same","she","should","so","some","such","than",
    "their","theirs","them","themselves","then","there","these","they","this","those","through","too","under","until","up","very","we","what","when","where","which","who",
    "whom","why","you","your","yours","yourself","yourselves",
    "million","billion","percent","per","year","years","quarter","quarters","fiscal","fy","q1","q2","q3","q4"
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'\-]{2,}")


def tokenize(text: str):
    tokens = []
    for m in TOKEN_RE.finditer(text or ""):
        t = m.group(0).lower()
        if t in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        if len(t) < 3:
            continue
        tokens.append(t)
    return tokens


def extract_terms(text: str, use_bigrams: bool = True):
    toks = tokenize(text)
    terms = toks[:]
    if use_bigrams and len(toks) >= 2:
        terms.extend([f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)])
    return terms


def compute_topic_coverage(text_iterable, topk=25):
    docs = []
    for text in text_iterable:
        terms = extract_terms(text)
        docs.append(terms)
    if not docs:
        return []

    df = Counter()
    tf = Counter()
    for terms in docs:
        tf.update(terms)
        df.update(set(terms))

    n_docs = len(docs)
    scores = {}
    for term, term_tf in tf.items():
        term_df = df[term]
        idf = math.log((n_docs + 1) / (term_df + 1)) + 1.0
        scores[term] = term_tf * idf

    top_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
    coverage = []
    for term, score in top_terms:
        coverage.append({
            "term": term,
            "score": round(score, 4),
            "doc_coverage": round(df[term] / n_docs, 4),
            "doc_count": df[term],
        })
    return coverage


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def dataset_characteristics(dataset_dir: Path, topk=25):
    name = dataset_dir.name
    splits = {}
    dpr_texts = []

    gt_table_ids = set()
    gt_text_ids = set()
    gt_synth_ids = set()

    total_gt_tables = 0
    total_gt_texts = 0
    total_gt_synth = 0
    dpr_lengths = []

    for split in ("train", "dev", "test"):
        file_path = dataset_dir / f"{name}_{split}.jsonl"
        if not file_path.exists():
            continue
        count = 0
        for obj in iter_jsonl(file_path):
            count += 1
            dpr = obj.get("DPR", "")
            dpr_texts.append(dpr)
            dpr_lengths.append(len(dpr.split()))

            gt = obj.get("ground_truth", {})
            tables = gt.get("table", []) or []
            texts = gt.get("text", []) or []
            synths = gt.get("synth_text", []) or []

            total_gt_tables += len(tables)
            total_gt_texts += len(texts)
            total_gt_synth += len(synths)

            gt_table_ids.update(tables)
            gt_text_ids.update(texts)
            gt_synth_ids.update(synths)

        splits[split] = count

    topic_coverage = compute_topic_coverage(dpr_texts, topk=topk)

    return {
        "dataset": name,
        "splits": splits,
        "total_examples": sum(splits.values()),
        "avg_dpr_length_words": round(mean(dpr_lengths), 2) if dpr_lengths else 0.0,
        "ground_truth_counts": {
            "total_table_refs": total_gt_tables,
            "total_text_refs": total_gt_texts,
            "total_synth_text_refs": total_gt_synth,
            "unique_table_ids": len(gt_table_ids),
            "unique_text_ids": len(gt_text_ids),
            "unique_synth_text_ids": len(gt_synth_ids),
        },
        "topic_coverage": topic_coverage,
    }


def corpus_characteristics(dataset_dir: Path, topk=25):
    name = dataset_dir.name
    corpus_path = dataset_dir / f"{name}_corpus.json"
    if not corpus_path.exists():
        return {"dataset": name, "corpus": None}

    with corpus_path.open() as f:
        corpus = json.load(f)

    table_titles = []
    synth_titles = []
    text_snippets = 0
    synth_entries = 0
    table_rows = []
    table_cols = []
    table_ids = set()

    for entry in corpus:
        table = entry.get("table") or {}
        if table:
            table_titles.append(table.get("title", ""))
            table_ids.add(table.get("table_id", ""))
            values = table.get("value") or []
            header = table.get("header") or []
            table_rows.append(len(values))
            table_cols.append(len(header))

        text_list = entry.get("text") or []
        text_snippets += len(text_list)

        synth_list = entry.get("synth_text") or []
        synth_entries += len(synth_list)
        for s in synth_list:
            if isinstance(s, dict):
                synth_titles.append(s.get("title", ""))

    topic_coverage = compute_topic_coverage(table_titles + synth_titles, topk=topk)

    return {
        "dataset": name,
        "corpus": {
            "entries": len(corpus),
            "unique_table_ids": len([t for t in table_ids if t]),
            "total_text_snippets": text_snippets,
            "total_synth_text": synth_entries,
            "avg_table_rows": round(mean(table_rows), 2) if table_rows else 0.0,
            "avg_table_cols": round(mean(table_cols), 2) if table_cols else 0.0,
        },
        "topic_coverage": topic_coverage,
    }


def render_markdown(results):
    lines = []
    lines.append("# Benchmark Characteristics")
    lines.append("")

    for item in results:
        name = item["dataset"]
        lines.append(f"## {name}")

        ds = item.get("dataset_stats")
        if ds:
            lines.append("### Dataset-Level")
            lines.append(f"- Total examples: {ds['total_examples']}")
            lines.append(f"- Splits: {ds['splits']}")
            lines.append(f"- Avg DPR length (words): {ds['avg_dpr_length_words']}")
            gt = ds["ground_truth_counts"]
            lines.append(f"- Ground truth refs: tables={gt['total_table_refs']}, text={gt['total_text_refs']}, synth_text={gt['total_synth_text_refs']}")
            lines.append(f"- Unique referenced IDs: tables={gt['unique_table_ids']}, text={gt['unique_text_ids']}, synth_text={gt['unique_synth_text_ids']}")
            lines.append("- Topic coverage (top terms):")
            for t in ds["topic_coverage"][:10]:
                lines.append(f"  - {t['term']} (coverage {t['doc_coverage']})")

        cs = item.get("corpus_stats")
        if cs and cs.get("corpus"):
            lines.append("### Corpus-Level")
            c = cs["corpus"]
            lines.append(f"- Entries: {c['entries']}")
            lines.append(f"- Unique table IDs: {c['unique_table_ids']}")
            lines.append(f"- Text snippets: {c['total_text_snippets']}")
            lines.append(f"- Synth text entries: {c['total_synth_text']}")
            lines.append(f"- Avg table rows: {c['avg_table_rows']}")
            lines.append(f"- Avg table cols: {c['avg_table_cols']}")
            lines.append("- Topic coverage (top terms):")
            for t in cs["topic_coverage"][:10]:
                lines.append(f"  - {t['term']} (coverage {t['doc_coverage']})")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Benchmark data root")
    parser.add_argument("--topk", type=int, default=25)
    parser.add_argument("--json-out", default="benchmark_characteristics.json")
    parser.add_argument("--md-out", default="benchmark_characteristics.md")
    args = parser.parse_args()

    root = Path(args.root)
    datasets = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]

    results = []
    for d in sorted(datasets):
        ds_stats = dataset_characteristics(d, topk=args.topk)
        cs_stats = corpus_characteristics(d, topk=args.topk)
        results.append({
            "dataset": d.name,
            "dataset_stats": ds_stats,
            "corpus_stats": cs_stats,
        })

    json_out = Path(args.json_out)
    json_out.write_text(json.dumps(results, indent=2))

    md_out = Path(args.md_out)
    md_out.write_text(render_markdown(results))

    print(f"Wrote {json_out} and {md_out}")


if __name__ == "__main__":
    main()
