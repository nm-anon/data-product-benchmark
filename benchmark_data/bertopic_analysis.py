from bertopic import BERTopic
import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def collect_texts(dataset_dir: Path):
    name = dataset_dir.name
    texts = []
    for split in ("train", "dev", "test"):
        file_path = dataset_dir / f"{name}_{split}.jsonl"
        if not file_path.exists():
            continue
        for obj in iter_jsonl(file_path):
            dpr = obj.get("DPR", "")
            if dpr:
                texts.append(dpr)
    return texts


def analyze_with_bertopic(texts, topk=10, min_topic_size=20):
    if not texts:
        return [], None, None

    # Use default embedding model unless user configures BERTopic externally.
    topic_model = BERTopic(min_topic_size=min_topic_size, verbose=False)
    topics, _ = topic_model.fit_transform(texts)

    topic_info = topic_model.get_topic_info()
    results = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id) or []
        top_words = [w for w, _ in words[:topk]]
        results.append(
            {
                "topic_id": topic_id,
                "count": int(row["Count"]),
                "name": row["Name"],
                "top_words": top_words,
            }
        )

    results.sort(key=lambda x: x["count"], reverse=True)
    return results, topic_model, topics


def save_bertopic_visuals(topic_model, dataset_name, out_dir: Path):
    if topic_model is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig = topic_model.visualize_barchart(top_n_topics=10, n_words=10)
        fig.write_html(str(out_dir / f"{dataset_name}_barchart.html"))
    except Exception:
        pass

    try:
        fig = topic_model.visualize_topics()
        fig.write_html(str(out_dir / f"{dataset_name}_topics.html"))
    except Exception:
        pass

    try:
        fig = topic_model.visualize_hierarchy()
        fig.write_html(str(out_dir / f"{dataset_name}_hierarchy.html"))
    except Exception:
        pass


def render_markdown(results):
    lines = []
    lines.append("# BERTopic Topic Analysis")
    lines.append("")
    for item in results:
        lines.append(f"## {item['dataset']}")
        topics = item["topics"]
        if not topics:
            lines.append("- No topics found.")
            lines.append("")
            continue
        for t in topics:
            words = ", ".join(t["top_words"])
            lines.append(f"- Topic {t['topic_id']} (count {t['count']}): {words}")
        lines.append("")
    return "\n".join(lines)





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Benchmark data root")
    parser.add_argument("--topk", type=int, default=10, help="Top words per topic")
    parser.add_argument("--min-topic-size", type=int, default=20, help="Minimum topic size")
    parser.add_argument("--json-out", default="bertopic_topics.json")
    parser.add_argument("--md-out", default="bertopic_topics.md")
    parser.add_argument("--html-dir", default="bertopic_visuals", help="Directory for BERTopic HTML charts")
    args = parser.parse_args()


    root = Path(args.root)
    datasets = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]

    results = []
    for d in sorted(datasets):
        texts = collect_texts(d)
        topics, topic_model, _ = analyze_with_bertopic(
            texts, topk=args.topk, min_topic_size=args.min_topic_size
        )
        results.append({"dataset": d.name, "topics": topics})
        save_bertopic_visuals(topic_model, d.name, Path(args.html_dir))

    json_out = Path(args.json_out)
    json_out.write_text(json.dumps(results, indent=2))

    md_out = Path(args.md_out)
    md_out.write_text(render_markdown(results))

    print(f"Wrote {json_out} and {md_out}")


if __name__ == "__main__":
    main()
