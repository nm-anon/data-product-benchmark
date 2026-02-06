import argparse
import json
import re
from math import cos, pi, sin
from pathlib import Path
import matplotlib.pyplot as plt

# COLORS = [
#     "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
#     "#EDC948", "#B07AA1", "#FF9DA7", 
#     "#8CD17D", "#D4A6C8", "#FABFD2",  "#D37295",
    
# ]

COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7",
    "#8CD17D", "#D4A6C8", "#FABFD2", "#D37295",
    "#86BCB6", "#F1CE63", "#E695A1", "#8DA0CB",
    "#66C2A5", "#FC8D62", "#E78AC3", "#A6D854",
    "#FFD92F", "#E5C494", "#B3B3B3", "#A1D99B",
    "#9EDAE5", "#C7E9C0", "#FDD0A2", "#C6DBEF",
    "#F2B6CF", "#BDBDBD",
]


def _clean_topic_name(name):
    text = name.strip().replace("_", " ")
    text = re.sub(r"^\d+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _topic_label(topic):
    name = _clean_topic_name(topic.get("name") or "")
    if name:
        return name
    words = topic.get("top_words") or []
    if words:
        return " / ".join(w.replace("_", " ") for w in words[:4]).title()
    topic_id = topic.get("topic_id")
    return f"Topic {topic_id}" if topic_id is not None else "Topic"


def save_topic_pie_from_json(
    json_path: Path,
    out_dir: Path,
    min_share: float = 0.02,
    min_count: int = 0,
):
    data = json.loads(json_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in data:
        dataset = item.get("dataset", "dataset")
        topics = item.get("topics", [])
        if not topics:
            continue

        topics_sorted = sorted(topics, key=lambda x: x.get("count", 0), reverse=True)
        total = sum(t.get("count", 0) for t in topics_sorted)
        if total == 0:
            continue

        labels = []
        sizes = []
        other_count = 0
        for t in topics_sorted:
            count = t.get("count", 0)
            share = count / total
            if count >= min_count and share >= min_share:
                labels.append(_topic_label(t))
                sizes.append(count)
            else:
                other_count += count

        other_share = other_count / total if total else 0
        if other_count > 0 and other_share <= 0.5:
            labels.append("Other")
            sizes.append(other_count)
        elif other_share > 0.5:
            # If "Other" would dominate, show all topics instead.
            labels = []
            sizes = []
            for t in topics_sorted:
                labels.append(_topic_label(t))
                sizes.append(t.get("count", 0))

        if not sizes:
            continue

        # Aggregate repeated labels to avoid duplicate legend entries.
        agg = {}
        for label, count in zip(labels, sizes):
            agg[label] = agg.get(label, 0) + count
        labels = list(agg.keys())
        sizes = list(agg.values())

        colors = [COLORS[i % len(COLORS)] for i in range(len(sizes))]
        sorted_data = sorted(zip(sizes, labels, colors), key=lambda x: x[0], reverse=True)
        counts_sorted, topics_sorted, colors_sorted = zip(*sorted_data)
        total_count = sum(counts_sorted)

        fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
        wedges, _ = ax.pie(
            counts_sorted,
            startangle=90,
            colors=colors_sorted,
            radius=1.0,
            wedgeprops=dict(width=0.4),
        )

        centre_circle = plt.Circle((0, 0), 0.60, fc="white")
        ax.add_artist(centre_circle)

        ax.text(
            0,
            0,
            f"{total_count:,}\nTotal",
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
        )

        for wedge, count in zip(wedges, counts_sorted):
            percent = count / total_count * 100
            if percent < 1.3:
                continue
            theta = (wedge.theta2 + wedge.theta1) / 2.0
            rad = pi / 180 * theta
            x = cos(rad) * 1.05
            y = sin(rad) * 1.05
            ax.text(x, y, f"{percent:.1f}%", fontsize=13, ha="center", va="center")

        ax.legend(
            wedges,
            topics_sorted,
            title="Topics",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=12,
            title_fontsize=14,
        )

        plt.title(f"{dataset} Topic Coverage", fontsize=18, fontweight="bold")
        plt.tight_layout()
        fig.savefig(out_dir / f"{dataset}_topic_pie.png", bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="bertopic_topics.json", help="BERTopic JSON output file")
    parser.add_argument("--out", default="bertopic_pies", help="Output directory for PNG charts")
    parser.add_argument("--min-share", type=float, default=0.02, help="Minimum share to show as its own slice")
    parser.add_argument("--min-count", type=int, default=0, help="Minimum count to show as its own slice")
    args = parser.parse_args()

    save_topic_pie_from_json(Path(args.json), Path(args.out), min_share=args.min_share, min_count=args.min_count)
    print(f"Wrote pie charts to {args.out}")


if __name__ == "__main__":
    main()
