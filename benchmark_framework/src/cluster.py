import os
import json
import argparse
from collections import defaultdict, Counter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from scipy.spatial.distance import cdist

def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def get_table_questions(corpus):
    table_questions = defaultdict(list)
    for entry in corpus:
        table_questions[entry['table']['table_id']].append(entry['questions'])
    return table_questions

def cluster_corpus(corpus):
    table_ids = []
    table_docs = []
    for entry in corpus:
        table_ids.append(entry['table']['table_id'])
        questions = entry['questions']
        # print("doc questions",questions)
        doc = " [SEP] ".join([f"{q}" for q in questions])
        table_docs.append(doc)
    return table_ids, table_docs

# def save_or_load_jsonl(table_questions, jsonl_path):
#     table_ids = []
#     table_docs = []
#     if not os.path.exists(jsonl_path):
#         with open(jsonl_path, 'w') as f:
#             for table_id, qa_pairs in table_questions.items():
#                 # if embed == 'qa':
#                 #     doc = " [SEP] ".join([f"{q} [ANS] {a}" for q, a in qa_pairs])
#                 # else:
#                 doc = " [SEP] ".join([f"{q}" for q in qa_pairs])
#                 # print("test set",doc)
#                 entry = {"table_id": table_id, "doc": doc}
#                 f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#                 table_ids.append(table_id)
#                 table_docs.append(doc)
#         print(f"Saved table-level Q&A docs to {jsonl_path}")
#     else:
#         print(f"{jsonl_path} already exists, skipping save.")
#         with open(jsonl_path, 'r') as f:
#             for line in f:
#                 entry = json.loads(line)
#                 table_ids.append(entry['table_id'])
#                 table_docs.append(entry['doc'])
#     return table_ids, table_docs

def visualize(topic_model, save_dir, embed):
    fig_path = os.path.join(save_dir, f"{embed}_intertopic_distance_map.html")
    topic_model.visualize_topics().write_html(fig_path)
    print(f"Saved BERTopic cluster visualization to {fig_path}")
    topic_model.visualize_hierarchy().write_html(os.path.join(save_dir, f"{embed}_hierarchy.html"))
    topic_model.visualize_heatmap().write_html(os.path.join(save_dir, f"{embed}_heatmap.html"))

def compute_metrics(topic_model, topics, table_ids, table_questions, save_dir, split, data_name):
    SAVE_PATH = os.path.join(save_dir, f'{data_name}_{split}_clusters.json')
    clustered_tables = defaultdict(list)
    for idx, topic in enumerate(topics):
        clustered_tables[str(topic)].append({
            "table_id": table_ids[idx],
            "questions": table_questions[table_ids[idx]],
        })
    with open(SAVE_PATH, 'w') as f:
        json.dump(clustered_tables, f, indent=2)

    topic_labels = np.array(topics)
    valid_idx = topic_labels != -1
    X_umap = topic_model.umap_model.embedding_[valid_idx]
    y_labels = topic_labels[valid_idx]

    finite_mask = (~np.isnan(X_umap).any(axis=1)) & (~np.isinf(X_umap).any(axis=1))
    X = X_umap[finite_mask]
    y = y_labels[finite_mask]
    print(f"Samples after NaN/Inf removal: {X.shape[0]}")

    counts = Counter(y)
    valid_clusters = [c for c, n in counts.items() if n >= 2]
    if len(valid_clusters) < 2:
        print("Not enough clusters with >=2 samples to compute metrics.")
        return

    mask = np.isin(y, valid_clusters)
    X_final = X[mask]
    y_final = y[mask]
    print(f"Final: {X_final.shape[0]} samples in {len(valid_clusters)} clusters (size >=2)")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="sklearn.utils.extmath"
        )
        sil_glob = silhouette_score(X_final, y_final, metric="cosine")
        ch_glob = calinski_harabasz_score(X_final, y_final)
        db_glob = davies_bouldin_score(X_final, y_final)
    print(f"Silhouette Score (cosine):      {sil_glob:.4f}")
    print(f"Calinski-Harabasz Index:         {ch_glob:.2f}")
    print(f"Davies-Bouldin Index:            {db_glob:.2f}")

    # cluster stats
    sil_vals = silhouette_samples(X_final, y_final, metric="cosine")
    centroids = {}
    cluster_stats = {}
    for c in valid_clusters:
        idx = np.where(y_final == c)[0]
        Xc = X_final[idx]
        mu = Xc.mean(axis=0)
        centroids[c] = mu
        intra = np.mean(np.linalg.norm(Xc - mu, axis=1) ** 2)
        cluster_stats[c] = {
            'silhouette': float(sil_vals[idx].mean()),
            'intra': float(intra)
        }
    centroid_matrix = np.vstack([centroids[c] for c in valid_clusters])
    M = cdist(centroid_matrix, centroid_matrix, metric='euclidean')
    for i, ci in enumerate(valid_clusters):
        d_row = M[i].copy()
        d_row[i] = np.nan
        inter = float(np.nanmin(d_row))
        nearest = int(valid_clusters[np.nanargmin(d_row)])
        furthest = int(valid_clusters[np.nanargmax(d_row)])
        Si = cluster_stats[ci]['intra']
        Rij = []
        for j, cj in enumerate(valid_clusters):
            if i == j: continue
            Sj = cluster_stats[cj]['intra']
            Mij = float(M[i, j])
            Rij.append((Si + Sj) / Mij)
        db_comp = float(max(Rij))
        cluster_stats[ci].update({
            'inter': inter,
            'db_comp': db_comp,
            'nearest_cluster': nearest,
            'furthest_cluster': furthest
        })

    summary = {
        "total clusters": len(clustered_tables),
        "noise cluster size": len(clustered_tables["-1"]),
        "total tables": len(table_ids),
        "metrics": {
            'silhouette_score (>0.25)': float(sil_glob),
            'calinski_harabasz_index (higher better)': float(ch_glob),
            'davies_bouldin_index (<2)': float(db_glob),
        },
        "clusters": []
    }
    for topic, tables in clustered_tables.items():
        cid = int(topic)
        stats = cluster_stats.get(cid, {})
        total_q = sum(len(t['questions']) for t in tables)
        if len(tables) > 0:
            avg_q = total_q / len(tables)
        else:
            avg_q = 0
        summary['clusters'].append({
            "cluster_id": topic,
            "num_tables": len(tables),
            "total_questions": total_q,
            "avg_questions_per_table": avg_q,
            "metrics": {
                "silhouette": stats.get('silhouette'),
                "intra_cluster_mse": stats.get('intra'),
                "inter_cluster_dist": stats.get('inter'),
                "db_component": stats.get('db_comp'),
                "nearest_cluster": stats.get('nearest_cluster'),
                "furthest_cluster": stats.get('furthest_cluster')
            },
            "sample_table_ids": [t['table_id'] for t in tables[:5]]
        })
    SUM_PATH = os.path.join(save_dir, f"{data_name}_{split}_clusters_summary.json")
    with open(SUM_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary (with nearest/furthest) to {SUM_PATH}")
    print(f"Total clusters: {summary['total clusters']}")

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = args.data_name
    corpus_path = args.corpus_path
    if dataset not in ['TATQA', 'ConvFinQA', 'HybridQA']:
        print("ERROR: Dataset not found!")
        return
    corpus = load_data(corpus_path)
    table_questions = get_table_questions(corpus)

    # if dataset in ['HybridQA']:
    #     data = load_data('data/raw/hybrid_qa/train.json')
    #     # save data by tables
    #     # table_questions, table_qa_pairs = build_table_qa_pairs(data)
    #     # print("Table total number", len(table_questions))
    # elif dataset in ['TATQA']:
    #     data = load_data('data/raw/tat_qa/tatqa_dataset_train.json')
    #     # table_questions, table_qa_pairs = tat_table_qa_pairs(data)
    # else:
    #     print("DATASET NOT FOUND ERROR:", dataset)
    #     return None
    # jsonl_path = os.path.join(args.save_dir, f"{dataset}_table.jsonl")
    table_ids, table_docs = cluster_corpus(corpus)

    embedding_model = SentenceTransformer(args.embedding_model)
    # table_embeddings = embedding_model.encode(table_docs, show_progress_bar=True)
    umap_model = UMAP(n_neighbors=args.umap_n_neighbors, min_dist=args.umap_min_dist, n_components=args.umap_n_components, metric=args.umap_metric, random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=args.hdbscan_min_cluster_size, metric=args.hdbscan_metric, cluster_selection_method='leaf', cluster_selection_epsilon=args.hdbscan_epsilon, prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=5, ngram_range=(1, 2))
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=5,
        top_n_words=10,
        verbose=True,
        low_memory=False
    )
    topics, _ = topic_model.fit_transform(table_docs)
    compute_metrics(topic_model, topics, table_ids, table_questions, args.save_dir, args.split, args.data_name)
    # visualize(topic_model, args.save_dir, args.embed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--embed', type=str, default='q') # q for question only, qa for question and answer
    parser.add_argument('--data_name', type=str, default='ConvFinQA')
    parser.add_argument('--corpus_path', type=str, default='data/output/ConvFinQA_corpus.json')
    parser.add_argument('--save_dir', type=str, default='data/output/')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--umap_n_neighbors', type=int, default=5)
    parser.add_argument('--umap_min_dist', type=float, default=0.1)
    parser.add_argument('--umap_n_components', type=int, default=10)
    parser.add_argument('--umap_metric', type=str, default='cosine')
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=5)
    parser.add_argument('--hdbscan_metric', type=str, default='euclidean')
    parser.add_argument('--hdbscan_epsilon', type=float, default=0.01)

    args = parser.parse_args()
    main(args)
