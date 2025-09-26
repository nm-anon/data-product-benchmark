# step 1: run cluster
# TATQA, HybridQA, ConvFinQA

# Clustering hyperparameters
DATA_NAME="ConvFinQA"
UMAP_N_NEIGHBORS=3
UMAP_MIN_DIST=0.0
UMAP_N_COMPONENTS=15
UMAP_METRIC="cosine"
HDBSCAN_MIN_CLUSTER_SIZE=2
HDBSCAN_METRIC="euclidean"
HDBSCAN_EPSILON=0.1

DATASETS=("HybridQA" "TATQA" "ConvFinQA")
SPLITS=("train" "dev" "test")

for DATA_NAME in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        echo "Clustering $DATA_NAME - $SPLIT"
        CORPUS_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_corpus.json"
        python benchmark_framework/src/cluster.py \
            --corpus_path "$CORPUS_PATH" \
            --data_name "$DATA_NAME" \
            --save_dir "benchmark_framework/data/output/${DATA_NAME}/${SPLIT}" \
            --split "${SPLIT}" \
            --embedding_model all-MiniLM-L6-v2 \
            --umap_n_neighbors "$UMAP_N_NEIGHBORS" \
            --umap_min_dist "$UMAP_MIN_DIST" \
            --umap_n_components "$UMAP_N_COMPONENTS" \
            --umap_metric cosine \
            --hdbscan_min_cluster_size "$HDBSCAN_MIN_CLUSTER_SIZE" \
            --hdbscan_metric euclidean \
            --hdbscan_epsilon "$HDBSCAN_EPSILON"
    done
done


