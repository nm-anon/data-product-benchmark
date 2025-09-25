# step 2: filtering and revising for clusters

# TATQA, HybridQA, ConvFinQA
DATA_NAME="ConvFinQA"
EMBED_MODEL="all-MiniLM-L6-v2"
MAX_TABLES=30

SPLITS=("train" "dev" "test")
for SPLIT in "${SPLITS[@]}"; do
    RAW_CLUSTER_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_clusters.json"
    CORPUS_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_corpus.json"
    OUTPUT_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_filtered_clusters.json"

    python src/filter.py \
        --raw_cluster_path "$RAW_CLUSTER_PATH" \
        --corpus_path "$CORPUS_PATH" \
        --output_path "$OUTPUT_PATH" \
        --embedding_model "$EMBED_MODEL" \
        --max_tables_per_cluster "$MAX_TABLES"
done