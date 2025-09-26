# step 2: filtering and revising for clusters

# TATQA, HybridQA, ConvFinQA
EMBED_MODEL="all-MiniLM-L6-v2"
MAX_TABLES=30

DATASETS=("HybridQA" "TATQA" "ConvFinQA")
SPLITS=("train" "dev" "test")

for DATA_NAME in "${DATASETS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
        echo "Filtering $DATA_NAME - $SPLIT"
        RAW_CLUSTER_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_clusters.json"
        CORPUS_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_corpus.json"
        OUTPUT_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_filtered_clusters.json"

        python benchmark_framework/src/filter.py \
            --raw_cluster_path "$RAW_CLUSTER_PATH" \
            --corpus_path "$CORPUS_PATH" \
            --output_path "$OUTPUT_PATH" \
            --embedding_model "$EMBED_MODEL" \
            --max_tables_per_cluster "$MAX_TABLES"
    done
done