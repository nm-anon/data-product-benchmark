# step 3: generation 
# TATQA, HybridQA, ConvFinQA
DATA_NAME="ConvFinQA"
MAX_WORKERS=50

SPLITS=("train" "test" "dev")
DPR_LLMS=("qwen-2-5-72b" "mixtral-8x22b" "llama-3-3-70b" "gpt-oss-120b" "DeepSeek-V3")

for SPLIT in "${SPLITS[@]}"; do
    for MODEL_NAME in "${DPR_LLMS[@]}"; do
        RAW_CLUSTER_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_filtered_clusters.json"
        CORPUS_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_corpus.json"
        OUTPUT_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_dprs-${MODEL_NAME}.jsonl"

        python benchmark_framework/src/generator.py \
            --raw_cluster_path "$RAW_CLUSTER_PATH" \
            --corpus_path "$CORPUS_PATH" \
            --model_name "$MODEL_NAME" \
            --output_path "$OUTPUT_PATH" \
            --max_workers "$MAX_WORKERS"
    done
done