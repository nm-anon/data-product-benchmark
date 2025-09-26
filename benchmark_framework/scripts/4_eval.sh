# step 4: evaluation

# TATQA, HybridQA, ConvFinQA
DATA_NAME="ConvFinQA"

SPLITS=("train" "dev" "test")
DPR_LLMS=("llama-3-3-70b" "gpt-oss-120b" "DeepSeek-V3" "qwen-2-5-72b" "mixtral-8x22b")
EVAL_LLMS=("llama-3-3-70b" "gpt-oss-120b" "DeepSeek-V3" "qwen-2-5-72b" "mixtral-8x22b")
DIMENSIONS=("alignment" "dpr_clarity")
MAX_WORKERS=50

for SPLIT in "${SPLITS[@]}"; do
  for DIMENSION in "${DIMENSIONS[@]}"; do
    for DPR_LLM in "${DPR_LLMS[@]}"; do
      for EVAL_LLM in "${EVAL_LLMS[@]}"; do
        DPR_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_dprs-${DPR_LLM}.jsonl"
        CORPUS_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_corpus.json"
        OUTPUT_PATH="benchmark_framework/data/output/${DATA_NAME}/${SPLIT}/${DATA_NAME}_${SPLIT}_dpr_eval_${DIMENSION}_g_${DPR_LLM}_e_${EVAL_LLM}.jsonl"

        echo "Running evaluation with SPLIT=${SPLIT}, DIMENSION=${DIMENSION}, DPR_LLM=${DPR_LLM}, EVAL_LLM=${EVAL_LLM}"

        python benchmark_framework/src/eval.py \
            --dprs_path "$DPR_PATH" \
            --corpus_path "$CORPUS_PATH" \
            --output_path "$OUTPUT_PATH" \
            --model_name "$EVAL_LLM" \
            --dimension "$DIMENSION" \
            --max_workers "$MAX_WORKERS"
      done
    done
  done
done