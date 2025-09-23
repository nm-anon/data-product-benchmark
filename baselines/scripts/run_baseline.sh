# run hybrid search (bm25+embedding)

# TATQA, HybridQA, ConvFinQA
DATA_NAME="ConvFinQA"
# train, dev, test
SPLIT="test"
# sentence-transformers/all-mpnet-base-v2, intfloat/multilingual-e5-large-instruct, ibm-granite/granite-embedding-125m-english, Qwen/Qwen3-Embedding-8B
MODEL="ibm-granite/granite-embedding-125m-english"
# mpnet, e5, granite, qwen
MODEL_SHORT="granite"
# 768, 1024, 4096
EMB_DIM=768

CORPUS_PATH="benchmark_data/${DATA_NAME}/${DATA_NAME}_corpus.json"
DPR_PATH="benchmark_data/${DATA_NAME}/${DATA_NAME}_${SPLIT}.jsonl"
DB_PATH="baselines/data/${DATA_NAME}/${DATA_NAME}_${MODEL_SHORT}.db"
COLLECTION_NAME="dp_benchmark_${DATA_NAME}"
OUTPUT_PATH="baselines/data/${DATA_NAME}/${DATA_NAME}_${SPLIT}_results_${MODEL_SHORT}.json"
EVAL_OUTPUT_PATH="baselines/data/${DATA_NAME}/${DATA_NAME}_${SPLIT}_results_eval_${MODEL_SHORT}.json"

echo "Dataset: ${DATA_NAME} - ${SPLIT}"
echo ""
echo "EMB MODEL: ${MODEL}"
echo "Corpus PATH: ${CORPUS_PATH}"
echo "DPR PATH: ${DPR_PATH}"
echo ""
echo "Milvus DB: ${DB_PATH}"
echo "Output PATH: ${OUTPUT_PATH}"
echo "Eval Output PATH: ${EVAL_OUTPUT_PATH}"

python baselines/src/baseline.py \
    --corpus "$CORPUS_PATH" \
    --dpr "$DPR_PATH" \
    --db "$DB_PATH" \
    --dataset "$DATA_NAME" \
    --collection "$COLLECTION_NAME" \
    --model "$MODEL" \
    --emb_dim "$EMB_DIM" \
    --index-type AUTOINDEX \
    --metric-type IP \
    --top_text 20 \
    --top_table 20 \
    --output_path "$OUTPUT_PATH"

python baselines/src/baseline_eval.py \
    --dpr "$DPR_PATH" \
    --sys_output "$OUTPUT_PATH" \
    --eval_output "$EVAL_OUTPUT_PATH"
