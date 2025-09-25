# step 4: refinement

# TATQA, HybridQA, ConvFinQA
DATA_NAME="TATQA"
SPLITS=("train" "dev" "test")

for SPLIT in "${SPLITS[@]}"; do
    python src/refinement.py \
        --base_dir "data/output/${DATA_NAME}/${SPLIT}/" \
        --dataset "$DATA_NAME" \
        --split "$SPLIT" \
        --alignment_cut_off 0.5
done