# 0-19 and 99
for ID in {0..39}; do

    if [ $ID -eq 99 ]; then
        EID="null"
        TRAIN_ONLY=False
    else
        EID=$ID
        TRAIN_ONLY=True
    fi

    OUT_DIR="out_dir/test_cfx_fineweb_debug_Aug8/${EID}"
    mkdir -p $OUT_DIR

    python -m levanter.main.train_lm --config_path config/llama_small_fineweb_debug.yaml \
        --out_dir $OUT_DIR \
        --cfx_seed $ID \
        --train_only $TRAIN_ONLY \
        --trainer.wandb.name "fineweb_debug_Aug8_${EID}"

done