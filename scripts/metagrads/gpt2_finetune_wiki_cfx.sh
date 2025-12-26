#OUT_DIR="/scr-ssd/sampark/levanter/gpt2_cpt_wiki_cfx_null"
#mkdir -p $OUT_DIR

#PYTHONPATH=src python -m levanter.main.train_lm --config_path config/_gpt2_wikitext.yaml \
#    --out_dir $OUT_DIR \
#    --train_only False \
#    --trainer.wandb.name "gpt2_cpt_wiki_cfx_null"

for ID in {0..19}; do
    if [ $ID -eq 0 ]; then
        EID="null"
        TRAIN_ONLY=False
    else
        EID=$ID
        TRAIN_ONLY=True
    fi

    # CHANGE: where results get saved
    OUT_DIR="/juice5b/scr5b/sampark/marin/results/gpt2_finetune_wiki/cfx_${EID}"
    mkdir -p $OUT_DIR

    PYTHONPATH=src python -m levanter.main.train_lm --config_path config/_gpt2_wikitext.yaml \
        --out_dir $OUT_DIR \
        --cfx_seed $ID \
        --drop_rate 0.05 \
        --train_only $TRAIN_ONLY \
        --trainer.wandb.name "gpt2_finetune_wiki_cfx_${EID}"

done