#!/bin/bash
#SBATCH --account=miso
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:8
#SBATCH --requeue
#SBATCH --job-name=meta
#SBATCH --mem=500G
#SBATCH --partition=miso
##SBATCH --partition=jag-hi
##SBATCH --nodelist=miso1,miso2,miso3,miso4
#SBATCH --time=14-0
#SBATCH --array=99
#SBATCH --exclude=sphinx[1-2],iliad1,iliad4,jagupard[26-29],jagupard[19-20],miso1
#SBATCH --output=logs/lev_%A_%a.out
#SBATCH --error=logs/lev_%A_%a.err
#SBATCH --open-mode=append

hostname

# 0-19 and 99
for ID in {0..19} 99; do

    if [ $ID -eq 99 ]; then
        EID="null"
        TRAIN_ONLY=False
    else
        EID=$ID
        TRAIN_ONLY=True
    fi

    OUT_DIR="out_dir/test_cfx_fineweb_finetune_short_small_lr/${EID}"
    mkdir -p $OUT_DIR

    srun python -m levanter.main.train_lm --config_path config/llama_small_fineweb_finetune.yaml \
        --out_dir $OUT_DIR \
        --cfx_seed $ID \
        --train_only $TRAIN_ONLY \
        --trainer.wandb.name "fineweb_finetune_f32_${EID}"

done