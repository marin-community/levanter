#!/bin/bash
#SBATCH --account=miso
#SBATCH --cpus-per-task=120
#SBATCH --gres=gpu:8
#SBATCH --requeue
#SBATCH --job-name=meta
#SBATCH --mem=500G
#SBATCH --partition=miso
##SBATCH --partition=jag-hi
#SBATCH --nodelist=miso4
#SBATCH --time=14-0
#SBATCH --array=0
#SBATCH --exclude=sphinx[1-2],iliad1,iliad4,jagupard[26-29],jagupard[19-20],miso1
#SBATCH --output=logs/lev_%A_%a.out
#SBATCH --error=logs/lev_%A_%a.err
#SBATCH --open-mode=append

hostname

srun python -m levanter.main.train_lm --config_path config/a_gpt2_fast.yaml