eval $(ssh-agent -s)
bash infra/babysit-tpu-vm.sh debug -z europe-west4-b -t v5litepod-64 --preemptible -- \
WANDB_API_KEY=$WANDB_API_KEY \
bash levanter/infra/run.sh python \
levanter/src/levanter/main/train_lm.py \
--config_path levanter/config/llama2_1b_adam.yaml  \
--trainer.checkpointer.base_path  gs://marin-us-central2/scratch/kaiyue/checkpoints/debug_mudam/mudam_reduced_to_muon  \
--optimizer.type mudam \
--optimizer.merge_small_dims True \
--optimizer.prefer_input_side True \
--optimizer.use_momentum_in_second_moment False \
--optimizer.use_nesterov_in_second_moment False \
--optimizer.reduce_to_muon False \
--trainer.wandb.name nesterov_shampoo_reduce_to_adam \
--optimizer.block_size 1 \
--optimizer.partition_grads_into_blocks True \
--trainer.num_train_steps 10001 \
--optimizer.warmup 1000 \
--optimizer.shampoo_beta 0.98 \
--optimizer.learning_rate 8e-3 \
--optimizer.adam_lr 8e-3 \
--optimizer.weight_decay 0.1 \
--optimizer.min_lr_ratio 0.0 \
--optimizer.lr_schedule linear \
--optimizer.cooldown 0.0 \
--model.seq_len 8192 \
--model.hidden_dim 768 \
--model.intermediate_dim 3072 \
--model.num_layers 12 \
--model.num_heads 12 \
--model.num_kv_heads 12 \
--trainer.train_batch_size 128 