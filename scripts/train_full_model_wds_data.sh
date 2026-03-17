#!/bin/bash

# edit that path to web-dataset root & total number of samples
train_data='/path/to/cc3m_np_wds/{00000..01411}.tar'
train_num_samples=2823019

# training hyperparameters
lr=1e-05
weight_decay=0.1
bs=96
epochs=5
npc_loss_scale=1.
xac_loss_scale=0.01

cd ..

output_name=C2LIP-lr-${lr}-wd-${weight_decay}-bs-${bs}-ep-${epochs}-npc-loss-scale-${npc_loss_scale}-xac-loss-scale-${xac_loss_scale}

torchrun --nproc_per_node 8 \
main.py \
--train-data $train_data \
--train-num-samples $train_num_samples \
--dataset-resampled \
--seed 42 \
--dataset-type cc3m_custom_np_wds \
--save-frequency 1 \
--report-to tensorboard \
--warmup 50 \
--batch-size $bs \
--lr $lr \
--wd $weight_decay \
--precision bf16 \
--epochs $epochs \
--workers 4 \
--pretrained webli \
--model ViT-B-16-SigLIP \
--siglip \
--beta1 0.9 \
--beta2 0.98 \
--eps 1e-06 \
--output-tokens \
--npc-loss \
--npc-loss-scale $npc_loss_scale \
--xac-loss \
--xac-loss-scale $xac_loss_scale \
--name $output_name