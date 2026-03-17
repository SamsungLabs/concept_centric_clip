#!/bin/bash

# edit that path to web-dataset root & total number of samples
train_data='/path/to/pkl/file/cc3m_np.pkl'
train_image_dir = '/path/to/image/dir'

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
--images-dir-path $train_image_dir \
--dataset-resampled \
--seed 42 \
--dataset-type cc3m_custom_np_pkl \
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