#!/bin/bash

export CUDA_VISIBLE_DEVICES="3,4,5,6"

torchrun --nproc_per_node 4 --rdzv-endpoint localhost:51395 -m training.main \
        --train-data '/home/SERILOCAL/hai.xuanpham/datasets/cc3m_wd/{000000000..000002876}.tar' \
        --dataset-type 'webdataset' --dataset-resampled --train-num-samples 2876999  \
        --workers 4 \
        --batch-size 256 --lr 5e-7 --wd 0.0 --epochs 3 --warmup 1000 \
        --precision amp \
        --ddp-static-graph --grad-checkpointing \
        --model ViT-B-16-SigLIP --pretrained webli \
        --siglip \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --num-captions 1 \
        --name 'training-run-cc3m-3ep-lr5e-7-finetune'