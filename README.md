
## Training

To train a CLIP model, first, we need to generate the BLIP2 captions. For this, please refer to the `captioning/README.md` file.

When the generation is done, and we have our captions as a JSON file, we can use them to train our model. First, install the repository using the following command:
```bash
pip install -e .
```

Then, to train, we can either use the JSON file as-is if the number of images is not too large, or we can use the script `add_captions_to_tar.py` in `captioning/` to add the captions to the tar files.

To launch training on 8 GPUs using a JSON file as input, we can use the following command:

```bash
torchrun --nproc_per_node 8 --nnodes=1 -m training.main \
        --data-path SAVE_PATH/blip2_caption.json --image-dir-path FULL_PATH/ \
        --dataset-type 'json' \
        --workers 4 \
        --batch-size 1024 --lr 5e-6 --wd 0.0 --epochs 10 --warmup 1000 \
        --precision amp \
        --ddp-static-graph --grad-checkpointing \
        --model ViT-B-16-SigLIP --pretrained webli --lock-image \
        --siglip \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --imagenet-val IMAGENET_PATH/images/val/ \
        --use-pseudo-labels --distill-model ViT-B-16-SigLIP --distill-pretrained webli \
        --num-captions 3 \
        --logs ./logs \
        --name 'training-run'
```
with `SAVE_PATH/blip2_caption.json` as the path to the JSON file containing BLIP2 generated captions, and `FULL_PATH/` as the path to the images. The model will be saved at `./logs/training-run/checkpoints`. Also, set `IMAGENET_PATH` as the path to the validation set in case you want to evaluate on zero-shot classification during training; otherwise, remove this argument.

If the captions were packed into a webdataset, replace 
```bash
--train-data SAVE_PATH/blip2_caption.json --images-dir-path FULL_PATH/ \
--dataset-type 'json' \
```
with 
```bash
--train-data 'FULL_PATH/{000000..000100}.tar' \
--dataset-type 'webdataset' --dataset-resampled --train-num-samples NUM_SAMPLES  \
```
with `NUM_SAMPLES` as the number of samples in the webdataset to be specified.

Train on CC3m dataset for 3 epochs

```bash
torchrun --nproc_per_node 4 --nnodes=1 -m training.main \
        --train-data '/home/SERILOCAL/hai.xuanpham/datasets/cc3m_wd/{000000000..000002876}.tar' \
        --dataset-type 'webdataset' --dataset-resampled --train-num-samples 2876999  \
        --workers 4 \
        --batch-size 1024 --lr 5e-7 --wd 0.0 --epochs 3 --warmup 1000 \
        --precision amp \
        --ddp-static-graph --grad-checkpointing \
        --model ViT-B-16-SigLIP --pretrained webli --lock-image \
        --siglip \
        --save-frequency 1 \
        --zeroshot-frequency 1 \
        --use-pseudo-labels --distill-model ViT-B-16-SigLIP --distill-pretrained webli \
        --num-captions 1 \
        --logs ./logs \
        --name 'training-run-cc3m-3ep-lr5e-7'
```