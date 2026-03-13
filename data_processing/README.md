# This page contains instructions on how to prepare data to train C2LIP. The general process is as below:
1. Download CC3m images, and new captions from [DreamLIP](https://github.com/ant-research/DreamLIP). Please note, that in our work we use the "shortLLA_captions" captions instead of the long ones in the DreamLIP paper.
2. Extract concepts from captions
3. (optional) Pack data as webdataset (useful when transferring data between machines)


## 1. Download data
The new captions are hosted without images on [Huggingface](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions), so in addition, if you will also need to download CC3m images. There are 2 ways to download images:
- Directly from URLs. For this, first you download the "Train_GCC-training.tsv" file from [Conceptual Captions website](https://ai.google.com/research/ConceptualCaptions/download), and run the following command
```
python data_processing/download_image_direct.py --hf-cache-dir [HF_CACHE_DIR] --image-dir [WHERE_TO_SAVE_IMAGE] --cc3m-tsv [PATH_TO_THE_TSV_FILE] --output-pkl [PATH_TO_SAVE_METADATA_PICKLE]
```
- As many URLs are no longer valid, there's another way to get them from a repository hosted on HuggingFace: https://huggingface.co/datasets/pixparse/cc3m-wds. Please run the following command:
```
python data_processing/download_image_cc3m_wds.py --hf-cache-dir [HF_CACHE_DIR] --image-dir [WHERE_TO_SAVE_IMAGE] --output-pkl [PATH_TO_SAVE_METADATA_PICKLE]
```

## 2. Extract nounphrase concepts
In the paper we use a different, proprietary preprocessing pipeline to extract the noun concepts. Here we provide an alternative method, using Phi-3.5-mini-instruct.
Run the following command:
```
python data_processing/extract_nounphrases.py --metadata-pkl-file [PATH_TO_SAVE_METADATA_PICKLE] --gpu [WHICH_GPU_TO_USE] --overwrite-metadata-file
```
This script will add a "nounphrases" attribute to each sample's metadata.


## 3. (optional) Pack data into Webdataset
Run the command
```
python data_processing/pack_data.py --image-root-dir [WHERE_TO_SAVE_IMAGE] --metadata-pkl-file [PATH_TO_SAVE_METADATA_PICKLE] --output-wds-dir [WDS_ROOT_DIR]
```