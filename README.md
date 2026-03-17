# Paper
H. X. Pham, D. Hoffmann, R. Guerrero and B. Martinez, **No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models**, CVPR 2026.

# Data preparation
We train C2LIP on CC3m dataset, with new captions from DreamLIP. Follow the instructions in "data_processing/README.md" to prepare the data, ready for training.

# Environment
Our code is derived from the common Open_CLIP github repo, so if you have an exising environment prepared for Open_CLIP, our code should work out of the box. Otherwise, please install the Python libraries from
```
pip install -r requirements-training.txt
```

# Launch training
The scripts are in "scripts" folder. Set the correct paths before you launch the code.
- If you prepare data in Webdataset format
```
cd scripts
bash train_full_model_wds_data.sh
```

- If you prepare data in separate caption PKL & image dirs
```
cd scripts
bash train_full_model_pkl_data.sh
```