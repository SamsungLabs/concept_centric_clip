# Implementation of the paper
**"No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models"**

H. X. Pham, D. Hoffmann, R. Guerrero and B. Martinez, CVPR 2026.

# Environment
Our code is developed from the common Open_CLIP github repo, so if you have an exising environment prepared for Open_CLIP, our code should work out of the box. Otherwise, please install the Python libraries from
```
pip install -r requirements-training.txt
```

# Data preparation
We train C2LIP on the CC3m dataset, with new captions from DreamLIP. Follow the instructions in "data_processing/README.md" to prepare the data, ready for training.


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

# Evaluation
The resulting model has the same architecture as well as forward pass as the standard SigLIP model, so you can reuse existing evaluation pipelines.

If you would like to get the checkpoints from the paper, please reach out to the authors.

#
If you find our work useful, please cite
```
@inproceedings{c2lip_cvpr26,
  title   = {No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models},
  author  = {Hai Xuan Pham and David Hoffmann and Ricardo Guerrero and Brais Martinez},
  booktitle = {CVPR},
  year    = {2026}
}
```