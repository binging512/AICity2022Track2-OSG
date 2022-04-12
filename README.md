# AICity2022Track2-OSG
This is the Part 2 of the BUPT_MCPRL_T2 in AICityChallenge2022 Track2. This is the code of OSG in the ablation study.

Paper Link: TBD 

The code of Part 1 can be found here: https://github.com/dyhBUPT/OMG

## Installation

- Download the repository

```shell
git clone https://github.com/binging512/AICity2022Track2-OSG.git
```

- Install the environment

```shell
pip install -r requirements.txt
```

## Getting Started

1. Split the annotated dataset into 5-fold cross validation dataset

```shell
python misc/split_trainval.py
```

2. Modify your data path or checkpoints path in ```config.py```
3. Train the OSG model

```shell
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python train.py --config configs/Swin+GRU+CLIP+NLP_AUG+COLOR.yaml --valnum 4
```

The training log will be written in ```outputs/METHOD_NAME/METHOD_NAME_fold_N/debug.log```

4. Test the model

```shell
CUDA_VISIBLE_DEVICES=0 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python test.py --config configs/Swin+GRU+CLIP+NLP_AUG+COLOR.yaml --valnum 4
```

## NLP Augment

The Natural Language Augment code can be found in ```nlp```.

1. We use [fanyi.baidu.com](https://fanyi.baidu.com) perform the English-Chinese-English backtranslation

```shell
python nlp/nlp_fix_1.py
```

2. We use [spaCy](https://spacy.io/) to conduct the dependency parsing on the text descriptions

```shell
python nlp/nlp_spacy_2.py
```

3. Then we generate the Color and the Type of the vehicles using voting strategy.

```shell
python nlp/nlp_merge_3.py
```

4. Also, we tried to decouple the appearance and the motion information in the annotation. 

```shell
python nlp/nlp_decouple_4.py
```

## ID

The IDs of the vehicle are all from the annotation for the AICityChallenge2022 Track1 (No code). And we rearrange all the vehicle IDs begin with 0. The rearrange code can be found in ```misc/add_id.py```.

