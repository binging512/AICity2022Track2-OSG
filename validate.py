"""
@Author: Du Yunhao
@Filename: validate.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/10 14:11
@Discription: validate
"""
import os
import torch
from torch import nn
from os.path import join
from collections import OrderedDict
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils import *
from config import get_default_config
from datasets import CityFlowNLDataset
from models.model import MultiStreamNetwork

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cfg = get_default_config()
cfg.merge_from_file('configs/Res50+BERT_5.yaml')
val_num = 0
root_ckpt = '/data/dyh/checkpoints/AICity2022Track2'

dataset_val = CityFlowNLDataset(
    cfg_data=cfg.DATA,
    transform=get_transforms(cfg, False),
    mode='val',
    val_num=val_num
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=cfg.TRAIN.BATCH_SIZE * 10,
    shuffle=False,
    num_workers=cfg.TRAIN.NUM_WORKERS
)
model = MultiStreamNetwork(cfg.MODEL)
model.cuda()
model = nn.DataParallel(model)
tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

checkpoint = torch.load(join(root_ckpt, 'Res50+BERT+1kEpoch_2/checkpoint_epoch890.pth'))
model.load_state_dict(checkpoint['state_dict'])

evaluate(model, tokenizer, dataloader_val, '*')