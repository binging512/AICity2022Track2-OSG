"""
@Author: Du Yunhao
@Filename: tmp.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 9:44
@Discription: tmp
"""
import json
import sys
import math
import torch
import random
import numpy as np
from torch import nn
from torch import optim
from datetime import datetime
import torch.nn.functional as F
from torchvision.models import resnet50
from transformers import BertTokenizer, BertModel
from utils import *
from config import get_default_config
from models.SENet import se_resnext50_32x4d
from models.model import MultiStreamNetwork





if __name__ == '__main__':
    # cfg = get_default_config()
    # model = MultiStreamNetwork(cfg.MODEL)
    # ckpt = torch.load('/data/dyh/checkpoints/AICity2022Track2/motion_SE_NOCLS_nlpaug_288.pth')
    # state_dict = dict()
    # for key, value in ckpt['state_dict'].items():
    #     if 'vis_backbone.' in key:
    #         state_dict[key.replace('module.vis_backbone', 'encoder_img')] = value
    # model.load_state_dict(state_dict, strict=False)

    json_path = '/home/zby/AICity2022Track2/nlp/train_nlp_aug_color_obj.json'
    data_dict = json.load(open(json_path,'r'))
    total_num = 0
    for k,v in data_dict.items():
        nls = v['nl']
        nls_aug = v['nl_aug']
        for nl in nls:
            if len(nl.split('.'))>=2:
                total_num += 1
        for nl in nls_aug:
            if len(nl.split('.'))>=2:
                total_num += 1
    print(total_num)
