"""
@Author: Du Yunhao
@Filename: config.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 9:27
@Discription: config
"""
from yacs.config import CfgNode as CN

_C = CN()
_C.METHOD_NAME = 'baseline'

_C.DATA = CN()
_C.DATA.ROOT_DATA = '/data0/CityFlow_NL'
_C.DATA.SIZE = 288
_C.DATA.MOTION_SIZE = 512
_C.DATA.ROOT_SAVE = '/home/zby/AICity2022Track2-OSG/outputs'
_C.DATA.TEST_SAVE = '/home/zby/AICity2022Track2-OSG/test'

_C.MODEL = CN()
_C.MODEL.BERT_NAME = 'bert-base-uncased'
_C.MODEL.FREEZE_TEXT_ENCODER = True
_C.MODEL.EMBED_DIM = 1024
_C.MODEL.SPLIT_TEXT = False

_C.TRAIN = CN()
_C.TRAIN.MAX_EPOCH = 40
_C.TRAIN.ONE_EPOCH_REPEAT = 30
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_WORKERS = 6
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.BASE_LR = 0.01
_C.TRAIN.WARMUP_EPOCH = 10
_C.TRAIN.WARMUP_START_LR = 1e-5
_C.TRAIN.COSINE_END_LR = 0.0
_C.TRAIN.LOSS_WEIGHT = False
_C.TRAIN.COLOR = False
_C.TRAIN.TYPE = False
_C.TRAIN.RECT = False
_C.TRAIN.MOTION_CLS = False
_C.TRAIN.MOTION_WEIGHT = 0.0
_C.TRAIN.ID = False
_C.TRAIN.MOTION = False
_C.TRAIN.DECOUPLE = False


def get_default_config():
    return _C.clone()