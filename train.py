"""
@Author: Du Yunhao
@Filename: train.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 9:39
@Discription: Train
"""
import itertools
import os
from tabnanny import check
from collections import defaultdict
import time
import torch
import argparse
from torch import nn
from torch import optim
from datetime import datetime
import torch.nn.functional as F
from os.path import join, exists
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from utils import *
from config import get_default_config
from datasets import CityFlowNLDataset
from models.Clip import clip
from models.model import MultiStreamNetwork
from models.baseline_motion_2d import Baseline_motion_2d
from models.baseline_motion_1d import Baseline_motion_1d
from models.baseline_motion_3d import Baseline_motion_3d
from models.baseline_motion_1d_nlpaug_color_type import baseline_motion_1d_nlpaug_color_type
from models.Res50_clip_motion_1d_nlpaug_color_type_split import Res50_clip_motion_1d_nlpaug_color_type_split
from models.Swin_clip_motion_1d_nlpaug_color_type_split import Swin_clip_motion_1d_nlpaug_color_type_split
from models.Res50_clip_motion_1d_cls_nlpaug_color_type import Res50_clip_motion_1d_cls_nlpaug_color_type
from models.Res50_clip_motion_1d_nlpaug_color_type_id import Res50_clip_motion_1d_nlpaug_color_type_id
from models.Res50_clip_nlpaug_color_type_id import Res50_clip_nlpaug_color_type_id
from models.Swin_clip_motion_1d_nlpaug_color_type_id import Swin_clip_motion_1d_nlpaug_color_type_id
from models.Swin_clip_nlpaug_color_type_id import Swin_clip_nlpaug_color_type_id
from models.Res50_clip_motion_1d_nlpaug_color_type_id_decouple import Res50_clip_motion_1d_nlpaug_color_type_id_decouple
from models.Swin_clip_motion_1d_nlpaug_color_type_id_decouple import Swin_clip_motion_1d_nlpaug_color_type_id_decouple
from loss.losses import BCE_CFNL


def gen_obj_car_id_matrix(car_id):
    flag = 0
    batch_size = car_id.shape[0]
    obj_car_id = torch.eye(batch_size)
    car_id_list = []
    for idx in car_id:
        if not idx in car_id_list:
            car_id_list.append(idx.item())
    id_list = [[i for i,x in enumerate(car_id) if x==y] for y in car_id_list]
    for pair in id_list:
        idx_list = list(itertools.permutations(pair))
        for idx in idx_list:
            if len(idx) < 2:
                continue
            else:
                flag = 1
                obj_car_id[idx[0],idx[1]] = 1
    # if flag == 1:
    #     print(obj_car_id)
    return obj_car_id

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser(description='AICity2022Track2 Training')
parser.add_argument('--config', default='configs/Res50+BERT.yaml', type=str, help='config_file')
parser.add_argument('--valnum', default=0, type=int, help='val_num')
args = parser.parse_args()

cfg = get_default_config()
cfg.merge_from_file(args.config)

val_num = args.valnum
dir_save_all = join(cfg.DATA.ROOT_SAVE, cfg.METHOD_NAME)
dir_save = join(cfg.DATA.ROOT_SAVE, cfg.METHOD_NAME, cfg.METHOD_NAME + '_fold%d' % val_num)
checkpoints_save = join(cfg.DATA.ROOT_SAVE, cfg.METHOD_NAME, cfg.METHOD_NAME + '_fold%d' % val_num,'checkpoints')

if not exists(dir_save):
    os.makedirs(dir_save)
    os.makedirs(checkpoints_save)


sys.stdout = Logger(join(dir_save, 'log.txt'))
logger = get_logger(logger_name=cfg.METHOD_NAME,save_to_dir=os.path.join(cfg.DATA.ROOT_SAVE, cfg.METHOD_NAME, cfg.METHOD_NAME + '_fold%d' % val_num))
logger.info(time.asctime(time.localtime(time.time())))
print(cfg)

logger.info("======================Program Initializing=======================")
dataset_train = CityFlowNLDataset(
    cfg_data=cfg.DATA,
    transform=get_transforms(cfg, True),
    mode='train',
    val_num=val_num
)
dataset_val = CityFlowNLDataset(
    cfg_data=cfg.DATA,
    transform=get_transforms(cfg, False),
    mode='val',
    val_num=val_num
)
dataloader_train = DataLoader(
    dataset_train,
    batch_size=cfg.TRAIN.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.TRAIN.NUM_WORKERS,
    drop_last=True
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=cfg.TRAIN.BATCH_SIZE*8,
    shuffle=False,
    num_workers=cfg.TRAIN.NUM_WORKERS
)

# model = Baseline_motion_1d(cfg.MODEL)
# model = Baseline_motion_2d(cfg.MODEL)
# model = Baseline_motion_3d(cfg.MODEL)
# model = MultiStreamNetwork(cfg.MODEL)
if cfg.METHOD_NAME in ["Res50+GRU+BERT+NLP_AUG+COLOR+TYPE",
                        "Res50+GRU+BERT+NLP_AUG+COLOR",
                        "Res50+GRU+BERT+NLP_AUG"]:
    logger.info('Loading Model: baseline_motion_1d_nlpaug_color_type')
    model = baseline_motion_1d_nlpaug_color_type(cfg.MODEL)
elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+TYPE',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+RECT',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT+ENHANCE',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+TYPE+WEIGHT',]:
    logger.info('Loading Model: Res50_clip_motion_1d_nlpaug_color_type')
    model = Res50_clip_motion_1d_nlpaug_color_type_split(cfg.MODEL)
    
elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+ID',
                         'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS']:
    logger.info('Loading Model: Res50_clip_motion_1d_nlpaug_color_type_id')
    model = Res50_clip_motion_1d_nlpaug_color_type_id(cfg.MODEL)

elif cfg.METHOD_NAME in ["Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS",]:
    logger.info('Loading Model: Swin_clip_motion_1d_nlpaug_color_type_id')
    model = Swin_clip_motion_1d_nlpaug_color_type_id(cfg.MODEL)

elif cfg.METHOD_NAME in ['Swin+CLIP+NLP_AUG+COLOR+ID_CLS']:
    logger.info('Loading Model: Swin_clip_nlpaug_color_type_id')
    model = Swin_clip_nlpaug_color_type_id(cfg.MODEL)

elif cfg.METHOD_NAME in ['Swin+GRU+CLIP+NLP_AUG+COLOR',
                            'Swin+GRU+CLIP+NLP_AUG+COLOR+TYPE',
                            'Swin+GRU+CLIP+NLP_AUG+COLOR+TYPE_2',
                            'Swin+GRU+CLIP+NLP_AUG+COLOR+TYPE_SPLIT',]:
    logger.info('Loading Model: Swin_clip_motion_1d_nlpaug_color_type')
    model = Swin_clip_motion_1d_nlpaug_color_type_split(cfg.MODEL)
elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w0.0',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w0.1',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w0.5',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w1.0']:
    logger.info('Loading Model: Res50_clip_motion_1d_cls_nlpaug_color_type')
    model = Res50_clip_motion_1d_cls_nlpaug_color_type(cfg.MODEL)
elif cfg.METHOD_NAME in ['Res50+CLIP+NLP_AUG+COLOR+ID',
                         'Res50+CLIP+NLP_AUG+COLOR+ID_CLS']:
    logger.info('Loading Model: Res50_clip_nlpaug_color_id')
    model = Res50_clip_nlpaug_color_type_id(cfg.MODEL)

elif cfg.METHOD_NAME in ["Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE"]:
    logger.info('Loading Model: Res50_clip_motion_1d_nlpaug_color_id_decouple')
    model = Res50_clip_motion_1d_nlpaug_color_type_id_decouple(cfg.MODEL)

elif cfg.METHOD_NAME in ["Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE"]:
    logger.info('Loading Model: Swin_clip_motion_1d_nlpaug_color_id_decouple')
    model = Swin_clip_motion_1d_nlpaug_color_type_id_decouple(cfg.MODEL)

model.cuda()
model = nn.DataParallel(model)
tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.BASE_LR)
bceloss = BCE_CFNL()

global_step = 0
best_r1 = 0
best_r5 = 0
best_r10 = 0 
best_mrr = 0
best_epoch = 0
for epoch in range(cfg.TRAIN.MAX_EPOCH):
    logger.info("======================Training Start=======================")
    # evaluate(model, tokenizer, dataloader_val, epoch)
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lr = get_lr(cfg.TRAIN, epoch)
    set_lr(optimizer, lr)
    progress = ProgressMeter(
        num_batches=len(dataloader_train) * cfg.TRAIN.ONE_EPOCH_REPEAT,
        meters=[batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )
    end = time.time()

    for _ in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        for idx, batch in enumerate(dataloader_train):
            optimizer.zero_grad()
            # For BERT tokenizer
            if not 'CLIP' in cfg.METHOD_NAME:
                tokens = tokenizer.batch_encode_plus(
                    batch['text'],
                    padding='longest',
                    return_tensors='pt'
                )
                # Split the text
                text_split_1,text_split_2,text_split_3 = [],[],[]
                for text in batch['text']:
                    text_split_1.append(text[0])
                    text_split_2.append(text[1])
                    text_split_3.append(text[2])
                tokens_split_1 = tokenizer.batch_encode_plus(
                    text_split_1,padding='longest',return_tensors='pt'
                )
                tokens_split_2 = tokenizer.batch_encode_plus(
                    text_split_2,padding='longest',return_tensors='pt'
                )
                tokens_split_3 = tokenizer.batch_encode_plus(
                    text_split_3,padding='longest',return_tensors='pt'
                )
                # For input data
                input_ = {
                    'crop': batch['crop'].cuda(),
                    'text_input_ids': tokens['input_ids'].cuda(),
                    'text_attention_mask': tokens['attention_mask'].cuda(),
                    # Add the split text
                    'text_input_ids_s1': tokens_split_1['input_ids'].cuda(),
                    'text_attention_mask_s1': tokens_split_1['attention_mask'].cuda(),
                    'text_input_ids_s2': tokens_split_2['input_ids'].cuda(),
                    'text_attention_mask_s2': tokens_split_2['attention_mask'].cuda(),
                    'text_input_ids_s3': tokens_split_3['input_ids'].cuda(),
                    'text_attention_mask_s3': tokens_split_3['attention_mask'].cuda(),
                    #####################
                    'motion': batch['motion'].cuda(),
                    'motion_boxes':batch['motion_boxes'].cuda(),
                    # 'motion_3d': batch['motion_3d'].cuda(),
                    'vote_color':batch['vote_color'].cuda(),
                    'vote_obj':batch['vote_obj'].cuda(),
                    'weight_color': batch['weight_color'].cuda(),
                    'weight_obj':batch['weight_obj'].cuda(),
                }
            # For CLIP tokenizer
            else:
                tokens = clip.tokenize(batch['text'],context_length=77)
                # Split the text
                text_split_1,text_split_2,text_split_3 = [],[],[]
                for text in batch['text']:
                    text_split_1.append(text[0])
                    text_split_2.append(text[1])
                    text_split_3.append(text[2])
                tokens_split_1 = clip.tokenize(text_split_1,context_length=77)
                tokens_split_2 = clip.tokenize(text_split_2,context_length=77)
                tokens_split_3 = clip.tokenize(text_split_3,context_length=77)

                tokens_app = clip.tokenize(batch['appearance'])
                tokens_mot = clip.tokenize(batch['motion_nl'])

                # For input data
                input_ = {
                    'crop': batch['crop'].cuda(),
                    'text_input_ids': tokens.cuda(),
                    # Add the split text
                    'text_input_ids_s1': tokens_split_1.cuda(),
                    'text_input_ids_s2': tokens_split_2.cuda(),
                    'text_input_ids_s3': tokens_split_3.cuda(),
                    #####################
                    'motion': batch['motion'].cuda(),
                    'motion_boxes':batch['motion_boxes'].cuda(),
                    # 'motion_3d': batch['motion_3d'].cuda(),
                    'vote_color':batch['vote_color'].cuda(),
                    'vote_obj':batch['vote_obj'].cuda(),
                    'weight_color': batch['weight_color'].cuda(),
                    'weight_obj':batch['weight_obj'].cuda(),
                    # Decouple the appearance and motion
                    'appearance': tokens_app.cuda(),
                    'motion_nl': tokens_mot.cuda(),
                }
            
            data_time.update(time.time() - end)
            # pairs, tau = model(input_)
            if cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+ID','Res50+CLIP+NLP_AUG+COLOR+ID',
                                    'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS','Res50+CLIP+NLP_AUG+COLOR+ID_CLS',
                                    'Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS','Swin+CLIP+NLP_AUG+COLOR+ID_CLS',
                                    'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE','Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE']:
                pairs, obj_color, obj_type, obj_id, tau, features_text_list = model(input_)
            elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT','Swin+GRU+CLIP+NLP_AUG+COLOR']:
                pairs, obj_color, obj_type, tau, features_text_list = model(input_)

            weight_list = gen_weight(features_text_list)
            tau = tau.mean().exp()
            # generate soft-label using id
            car_id = batch['car_id'].cuda()
            obj_car_id = gen_obj_car_id_matrix(car_id)

            loss = 0
            pair_num = 0
            ii = 0
            for (visual_embedding,language_embedding) in pairs:
                pair_num += 1
                similarity_i2t = (tau * visual_embedding) @ language_embedding.t()
                similarity_t2i = similarity_i2t.t()
                if cfg.TRAIN.ID == False:
                    if cfg.TRAIN.LOSS_WEIGHT == False:
                        infoNCE_i2t = F.cross_entropy(similarity_i2t, torch.arange(cfg.TRAIN.BATCH_SIZE).cuda())
                        infoNCE_t2i = F.cross_entropy(similarity_t2i, torch.arange(cfg.TRAIN.BATCH_SIZE).cuda())
                    else:
                        infoNCE_i2t = F.binary_cross_entropy_with_logits(similarity_i2t, weight_list[int(ii/3)].cuda())
                        infoNCE_t2i = F.binary_cross_entropy_with_logits(similarity_t2i, weight_list[int(ii/3)].cuda())
                else:
                    # if ii%3 == 0:
                    #     infoNCE_i2t = F.binary_cross_entropy_with_logits(similarity_i2t,obj_car_id.cuda())
                    #     infoNCE_t2i = F.binary_cross_entropy_with_logits(similarity_t2i,obj_car_id.cuda())
                    #     pass
                    # else:
                    infoNCE_i2t = F.cross_entropy(similarity_i2t, torch.arange(cfg.TRAIN.BATCH_SIZE).cuda())
                    infoNCE_t2i = F.cross_entropy(similarity_t2i, torch.arange(cfg.TRAIN.BATCH_SIZE).cuda())
                loss += (infoNCE_i2t + infoNCE_t2i) / 2
                ii += 1
            if cfg.TRAIN.MOTION == True:
                loss = loss/(pair_num/2)
            else:
                loss = loss/(pair_num/3)
            
            # Add color and type information
            if cfg.TRAIN.COLOR == True:
                loss_color_bce = bceloss(obj_color, input_['vote_color'],input_['weight_color'])
                loss = loss + loss_color_bce
            if cfg.TRAIN.TYPE == True:
                loss_type_bce = bceloss(obj_type,input_['vote_obj'],input_['weight_obj'])
                loss = loss + loss_type_bce
            if cfg.TRAIN.RECT == True:
                features_obj_main,features_text_obj_main = pairs[0]
                features_motion_main,features_text_motion_main = pairs[1]
                loss_rect = torch.abs(torch.sum(features_obj_main*features_motion_main, dim=1)) \
                          + torch.abs(torch.sum(features_text_obj_main*features_text_motion_main, dim=1))
                loss += torch.mean(10*loss_rect)
            if cfg.TRAIN.ID == True:
                loss += F.cross_entropy(obj_id, car_id)

            losses.update(loss.item(), cfg.TRAIN.BATCH_SIZE)
            loss.backward()
            global_step += 1
            optimizer.step()
            batch_time.update(time.time() - end)
            if idx % cfg.TRAIN.PRINT_FREQ == 0:
                logger.info("Training | Epoch:[{}/{}] | Batch:[{}/{}] | loss:{} | lr:{}".format(
                    epoch,
                    cfg.TRAIN.MAX_EPOCH,
                    idx,
                    len(dataloader_train),
                    loss.item(),
                    lr)
                )
                # progress.display(global_step % (len(dataloader_train) * cfg.TRAIN.ONE_EPOCH_REPEAT))
    if epoch% 5 == 0:
        r1_list,r5_list,r10_list,mrr_list = evaluate(cfg, model, tokenizer, dataloader_val, epoch, logger)
    for ii in range(len(r1_list)):
        r1 = r1_list[ii]
        r5 = r5_list[ii]
        r10 = r10_list[ii]
        mrr = mrr_list[ii]
        if mrr>best_mrr:
            best_pair = ii
            best_mrr = mrr
            best_r1 = r1
            best_r5 = r5
            best_r10 = r10
            best_epoch = epoch
            path_save = join(dir_save, 'Best_checkpoints.pth')
            torch.save(
                {
                    'epoch': epoch,
                    'global_step': global_step,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                path_save
            )
    logger.info('Best checkpoints: epoch:{}, pair:{}, MRR:{:.4f}'.format(best_epoch,best_pair,best_mrr))
    logger.info('Best results: {:.4f} | {:.4f} | {:.4f}'.format(best_r1,best_r5,best_mrr))

    if epoch % 200 == 0:
        path_save = join(dir_save, "checkpoints", 'checkpoint_epoch%d.pth' % epoch)
        torch.save(
            {
                'epoch': epoch,
                'global_step': global_step,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
            path_save
        )
    
