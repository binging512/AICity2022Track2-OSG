"""
@Author: Du Yunhao
@Filename: utils.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 10:36
@Discription: utils
"""
import sys
import math
import torch
import numpy as np
from datetime import datetime
from torchvision import transforms
import io
import logging
import os
import colorlog
from models.Clip import clip

def get_transforms(cfg, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(0.8, 1)),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.DATA.SIZE, cfg.DATA.SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

def get_motion_transforms(cfg, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.MOTION_SIZE, scale=(0.8, 1)),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.MOTION_SIZE, cfg.MOTION_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

def get_lr(cfg_train, curr_epoch):
    if curr_epoch < cfg_train.WARMUP_EPOCH:
        return (
            cfg_train.WARMUP_START_LR
            + (cfg_train.BASE_LR - cfg_train.WARMUP_START_LR)
            * curr_epoch
            / cfg_train.WARMUP_EPOCH
        )
    else:
        return (
            cfg_train.COSINE_END_LR
            + (cfg_train.BASE_LR - cfg_train.COSINE_END_LR)
            * (
                math.cos(
                    math.pi * (curr_epoch - cfg_train.WARMUP_EPOCH) / (cfg_train.MAX_EPOCH - cfg_train.WARMUP_EPOCH)
                )
                + 1.0
            )
            * 0.5
        )

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def evaluate(cfg,model, tokenizer, dataloader, epoch,logger):
    logger.info("=====================Evaluation Start=====================")
    model.eval()
    FeaturesCrop, FeaturesText = [], []
    Motion_scores,Motion_cls = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
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

                input_ = {
                    'crop': batch['crop'].cuda(),
                    'text_input_ids': tokens['input_ids'].cuda(),
                    # Add the split text
                    'text_input_ids_s1': tokens_split_1['input_ids'].cuda(),
                    'text_attention_mask_s1': tokens_split_1['attention_mask'].cuda(),
                    'text_input_ids_s2': tokens_split_2['input_ids'].cuda(),
                    'text_attention_mask_s2': tokens_split_2['attention_mask'].cuda(),
                    'text_input_ids_s3': tokens_split_3['input_ids'].cuda(),
                    'text_attention_mask_s3': tokens_split_3['attention_mask'].cuda(),
                    #####################
                    'text_attention_mask': tokens['attention_mask'].cuda(),
                    'motion': batch['motion'].cuda(),
                    'motion_boxes':batch['motion_boxes'].cuda(),
                    # 'motion_3d': batch['motion_3d'].cuda()
                    'vote_color': batch['vote_color'].cuda(),
                    'vote_obj': batch['vote_obj'].cuda(),
                }
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
                    'appearance': tokens_app.cuda(),
                    'motion_nl': tokens_mot.cuda(),
                }
                # if cfg.TRAIN.MOTION_CLS == False:
                #     pairs, obj_color, obj_type, tau, features_text_raw = model(input_)
                # else:
                #     pairs, obj_color, obj_type, tau, motion_cls, features_text_raw = model(input_)
                #     Motion_scores.append(motion_cls)
                #     Motion_cls.append(batch['motion_cls'].cuda())
                if cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+ID','Res50+CLIP+NLP_AUG+COLOR+ID',
                                        'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS','Res50+CLIP+NLP_AUG+COLOR+ID_CLS',
                                        'Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS','Swin+CLIP+NLP_AUG+COLOR+ID_CLS',
                                        'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE','Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE']:
                    pairs, obj_color, obj_type, obj_id, tau, features_text_list = model(input_)
                elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT','Swin+GRU+CLIP+NLP_AUG+COLOR']:
                    pairs, obj_color, obj_type, tau, features_text_list = model(input_)
                # pairs, obj_color, obj_type, obj_id, tau, features_text_list = model(input_)

                if len(FeaturesCrop)!= len(pairs):
                    FeaturesCrop = [ [] for x in range(len(pairs))]
                    FeaturesText = [ [] for x in range(len(pairs))]
                for ii, pair in enumerate(pairs):
                    features_img,features_text = pair
                    FeaturesCrop[ii].append(features_img)
                    FeaturesText[ii].append(features_text)

            # features_crop, features_text = pairs[2]
            # FeaturesCrop.append(features_crop)
            # FeaturesText.append(features_text)
    tau = tau.mean().exp()

    FeaturesCrop = [torch.cat(x, dim=0) for x in FeaturesCrop]
    FeaturesText = [torch.cat(x, dim=0) for x in FeaturesText]
    if cfg.TRAIN.MOTION_CLS == True:
        Motion_scores = torch.cat(Motion_scores,dim=0)
        Motion_scores = torch.sigmoid(Motion_scores)    # using the sigmoid
        Motion_cls = torch.cat(Motion_cls,dim=0)
        motion_pred = torch.argmax(Motion_scores,dim=1)
        motion_target = torch.argmax(Motion_cls,dim=1)
        correct_num = 0
        for i in range(motion_pred.shape[0]):
            if motion_pred[i] == motion_target[i]:
                correct_num += 1
        print("Correct Rate: {:.4f} ".format(correct_num/motion_pred.shape[0]))

        motion_scores_add = Motion_scores @ Motion_cls.t()
        # print(motion_scores_add.shape)
        # print(Motion_scores)
        # print(Motion_cls)
        # print(Motion_cls)
        # print(motion_scores_add)
    similarity = []
    for ii in range(len(pairs)):
        sim = FeaturesText[ii] @ (tau * FeaturesCrop[ii]).t()
        similarity.append(sim)
        if cfg.TRAIN.MOTION_CLS == True:
            sim_01 = 0.1*motion_scores_add.t()
            sim_05 = 0.5*motion_scores_add.t()
            sim_10 = 1.0*motion_scores_add.t()
            similarity.append(sim_01+sim)
            similarity.append(sim_05+sim)
            similarity.append(sim_10+sim)
    sim_total = 0
    for sim in similarity:
        sim_total += sim
    sim_total = sim_total/len(similarity)
    similarity.append(sim_total)

    gt = torch.arange(similarity[0].shape[0])
    r1_list,r5_list,r10_list,mrr_list = [],[],[],[]
    for sim in similarity:
        r1, r5, r10, mrr = evaluate_recall_mrr(sim, gt)
        r1_list.append(r1)
        r5_list.append(r5)
        r10_list.append(r10)
        mrr_list.append(mrr)
    # print(datetime.now())
    logger.info(datetime.now())
    for ii,sim in enumerate(similarity):
        logger.info('{}th epoch: Pair {} | R@1 {} | R@5 {} | R@10 {} | MRR {}'.format(epoch, ii, r1_list[ii], r5_list[ii], r10_list[ii], mrr_list[ii]))
    # print('{}th epoch: R@1 {} | R@5 {} | R@10 {} | MRR {}'.format(epoch, r1, r5, r10, mrr))
    logger.info("=====================Evaluation Complete======================")
    return r1_list,r5_list,r10_list,mrr_list

def evaluate_batch(model, tokenizer, dataloader, epoch):
    model.eval()
    R1 = AverageMeter('Recall@1', ':6.2f')
    R5 = AverageMeter('Recall@5', ':6.2f')
    R10 = AverageMeter('Recall@10', ':6.2f')
    MRR = AverageMeter('MRR', ':6.2f')
    progress = ProgressMeter(
        num_batches=len(dataloader),
        meters=[R1, R5, R10, MRR],
        prefix="Test Epoch: [{}]".format(epoch)
    )
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_size = batch['crop'].size(0)
            tokens = tokenizer.batch_encode_plus(
                batch['text'],
                padding='longest',
                return_tensors='pt'
            )
            input_ = {
                'crop': batch['crop'].cuda(),
                'text_input_ids': tokens['input_ids'].cuda(),
                'text_attention_mask': tokens['attention_mask'].cuda()
            }
            features_crop, features_text, tau = model(input_)
            tau = tau.mean().exp()
            similarity_t2i = features_text @ (tau * features_crop).t()
            r1, r5, r10, mrr = evaluate_recall_mrr(similarity_t2i, torch.arange(batch_size).cuda())
            R1.update(r1, batch_size)
            R5.update(r5, batch_size)
            R10.update(r10, batch_size)
            MRR.update(mrr, batch_size)
            progress.display(idx)

def evaluate_recall_mrr(sim, gt):
    r1, r5, r10, mrr = 0, 0, 0, 0
    batch_size = gt.shape[0]
    for row, label in zip(sim, gt):
        idx = row.argsort(descending=True).tolist()
        rank = idx.index(label)
        if rank < 1:
            r1 += 1
        if rank < 5:
            r5 += 1
        if rank < 10:
            r10 += 1
        mrr += 1.0 / (rank + 1)
    return r1 / batch_size, r5 / batch_size, r10 / batch_size, mrr / batch_size

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class TqdmToLogger(io.StringIO):
    logger = None
    level = None
    buf = ''

    def __init__(self):
        super(TqdmToLogger, self).__init__()
        self.logger = get_logger('tqdm')

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.info(self.buf)

def get_logger(logger_name='default', debug=False, save_to_dir=None):
    if debug:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(pathname)s[%(lineno)d]:'
            '%(funcName)s - '
            '%(message)s'
        )
    else:
        log_format = (
            '%(asctime)s - '
            '%(levelname)s : '
            '%(name)s - '
            '%(message)s'
        )
    bold_seq = '\033[1m'
    colorlog_format = f'{bold_seq} %(log_color)s {log_format}'
    colorlog.basicConfig(format=colorlog_format, datefmt='%y-%m-%d %H:%M:%S')
    logger = logging.getLogger(logger_name)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if save_to_dir is not None:
        fh = logging.FileHandler(os.path.join(save_to_dir, 'debug.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # fh = logging.FileHandler(
        #     os.path.join(save_to_dir, 'warning.log'))
        # fh.setLevel(logging.WARNING)
        # formatter = logging.Formatter(log_format)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

        # fh = logging.FileHandler(os.path.join(save_to_dir, 'error.log'))
        # fh.setLevel(logging.ERROR)
        # formatter = logging.Formatter(log_format)
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)

    return logger


def gen_weight(features_text_list):
    # weight_num = len(features_text_list)
    similarity_list = []
    for features_text in features_text_list:
        similarity = features_text @ features_text.t()
        score_max,_ = torch.max(similarity,dim=1)
        # similarity = torch.nn.functional.normalize(similarity,dim=1)
        similarity = similarity/score_max
        similarity_list.append(similarity)

    return similarity_list


if __name__ == '__main__':
    print(evaluate_recall_mrr(
        sim=torch.tensor([
            [1, 0.5, 0],
            [0, 0.5, 1],
            [1, 0, .5]
        ]).numpy(),
        gt=torch.tensor([0, 0, 1]).numpy()
    ))