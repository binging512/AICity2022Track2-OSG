import os
import sys
import argparse
import json
import numpy as np
from numpy import isin, linspace
from random import random, sample, uniform, randint
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import *
from config import get_default_config
from models.Clip import clip
from models.baseline_motion_1d_nlpaug_color_type import baseline_motion_1d_nlpaug_color_type
from models.Res50_clip_motion_1d_nlpaug_color_type_split import Res50_clip_motion_1d_nlpaug_color_type_split
from models.Swin_clip_motion_1d_nlpaug_color_type_split import Swin_clip_motion_1d_nlpaug_color_type_split
from models.Res50_clip_motion_1d_cls_nlpaug_color_type import Res50_clip_motion_1d_cls_nlpaug_color_type
from models.Swinv2_clip_motion_1d_nlpaug_color_type_split import Swinv2_clip_motion_1d_nlpaug_color_type_split
from models.Res50_clip_motion_1d_nlpaug_color_type_id import Res50_clip_motion_1d_nlpaug_color_type_id
from models.Res50_clip_nlpaug_color_type_id import Res50_clip_nlpaug_color_type_id
from models.Swin_clip_motion_1d_nlpaug_color_type_id import Swin_clip_motion_1d_nlpaug_color_type_id
from models.Swin_clip_nlpaug_color_type_id import Swin_clip_nlpaug_color_type_id
from models.Res50_clip_motion_1d_nlpaug_color_type_id_decouple import Res50_clip_motion_1d_nlpaug_color_type_id_decouple
from models.Swin_clip_motion_1d_nlpaug_color_type_id_decouple import Swin_clip_motion_1d_nlpaug_color_type_id_decouple

class CityFlow_test(Dataset):
    def __init__(self,cfg_data, transform):
        super(CityFlow_test,self).__init__()
        track_anno = json.load(open('/data0/CityFlow_NL/test_tracks_filtered.json','r'))
        query_anno = json.load(open('/data0/CityFlow_NL/test_nlp_aug_color_obj_decouple.json','r'))
        self.track_uuids = list(track_anno.keys())
        self.query_uuids = list(query_anno.keys())
        self.tracks = list(track_anno.values())
        self.queries = list(query_anno.values())
        self.indexs = list(range(len(self.track_uuids)))
        self.cfg_data = cfg_data
        self.transform = transform

    def __getitem__(self, item):
        index = self.indexs[item]
        track_uuid = self.track_uuids[index]
        query_uuid = self.query_uuids[index]
        track = self.tracks[index]
        query = self.queries[index]
        
        index_frame = len(track['frames']) // 2
        sentence = query['nl']
        rand_sent = sample(sentence,3)
        text, text_ov = '',''
        appearance, motion_nl = '',''
        texts = []
        for nl in rand_sent:
            if len(nl.split(' '))>=27:
                nl = nl.split('.')[0]+'.'
            text += nl
            texts.append(nl)
        turns_num = 0
        for t in query['nl']:
            if 'turn' in t:
                turns_num +=1
        motion_cls = torch.tensor([3-turns_num,turns_num])/len(query['nl'])
        
        for nl in query['nl_other_views']:
            text_ov += nl

        for app in query['appearance']:
            appearance += app

        for mot in query['motion']:
            motion_nl += mot
        
        path_img = os.path.join(self.cfg_data.ROOT_DATA , track['frames'][index_frame])
        img = Image.open(path_img).convert('RGB')
        # Reading Motion map
        path_motion = os.path.join(self.cfg_data.ROOT_DATA, 'motion_map',"{}.jpg".format(track_uuid))
        motion = Image.open(path_motion).convert("RGB")
        frame_shape = img.size
        # box = track['boxes'][index_frame]
        # boxes_num = len(track['boxes'])
        box = track['boxes'][index_frame]
        boxes_num = len(track['boxes_new'])
        index_boxes = linspace(0, boxes_num-1, 16)
        # motion_3d_paths = [join(self.cfg_data.ROOT_DATA,track['frames'][int(i)]) for i in index_boxes]
        # motion_3d_boxes = [track['boxes'][int(i)] for i in index_boxes]

        boxes = [motion_1d_transform(track['boxes'][int(i)],frame_shape, False) for i in index_boxes]
        vote_color, weight_color = get_vote_color(query['vote_color'])
        vote_obj, weight_obj = get_vote_obj(query['vote_obj'])
        # car_id = track['id']

        crop = img.crop([box[0], box[1], box[0] + box[2], box[1] + box[3]])
        crop = self.transform(crop)
        motion = get_motion_transforms(self.cfg_data,False)(motion)

        return {
            'index': index,
            'track_uuids': track_uuid,
            'query_uuids':query_uuid,
            'crop': crop,
            'text': text,
            'text_split': texts,
            'text_ov': text_ov,
            'motion': motion,
            'motion_boxes': torch.tensor(boxes),
            'motion_cls': motion_cls,
            'vote_color': vote_color,
            'weight_color': torch.tensor(weight_color),
            'vote_obj':vote_obj,
            'weight_obj': torch.tensor(weight_obj),
            # 'car_id': torch.tensor(car_id),
            'appearance': appearance,
            'motion_nl': motion_nl,
            # 'motion_3d':motion_3d,
        }
    
    def __len__(self):
        return len(self.indexs)

def motion_1d_transform(boxes, frame_shape,mode):
    W,H = frame_shape
    x, y, w, h = boxes
    if mode == True:
        x_off = randint(-3,3)
        y_off = randint(-3,3)
        w_off = randint(-3,3)
        h_off = randint(-3,3)
        x = min(max(0,x+x_off),W)
        y = min(max(0,y+y_off),H)
        w = min(max(0,w+w_off),W-x)
        h = min(max(0,h+h_off),H-y)

    x_nor = x/W
    y_nor = y/H
    w_nor = w/W
    h_nor = h/H
    return [x_nor,y_nor,w_nor,h_nor]

def get_vote_color(vote_color):
    vote_color_new = []
    for k,v in vote_color.items():
        if not k in ['others']:
            vote_color_new.append(v)
    vote_color = torch.tensor(vote_color_new)
    vote_count = torch.sum(vote_color)
    if vote_count.item() == 0:
        weight_color = 0
    else:
        weight_color = 1
    vote_color = vote_color/(vote_count+1e-5)
    return vote_color,weight_color

def get_vote_obj(vote_obj):
    vote_obj_new = []
    for k,v in vote_obj.items():
        if not k in ['others','car']:
            vote_obj_new.append(v)
    vote_obj = torch.tensor(vote_obj_new)
    vote_count = torch.sum(vote_obj)
    if vote_count.item() == 0:
        weight_obj = 0
    else:
        weight_obj = 1
    vote_obj = vote_obj/(vote_count+1e-5)
    return vote_obj, weight_obj

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

def test(cfg):
    print(cfg)
    print('=====================Test Start=====================')
    dir_save = os.path.join(cfg.DATA.TEST_SAVE, cfg.METHOD_NAME)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    
    dataset_test = CityFlow_test(
        cfg_data=cfg.DATA,
        transform=get_transforms(cfg, False),
        )
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=cfg.TRAIN.BATCH_SIZE*8,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS
        )
    
    if cfg.METHOD_NAME in ["Res50+GRU+BERT+NLP_AUG+COLOR+TYPE",
                        "Res50+GRU+BERT+NLP_AUG+COLOR",
                        "Res50+GRU+BERT+NLP_AUG"]:
        print('Loading Model: baseline_motion_1d_nlpaug_color_type')
        model = baseline_motion_1d_nlpaug_color_type(cfg.MODEL)
    elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+TYPE',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+RECT',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT+ENHANCE',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+TYPE+WEIGHT',]:
        print('Loading Model: Res50_clip_motion_1d_nlpaug_color_type')
        model = Res50_clip_motion_1d_nlpaug_color_type_split(cfg.MODEL)
    elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+ID',
                            'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS']:
        print('Loading Model: Res50_clip_motion_1d_nlpaug_color_type_id')
        model = Res50_clip_motion_1d_nlpaug_color_type_id(cfg.MODEL)
    elif cfg.METHOD_NAME in ["Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS",]:
        print('Loading Model: Swin_clip_motion_1d_nlpaug_color_type_id')
        model = Swin_clip_motion_1d_nlpaug_color_type_id(cfg.MODEL)
    elif cfg.METHOD_NAME in ['Swin+CLIP+NLP_AUG+COLOR+ID_CLS']:
        print('Loading Model: Swin_clip_nlpaug_color_type_id')
        model = Swin_clip_nlpaug_color_type_id(cfg.MODEL)
    elif cfg.METHOD_NAME in ['Swin+GRU+CLIP+NLP_AUG+COLOR',
                                'Swin+GRU+CLIP+NLP_AUG+COLOR+TYPE',
                                'Swin+GRU+CLIP+NLP_AUG+COLOR+TYPE_2',
                                'Swin+GRU+CLIP+NLP_AUG+COLOR+TYPE_SPLIT',]:
        print('Loading Model: Swin_clip_motion_1d_nlpaug_color_type')
        model = Swin_clip_motion_1d_nlpaug_color_type_split(cfg.MODEL)
    elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w0.0',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w0.1',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w0.5',
                                'Res50+GRU+CLIP+NLP_AUG+COLOR+MOTION_CLS_w1.0']:
        print('Loading Model: Res50_clip_motion_1d_cls_nlpaug_color_type')
        model = Res50_clip_motion_1d_cls_nlpaug_color_type(cfg.MODEL)
    elif cfg.METHOD_NAME in ['Swinv2+GRU+CLIP+NLP_AUG+COLOR']:
        print('Loading Model: Swinv2_clip_motion_1d_cls_nlpaug_color_type')
        model = Swinv2_clip_motion_1d_nlpaug_color_type_split(cfg.MODEL)
    elif cfg.METHOD_NAME in ['Res50+CLIP+NLP_AUG+COLOR+ID',
                            'Res50+CLIP+NLP_AUG+COLOR+ID_CLS']:
        print('Loading Model: Res50_clip_nlpaug_color_id')
        model = Res50_clip_nlpaug_color_type_id(cfg.MODEL)
    elif cfg.METHOD_NAME in ["Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE"]:
        print('Loading Model: Res50_clip_motion_1d_nlpaug_color_id_decouple')
        model = Res50_clip_motion_1d_nlpaug_color_type_id_decouple(cfg.MODEL)
    elif cfg.METHOD_NAME in ["Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE"]:
        print('Loading Model: Swin_clip_motion_1d_nlpaug_color_id_decouple')
        model = Swin_clip_motion_1d_nlpaug_color_type_id_decouple(cfg.MODEL)

    model.cuda()
    model = nn.DataParallel(model)
    ckpt_path = '/home/zby/AICity2022Track2/outputs/{}/{}_fold4/Best_checkpoints.pth'.format(cfg.METHOD_NAME,cfg.METHOD_NAME)
    # ckpt_path = '/home/zby/AICity2022Track2/outputs/{}/{}_fold4/checkpoints/checkpoint_epoch600.pth'.format(cfg.METHOD_NAME,cfg.METHOD_NAME)
    print("Loading Checkpoints from: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    Track_uuids,Query_uuids = [], []
    FeaturesCrop = []
    FeaturesText = []

    for idx, batch in enumerate(dataloader_test):
        # print(batch)
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
            # 'vote_color':batch['vote_color'].cuda(),
            # 'vote_obj':batch['vote_obj'].cuda(),
            # 'weight_color': batch['weight_color'].cuda(),
            # 'weight_obj':batch['weight_obj'].cuda(),
            # Decouple the appearance and motion
            'appearance': tokens_app.cuda(),
            'motion_nl': tokens_mot.cuda(),
        }
        with torch.no_grad():
            if cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+ID','Res50+CLIP+NLP_AUG+COLOR+ID',
                                    'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS','Res50+CLIP+NLP_AUG+COLOR+ID_CLS',
                                    'Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS','Swin+CLIP+NLP_AUG+COLOR+ID_CLS',
                                    'Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE','Swin+GRU+CLIP+NLP_AUG+COLOR+ID_CLS+DECOUPLE']:
                pairs, obj_color, obj_type, obj_id, tau, features_text_list = model(input_)
            elif cfg.METHOD_NAME in ['Res50+GRU+CLIP+NLP_AUG+COLOR+SPLIT']:
                pairs, obj_color, obj_type, tau, features_text_list = model(input_)
        
        Track_uuids += batch['track_uuids']
        Query_uuids += batch['query_uuids']

        if len(FeaturesCrop)!= len(pairs):
            FeaturesCrop = [ [] for x in range(len(pairs))]
            FeaturesText = [ [] for x in range(len(pairs))]
        
        for ii, pair in enumerate(pairs):
            features_img,features_text = pair
            FeaturesCrop[ii].append(features_img)
            FeaturesText[ii].append(features_text)
    tau = tau.mean().exp()
    FeaturesCrop = [torch.cat(x, dim=0) for x in FeaturesCrop]
    FeaturesText = [torch.cat(x, dim=0) for x in FeaturesText]

    similarity = []
    for ii in range(len(pairs)):
        sim = FeaturesText[ii] @ (tau * FeaturesCrop[ii]).t()
        similarity.append(sim)
    sim_total = 0
    for sim in similarity:
        sim_total += sim
    sim_total = sim_total/len(similarity)
    similarity.append(sim_total)
    features_dict = {'track_uuids': Track_uuids,
                     'query_uuids': Query_uuids,
                     'feature_track': FeaturesCrop,
                     'feature_text': FeaturesText,
                     'similarity': similarity,
                    }
    torch.save(features_dict,os.path.join(dir_save,'features_best.pth'))

    save_np(features_dict,dir_save)

    results = dict()
    print(similarity[0].shape)
    for i, sim in enumerate(similarity[-1]):
        idx = torch.argsort(sim,descending=True)
        results[Query_uuids[i]]= [Track_uuids[ii] for ii in idx]
        # print(results)
    json.dump(
        results,
        open(os.path.join(dir_save,'results_best.json'),'w'),
        indent=2
    )

def save_np(feature_dict,dir_save):
    if not isinstance(feature_dict,dict):
        feature_dict = torch.load(feature_dict)
    else:
        pass
    Test_uuidImg = feature_dict['track_uuids']
    Test_uuidText = feature_dict['query_uuids']
    Test_FeaturesImg = feature_dict['feature_track'][-1].cpu().numpy()
    Test_FeaturesText = feature_dict['feature_text'][-1].cpu().numpy()
    json.dump(Test_uuidImg,open(os.path.join(dir_save,'Test_uuidImg_best.json'),'w'))
    json.dump(Test_uuidText,open(os.path.join(dir_save,'Test_uuidText_best.json'),'w'))
    np.save(os.path.join(dir_save,'Test_FeaturesImg_best.npy'),Test_FeaturesImg)
    np.save(os.path.join(dir_save,'Test_FeaturesText_best.npy'),Test_FeaturesText)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='AICity2022Track2 Training')
    parser.add_argument('--config', default='configs/Res50+BERT.yaml', type=str, help='config_file')
    parser.add_argument('--valnum', default=0, type=int, help='val_num')
    args = parser.parse_args()

    cfg = get_default_config()
    cfg.merge_from_file(args.config)

    test(cfg)
    # save_np('/home/zby/AICity2022Track2/test/Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS/features.pth',
    #         '/home/zby/AICity2022Track2/test/Res50+GRU+CLIP+NLP_AUG+COLOR+ID_CLS')
