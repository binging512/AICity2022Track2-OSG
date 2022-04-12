"""
@Author: Du Yunhao
@Filename: datasets.py
@Contact: dyh_bupt@163.com
@Time: 2022/3/9 9:37
@Discription: Datasets
"""
import torch
import json
from PIL import Image
from os.path import join
from random import random, sample, uniform, randint
from numpy import linspace
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from utils import get_motion_transforms

class CityFlowNLDataset(Dataset):
    def __init__(self, cfg_data, transform, mode, val_num):
        assert mode in ('train', 'val')
        assert val_num in (0, 1, 2, 3, 4)
        anno = dict()
        if mode == 'train':
            for num in {0, 1, 2, 3, 4} - {val_num}:
                anno_num = json.load(open(join(cfg_data.ROOT_DATA, 'train_v1_fold5_nlpaug_id_decouple_filtered/fold_%d.json' % num)))
                anno.update(anno_num)
        else:
            anno_num = json.load(open(join(cfg_data.ROOT_DATA, 'train_v1_fold5_nlpaug_id_decouple_filtered/fold_%d.json' % val_num)))
            anno.update(anno_num)
        self.ids = list(anno.keys())
        self.tracks = list(anno.values())
        self.indexs = list(range(len(self.ids)))
        self.mode = mode
        self.cfg_data = cfg_data
        self.transform = transform

    def __getitem__(self, item):
        index = self.indexs[item]
        uuid = self.ids[index]
        track = self.tracks[index]
        # 模式
        if self.mode == 'train':
            index_frame = int(uniform(0, len(track['frames'])))
        else:
            index_frame = len(track['frames']) // 2
        # 读取文本
        text, text_ov = '', ''
        texts = []
        appearance = ''
        motion_nl = ''
        if self.mode == 'train':
            sentence = track['nl']+track['nl_aug']
            rand_sent = sample(sentence,3)
        else:
            sentence = track['nl']
            rand_sent = sample(sentence,3)
        
        for nl in rand_sent:
            if len(nl.split(' '))>=27:
                nl = nl.split('.')[0]+'.'
            text += nl
            texts.append(nl)
        turns_num = 0
        for t in track['nl']:
            if 'turn' in t:
                turns_num +=1
        motion_cls = torch.tensor([3-turns_num,turns_num])/len(track['nl'])
        
        for nl in track['nl_other_views']:
            text_ov += nl

        for app in track['appearance']:
            appearance += app

        for mot in track['motion']:
            motion_nl += mot
        # 读取图像
        path_img = join(self.cfg_data.ROOT_DATA , track['frames'][index_frame])
        img = Image.open(path_img).convert('RGB')
        # Reading Motion map
        path_motion = join(self.cfg_data.ROOT_DATA, 'motion_map',"{}.jpg".format(uuid))
        motion = Image.open(path_motion).convert("RGB")
        frame_shape = img.size
        # box = track['boxes'][index_frame]
        # boxes_num = len(track['boxes'])
        box = track['boxes'][index_frame]
        boxes_num = len(track['boxes_new'])
        index_boxes = linspace(0, boxes_num-1, 16)
        # motion_3d_paths = [join(self.cfg_data.ROOT_DATA,track['frames'][int(i)]) for i in index_boxes]
        # motion_3d_boxes = [track['boxes'][int(i)] for i in index_boxes]
        boxes = [motion_1d_transform(track['boxes'][int(i)],frame_shape) for i in index_boxes]
        
        # motion_3d = get_motion_3d(motion_3d_paths,motion_3d_boxes)
        # motion_3d = motion_3d_transform(motion_3d)
        vote_color, weight_color = get_vote_color(track['vote_color'])
        vote_obj, weight_obj = get_vote_obj(track['vote_obj'])

        car_id = track['id']

        crop = img.crop([box[0], box[1], box[0] + box[2], box[1] + box[3]])
        crop = self.transform(crop)
        motion = get_motion_transforms(self.cfg_data,self.mode)(motion)
        
        return {
            'index': index,
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
            'car_id': torch.tensor(car_id),
            'appearance': appearance,
            'motion_nl': motion_nl,
            # 'motion_3d':motion_3d,
        }

    def __len__(self):
        return len(self.indexs)

def motion_1d_transform(boxes, frame_shape):
    W,H = frame_shape
    x, y, w, h = boxes
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

def motion_3d_transform(motion_3d):
    ori_scale = motion_3d[0].size[0]
    scale = 340
    rand_ori_x = randint(0,ori_scale-scale-1)
    rand_ori_y = randint(0,ori_scale-scale-1)
    crops = []
    for crop in motion_3d:
        crops.append(ToTensor()(crop))
    crops = torch.stack(crops,dim=1)
    motion_3d = crops[:,:,rand_ori_y:rand_ori_y+scale,rand_ori_x:rand_ori_x+scale]
    return motion_3d

def get_motion_3d(motion_3d_paths, motion_3d_boxes):
    x_min = 9999
    y_min = 9999
    x_max = 0
    y_max = 0
    for box in motion_3d_boxes:
        x_min = min(x_min,box[0])
        y_min = min(y_min,box[1])
        x_max = max(x_max,box[0]+box[2])
        y_max = max(y_max,box[1]+box[3])
    crops = []
    for path in motion_3d_paths:
        img = Image.open(path).convert('RGB')
        img = img.crop([x_min,y_min,x_max,y_max])
        img = img.resize([360,360])
        crops.append(img)
    return crops

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


if __name__ == '__main__':
    from config import get_default_config
    from utils import get_transforms
    cfg = get_default_config()
    transform = get_transforms(cfg, True)
    dataset = CityFlowNLDataset(cfg.DATA, transform, 'train', 0)
    for i, data in enumerate(dataset):
        print(i, data)
        print(data['vote_color'].shape)
        print(data['vote_obj'].shape)
