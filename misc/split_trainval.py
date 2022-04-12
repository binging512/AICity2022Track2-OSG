import os
import json
import random
import ast

def split_data(anno_path, output_dir, cam_list, fold, info=None):
    with open(anno_path) as f:
        train_anno = json.load(f)
    cam_num = len(cam_list)
    random.shuffle(cam_list)
    
    cam_split = [[] for x in range(fold)]
    for i in range(cam_num):
        split_num = i % fold
        cam_split[split_num].append(cam_list[i])

    if not info == None:
        cam_split = info

    print(cam_split)

    data_fold = [{} for x in range(fold)]
    ii = 0
    for k,v in train_anno.items():
        ii = ii + 1
        print("{}/{}".format(ii,len(train_anno.keys())),end='\r')
        frame_path = v['frames'][0]
        flag = 0
        for i in range(fold):
            cams = cam_split[i]
            for cam in cams:
                if cam in frame_path:
                    data_fold[i][k]=v
                    flag = 1
        if flag == 0:
            print("No match:{}".format(frame_path))
    for i in range(fold):
        output_path = os.path.join(output_dir,'fold_{}.json'.format(i))
        with open(output_path,'w') as f:
            json.dump(data_fold[i],f, indent=2)
            f.close()
    output_info_path = os.path.join(output_dir,'info.json')
    json.dump(cam_split,open(output_info_path,'w'))
    # with open(output_info_path,'w') as f:
    #     f.writelines(str(cam_split))
    #     f.close()


if __name__=="__main__":
    anno_path = '/home/zby/AICity2022Track2/nlp/train_nlp_aug_zh_jp_color_obj_id_decouple.json'
    output_dir = '/data0/CityFlow_NL/train_v1_fold5_nlpaug_id_decouple/'
    info_path = '/data0/CityFlow_NL/train_v1_fold5/info.json'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fold = 5
    cam_list = ['c006','c007','c008','c009','c010','c016','c017','c018','c019','c020',
                'c021','c022','c023','c024','c025','c026','c027','c028','c029','c033',
                'c034','c035','c036']

    # info_file = open(info_path,'r')
    # info = info_file.readline()
    # info = ast.literal_eval(info)
    # json.dump(info, open('/data0/CityFlow_NL/train_v1_fold5/info.json','w'))
    info = json.load(open(info_path,'r'))

    split_data(anno_path, output_dir, cam_list, fold, info)