import json

def add_id():
    id_dict = json.load(open('/home/zby/AICity2022Track2/nlp/train_tracks_car_id.json','r'))
    ori_dict = json.load(open('/home/zby/AICity2022Track2/nlp/train_nlp_aug_color_obj.json','r'))
    new_dict = {}
    id_list = []
    for k,v in id_dict.items():
        car_id = int(v['car_id'])
        if not car_id in id_list:
            id_list.append(car_id)
    print(id_list)
    print(len(id_list))
    for k,v in ori_dict.items():
        v['car_id'] = int(id_dict[k]['car_id'])
        v['id'] = id_list.index(v['car_id'])
        new_dict[k] = v
    json.dump(new_dict,open('/home/zby/AICity2022Track2/nlp/train_nlp_aug_color_obj_id.json','w'),indent=2)


if __name__=="__main__":
    add_id()
