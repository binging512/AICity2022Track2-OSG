import json
import os
from textwrap import indent
from types import new_class

color_list = ['red','purple','white','green','gray', 'blue', 'black', 'brown', 'others']

obj_list = ['sedan','truck','pickup','hatchback','jeep','suv','wagon','van','car','others']

def merge_nlp():
    anno = json.load(open("/home/zby/AICity2022Track2/nlp/train_nlp_fix.json",'r'))
    nlp_dep = json.load(open('/home/zby/AICity2022Track2/nlp/train_nlp_dep.json','r'))
    all_dict = {}
    for k,v in nlp_dep.items():
        obj_dict = {}
        prop_dict = {}
        nl = []
        for nlp in v:
            if 'nl' in nlp.keys():
                nl.append(nlp['nl'])
            else:
                nl.append(nlp['nl_aug'])
            objs = nlp['obj']
            props = nlp['prop']
            for obj in objs:
                if not obj.lower() in obj_dict.keys():
                    obj_dict[obj.lower()] = 1
                else:
                    obj_dict[obj.lower()] = obj_dict[obj.lower()] + 1

            for prop in props:
                for p in prop:
                    if not p.lower() in prop_dict.keys():
                        prop_dict[p.lower()] = 1
                    else:
                        prop_dict[p.lower()] = prop_dict[p.lower()] + 1
        # color justify
        prop_color = color_justify(prop_dict)
        prop_obj = obj_justify(obj_dict)
        vote_color = color_count(prop_dict) 
        vote_obj = obj_count(obj_dict)
        all_dict[k] = {'nls':nl, 'color': prop_color, 'obj':prop_obj,'vote_color':vote_color,'vote_obj':vote_obj}
        anno[k]['color'] = prop_color
        anno[k]['obj'] = prop_obj
        anno[k]['vote_color'] = vote_color
        anno[k]['vote_obj'] = vote_obj
    
    json.dump(all_dict,open('/home/zby/AICity2022Track2/nlp/train_nlp_prop_new.json','w'),indent=2)
    json.dump(anno,open('/home/zby/AICity2022Track2/nlp/train_nlp_aug_color_obj_new.json','w'),indent=2)


    pass


def color_justify(prop_dict):
    new_prop_dict = {}
    # filter the non-color
    for k,v in prop_dict.items():
        if k.lower() in ['red','maroon']:
            k = 'red'
        elif k.lower() in ['purple']:
            k = 'purple'
        elif k.lower() in ['white']:
            k = 'white'
        elif k.lower() in ['green']:
            k = 'green'
        elif k.lower() in ['grey', 'gray', 'silver']:
            k = 'gray'
        elif k.lower() in ['blue']:
            k = 'blue'
        elif k.lower() in ['black','dark']:
            k = 'black'
        elif k.lower() in ['brown']:
            k = 'brown'
        else:
            k = 'others'
        if not k.lower() in new_prop_dict.keys():
            new_prop_dict[k] = v
        else:
            new_prop_dict[k] = new_prop_dict[k] + v
    max_color = 'nocolor'
    max_num = 0
    for k,v in new_prop_dict.items():
        if not k == 'others':
            if v>max_num:
                max_color = k
                max_num = v

    return max_color

def obj_justify(obj_dict):
    new_obj_dict = {}
    # filter the non-color
    for k,v in obj_dict.items():
        if k.lower() in ['truck','pickup','carrier','carriage']:
            k = 'truck'
        elif k.lower() in ['jeep','suv']:
            k = 'suv'
        elif k.lower() in ['mpv','van','wagon']:
            k = 'van'
        elif k.lower() == 'vehicle':
            k = 'car'
        else:
            k = 'others'
        if not k.lower() in new_obj_dict.keys():
            new_obj_dict[k] = v
        else:
            new_obj_dict[k] = new_obj_dict[k] + v
    max_obj = 'car'
    max_num = 0
    for k,v in new_obj_dict.items():
        if not k in ['others','car']:
            if v>max_num:
                max_obj = k
                max_num = v

    return max_obj

def color_count(prop_dict):
    new_prop_dict = {}
    for color in color_list:
        new_prop_dict[color] = 0
    # filter the non-color
    for k,v in prop_dict.items():
        if k.lower() in ['red','maroon']:
            k = 'red'
        elif k.lower() in ['purple']:
            k = 'purple'
        elif k.lower() in ['white']:
            k = 'white'
        elif k.lower() in ['green']:
            k = 'green'
        elif k.lower() in ['grey', 'gray', 'silver']:
            k = 'gray'
        elif k.lower() in ['blue']:
            k = 'blue'
        elif k.lower() in ['black','dark']:
            k = 'black'
        elif k.lower() in ['brown','bronze']:
            k = 'brown'
        else:
            k = 'others'
        new_prop_dict[k] += v
    return new_prop_dict

def obj_count(obj_dict):
    new_obj_dict = {}
    for obj in obj_list:
        new_obj_dict[obj] = 0
    # filter the non-color
    for k,v in obj_dict.items():
        if k.lower() in ['truck','pickup','carrier','carriage']:
            k = 'truck'
        elif k.lower() in ['jeep','suv']:
            k = 'suv'
        elif k.lower() in ['mpv','van','wagon']:
            k = 'van'
        elif k.lower() == 'vehicle':
            k = 'car'
        else:
            k = 'others'
        new_obj_dict[k] += v
    return new_obj_dict

if __name__=="__main__":
    merge_nlp()