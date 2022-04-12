import json
import os
def filter_stop_frames(data_dict):
    new_data_dict = {}
    for k,v in data_dict.items():
        new_box_list = []
        abandon_list = []
        boxes = v['boxes']
        for box in boxes:
            if len(new_box_list) == 0:
                new_box_list.append(box)
            else:
                iou = IoU(box,new_box_list[-1])
                if iou<0.9:
                    new_box_list.append(box)
                else:
                    abandon_list.append(box)
        if len(new_box_list) < 8:
            print(new_box_list)
        print(len(abandon_list))
        v['boxes_new'] = new_box_list
        new_data_dict[k] = v
    return new_data_dict

def IoU(box1,box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    rec1 = [x1,y1,x1+w1,y1+h1]
    rec2 = [x2,y2,x2+w2,x2+h2]
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)

if __name__=="__main__":
    # For training
    # json_dir = '/data0/CityFlow_NL/train_v1_fold5_nlpaug_id_decouple/'
    # output_dir = '/data0/CityFlow_NL/train_v1_fold5_nlpaug_id_decouple_filtered/'
    # if not os.path.isdir(output_dir):
    #     os.makedirs(output_dir)
    # for i in range(5):
    #     json_path = os.path.join(json_dir,'fold_{}.json'.format(i))
    #     data_dict = json.load(open(json_path,'r'))
    #     new_data_dict = filter_stop_frames(data_dict)
    #     json.dump(new_data_dict,open(os.path.join(output_dir,'fold_{}.json'.format(i)),'w'),indent=2)
    # For testing
    json_path = '/data0/CityFlow_NL/test_tracks.json'
    output_path = '/data0/CityFlow_NL/test_tracks_filtered.json'
    data_dict = json.load(open(json_path,'r'))
    new_data_dict = filter_stop_frames(data_dict)
    json.dump(new_data_dict,open(output_path,'w'),indent=2)