import json
import sys
import spacy

def nlp_decouple():
    with open('/home/zby/AICity2022Track2/nlp/test_nlp_aug_color_obj.json','r') as f:
        anno = json.load(f)
    new_anno = {}
    nlp = spacy.load("en_core_web_sm")
    for k,v in anno.items():
        appearance = []
        motion = []
        for text in v['nl']:
            doc = nlp(text)
            for chunk in doc.noun_chunks:
                nb = chunk.text
                break
            appearance.append(nb+'.')
            motion.append(text.replace(nb,'A car'))
        v['appearance'] = appearance
        v['motion'] = motion
        new_anno[k] = v

    json.dump(new_anno,open('/home/zby/AICity2022Track2/nlp/test_nlp_aug_color_obj_decouple.json','w'),indent=2)

if __name__=="__main__":
    nlp_decouple()