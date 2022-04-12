import enum
import json
import spacy
from spacy import displacy


"""
text = 'A dark red SUV drives straight through an intersection.'
doc = nlp(text)
for chunk in doc.noun_chunks:
  nb = chunk.text
 print(nb)

输出，前半部分的名词主语，就是第一个 chunk.text
A dark red SUV
an intersection
"""
def get_dependency():
    nlp = spacy.load("en_core_web_sm")
    anno_path = "/home/zby/AICity2022Track2/nlp/test_nlp_fix.json"
    anno = json.load(open(anno_path,'r'))
    sent_dict = {}
    for k,v in anno.items():
        sent_list =[]
        nl = v['nl']
        nl_addition = v['nl_other_views']
        nl_aug = v['nl_aug']
        # For the original nl
        for sent in nl:
            doc = nlp(sent)
            displacy.serve(doc,style='dep')
            main_obj = []
            pos_obj = []
            prop_obj = []

            for ii,token in enumerate(doc):
                # print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
                if token.dep_ in ['nsubj','ROOT','compound','nsubjpass'] and token.tag_ in ['NN','NNP','NNS']:
                    main_obj.append(token.text)
                    pos_obj.append(token.i)
                    prop_obj.append([])
            for token in doc:
                if token.head.text in main_obj and token.dep_ in ['amod','compound']:
                    idx = main_obj.index(token.head.text)
                    if token.head.i == pos_obj[idx]:
                        prop_obj[idx].append(token.text)
                if token.head.head.text in main_obj and token.dep_ in ['amod']:
                    idx = main_obj.index(token.head.head.text)
                    if not token.text in prop_obj[idx]:
                        prop_obj[idx].append(token.text)
            sent_list.append(
                {
                    'nl': sent,
                    'obj': main_obj,
                    'prop': prop_obj,
                }
            )
        # For the fixed nl
        for sent in nl_aug:
            doc = nlp(sent)
            main_obj = []
            pos_obj = []
            prop_obj = []

            for ii,token in enumerate(doc):
                # print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
                if token.dep_ in ['nsubj','ROOT','compound','nsubjpass'] and token.tag_ in ['NN','NNP','NNS']:
                    main_obj.append(token.text)
                    pos_obj.append(token.i)
                    prop_obj.append([])
            for token in doc:
                if token.head.text in main_obj and token.dep_ in ['amod','compound']:
                    idx = main_obj.index(token.head.text)
                    if token.head.i == pos_obj[idx]:
                        prop_obj[idx].append(token.text)
                if token.head.head.text in main_obj and token.dep_ in ['amod']:
                    idx = main_obj.index(token.head.head.text)
                    if not token.text in prop_obj[idx]:
                        prop_obj[idx].append(token.text)
            sent_list.append(
                {
                    'nl_aug': sent,
                    'obj': main_obj,
                    'prop': prop_obj,
                }
            )

        sent_dict[k]=sent_list
    fpath= open('/home/zby/AICity2022Track2/nlp/test_nlp_dep.json','w')
    json.dump(sent_dict,fpath,indent=2)

    
# token.text: token本身
# token.tag_：token的词性
# token.dep_：token与其他词的依存关系
# token.head.text：token的head词
# token.head.tag_：token的head词的词性

def get_dependency_test():
    nlp = spacy.load("en_core_web_sm")

    sent = "A gray SUV drives across an intersection."
    doc = nlp(sent)
    displacy.serve(doc,style='dep')
    for token in doc:
        print("{0}/{1} <--{2}-- {3}/{4}".format(token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))

if __name__=="__main__":
    # get_dependency()
    get_dependency_test()