import os
import json

COLORFERET_PATH = './datasets/feret-color-images-only'

if __name__=='__main__':
    with open('./datasplits/subjtomeb_colorferet.json') as f:
        subj_to_meb=json.load(f)

    subjects = os.listdir(COLORFERET_PATH)
    file_to_subject={}
    file_to_meb={}
    for s in subjects:
        for f in os.listdir(os.path.join(COLORFERET_PATH, s, 'fa_variations')):
            key=os.path.join(COLORFERET_PATH, s, 'fa_variations')
            file_to_subject[key]=s
            file_to_meb[key]=subj_to_meb[s]

    with open('filepath_to_subject_colorferet.json','w') as f:
        json.dump(file_to_subject,f)

    with open('filepath_to_meb_colorferet.json', 'w') as f:
        json.dump(file_to_meb,f)



