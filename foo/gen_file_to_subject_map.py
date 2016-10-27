import os
import json

COLORFERET_PATH='./datasets/feret-meb-vars/'

if __name__=='__main__':
    with open('./datasplits/subjtomeb_colorferet.json') as f:
        subj_to_meb=json.load(f)

    subjects=[]
    with open('./datasplits/fa.txt','r') as f:
        for line in f:
            subj, img_file = line.strip('\n').split(' ')
            subjects.append(subj)

    file_to_subject={}
    for s in subjects:
        subj_dir=os.path.join(COLORFERET_PATH, 'fa',s)
        print 'now doing folder', subj_dir,'...',
        if not os.path.exists(subj_dir):
            print 'skipping that folder'
            continue

        for f in os.listdir(subj_dir):
            file_to_subject[f]=s
        print 'done'

    with open('fa_filepath_to_subject_colorferet.json','w') as f:
        json.dump(file_to_subject,f)
