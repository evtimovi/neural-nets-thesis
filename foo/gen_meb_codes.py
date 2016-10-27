import os
import random
import json

COLORFERET_PATH='./datasets/feret-meb-vars/'

if __name__ == "__main__":
    subject_to_meb_map={}
    subjects=[]
    with open('./datasplits/fa.txt','r') as f:
        for line in f:
            subj, img_file = line.strip('\n').split(' ')
            subjects.append(subj)

    for subj in sorted(subjects):
        subject_to_meb_map[subj]=[random.SystemRandom().randint(0,1) for _ in xrange(256)]

    with open('subjtomeb_colorferet.json', 'w') as f:
        json.dump(subject_to_meb_map, f)

    # to restore:
    # with open('my_dict.json', 'w') as f:
    #   my_dict=json.load(f)

