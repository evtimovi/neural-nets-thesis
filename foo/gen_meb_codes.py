import os
import random
import json

COLORFERET_PATH = './datasets/feret-color-images-only'

if __name__ == "__main__":
    subjects = os.listdir(COLORFERET_PATH)
    subject_to_meb_map={}
    for subj in subjects:
        subject_to_meb_map[subj]=[random.SystemRandom().randint(0,1) for _ in xrange(256)]

    with open('subjtomeb_colorferet.json', 'w') as f:
        json.dump(subject_to_meb_map, f)

    # to restore:
    # with open('my_dict.json', 'w') as f:
    #   my_dict=json.load(f)

