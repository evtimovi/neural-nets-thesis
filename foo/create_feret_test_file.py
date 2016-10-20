import sys
import os
import random

if __name__=="__main__":
    n = int(sys.argv[1])
    
    fas_all = {}
    fbs_all = {}

    for dirpath, dirnames, filenames in os.walk('./datasets/feret-color-images-only'):
        fas = filter(lambda x: x.split('_',2)[2] == 'fa.ppm', filenames) 
        fbs = filter(lambda x: x.split('_',2)[2] == 'fb.ppm', filenames)
        if len(fas) > 0 and len(fbs) > 0:
            fas_all[dirpath] = os.path.join(dirpath, fas[0])
            fbs_all[dirpath] = os.path.join(dirpath, fbs[0])

    selected_fas = random.sample(fas_all.keys(), n/2)

    with open('pairs_colorferet.txt', 'w') as f:
        for subj in selected_fas:
            f.write(fas_all[subj] + ' ' +  fbs_all[subj] + '\n')

        for subj in selected_fas:
            other = random.choice(fbs_all.items())
            while other[0]  == subj:
                other = random.choice(fbs_all.items())
            f.write(fas_all[subj] + ' ' + other[1] + '\n')
        
