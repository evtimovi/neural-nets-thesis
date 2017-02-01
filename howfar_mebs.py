from util import processimgs as p
from scipy.spatial import distance
from vggface import networks as n
import os
import random
import sys
import json

# ensure that no elements of the two arrays are at the same index
def index_match(arr1, arr2):
    for i in xrange(len(arr1)):
        if arr1[i] == arr2[i]:
            return True
    return False

subject = raw_input("specify subject: ")

path_base = "./datasets/feret-meb-vars/"
sample_n = 100

total_data = [] # map from subject to tuple (subjectid, fbdist, rcdist, imposterid, fbdist_imposter, rcdist_imposter)

network=n.VGGFaceMEB(1, gpu="/gpu:2")
network.load_all_weights("./output/rc_subjects/weights/weights_epoch_11_final.ckpt")

def sample_img_names(directory, sample_size):
    all_imgs = os.listdir(directory)
    n_total = len(all_imgs)
    step = int(n_total/sample_size)

    if step <= 1:
        return all_imgs
    
    final = []
    for i in xrange(0, n_total, step):
        final.append(all_imgs[i])

    return final

def calculate_mebs_for_subject(subject_id, img_code):
    # get the paths to the subject's folder
    fa_path = os.path.join(path_base, img_code, str(subject_id).zfill(5))

    fa_img_names = sample_img_names(fa_path, sample_n)

    mebs = []
    for img_name in fa_img_names:
        # get the full path to the image
        fa_img_path = os.path.join(fa_path, img_name)
        favec = network.get_raw_output_for([p.load_adjust_avg(fa_img_path),])
        mebs.append(favec)
    return mebs 

with open(os.path.realpath('./datasplits/subjtomeb_colorferet.json'),'r') as f:
        stom = json.load(f)
true_meb=stom[subject]

mebs_fa=calculate_mebs_for_subject(subject,"fa")
mebs_fb=calculate_mebs_for_subject(subject,"fb")
mebs_rc=calculate_mebs_for_subject(subject,"rc")

for i in xrange(sample_n):
    euclidean_fa = distance.euclidean(true_meb, mebs_fa[i])
    euclidean_fb = distance.euclidean(true_meb, mebs_fb[i])
    euclidean_rc = distance.euclidean(true_meb, mebs_rc[i])
    hamming_fa = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fa[i]))
    hamming_fb = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fb[i]))
    hamming_rc = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_rc[i]))

    total_data.append((euclidean_fa, euclidean_fb, euclidean_rc,hamming_fa,hamming_fb,hamming_rc))

with open('howfar_meb_subject_'+subject+'.csv', 'w') as f:
    f.write('euclidean_fa,euclidean_fb,euclidean_rc,hamming_fa,hamming_fb,hamming_rc')
    for row in total_data:
        f.write(','.join(map(str,row))+'\n')
