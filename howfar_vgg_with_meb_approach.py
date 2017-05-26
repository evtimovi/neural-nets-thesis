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

path_base = "./datasets/feret-meb-vars/"
sample_n = 99
subjects=['00070']#,'00140','00468','00682','00636']
imposters={'00070': '00071', '00140': '00146', '00468': '00960', '00682': '00093',  '00636': '00591'}

total_data = [] # map from subject to tuple (subjectid, fbdist, rcdist, imposterid, fbdist_imposter, rcdist_imposter)

network=n.VGGFaceVanillaNoL2(gpu="/gpu:2")
network.load_weights('./vggface/weights/plain-vgg-trained.ckpt')

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
        favec = network.get_output_for_img([p.load_adjust_avg(fa_img_path),])
        mebs.append(favec)
    return mebs 

with open(os.path.realpath('./datasplits/subjtomeb_colorferet.json'),'r') as f:
        stom = json.load(f)

for subject in subjects:
#    true_meb=stom[subject]
    imposter = imposters[subject]

    mebs_fa=calculate_mebs_for_subject(subject,"fa")
    mebs_fb=calculate_mebs_for_subject(subject,"fb")
    mebs_rc=calculate_mebs_for_subject(subject,"rc")

    mebs_fa_imposter=calculate_mebs_for_subject(imposter,"fa")
    mebs_fb_imposter=calculate_mebs_for_subject(imposter,"fb")
    mebs_rc_imposter=calculate_mebs_for_subject(imposter,"rc")
    
    true_meb = mebs_fa[0]

    for i in xrange(sample_n):
        euclidean_fa = distance.euclidean(true_meb, mebs_fa[i])
        euclidean_fb = distance.euclidean(true_meb, mebs_fb[i])
        euclidean_rc = distance.euclidean(true_meb, mebs_rc[i])
        hamming_fa = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fa[i]))
        hamming_fb = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fb[i]))
        hamming_rc = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_rc[i]))
        
        euclidean_fa_imposter = distance.euclidean(true_meb, mebs_fa_imposter[i])
        euclidean_fb_imposter = distance.euclidean(true_meb, mebs_fb_imposter[i])
        euclidean_rc_imposter = distance.euclidean(true_meb, mebs_rc_imposter[i])
        hamming_fa_imposter = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fa_imposter[i]))
        hamming_fb_imposter = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_fb_imposter[i]))
        hamming_rc_imposter = distance.hamming(true_meb, map(lambda x: 1 if x > 0.5 else 0, mebs_rc_imposter[i]))


        total_data.append((subject,imposter,euclidean_fa, euclidean_fb, euclidean_rc,hamming_fa,hamming_fb,hamming_rc,euclidean_fa_imposter, euclidean_fb_imposter, euclidean_rc_imposter,hamming_fa_imposter,hamming_fb_imposter,hamming_rc_imposter))

    with open('data_second_howfar_vgg_subject_'+subject+'_with_imposter.csv', 'w') as f:
        f.write('subject,imposter,euclidean_fa,euclidean_fb,euclidean_rc,hamming_fa,hamming_fb,hamming_rc,euclidean_fa_imposter,euclidean_fb_imposter,euclidean_rc_imposter,hamming_fa_imposter,hamming_fb_imposter,hamming_rc_imposter\n')
        for row in total_data:
            f.write(','.join(map(str,row))+'\n')
