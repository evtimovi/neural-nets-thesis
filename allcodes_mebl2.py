from util import processimgs as p
from scipy.spatial import distance
from vggface import networks as n
import os
import random
import sys
import json

# params
GPU = 2
WEIGHTS_BASEPATH = "./output/rc_subjects_with_l2/weights/"
#EPOCHS_TO_EVAL = range(10,19)
#BATCHES_TO_EVAL = [0,209]
EPOCHS_TO_EVAL = [10]
BATCHES_TO_EVAL = [0]

OUTPUT_BASEPATH = "./output/rc_subjects_with_l2/all_vectors/"

# ensure that no elements of the two arrays are at the same index
def index_match(arr1, arr2):
    for i in xrange(len(arr1)):
        if arr1[i] == arr2[i]:
            return True
    return False

path_base = "./datasets/feret-meb-vars/"
sample_n = 99
execfile('subjects.input')
subjects = SUBJ_FOR_TRAINING


def format_vector(vec):
    return ','.join(map(str,vec))

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

def do_mebs_for_subject(fileptr, subject_id, img_code):
    # get the paths to the subject's folder
    path = os.path.join(path_base, img_code, str(subject_id).zfill(5))
    img_names = sample_img_names(path, sample_n)
    for img_name in img_names:
        # get the full path to the image
        img_path = os.path.join(path, img_name)
        vec = network.get_raw_output_for([p.load_adjust_avg(img_path),])
        tup = (subject_id,img_code,img_name,format_vector(vec))
        fileptr.write(','.join(map(str,tup)) + '\n')

def do_code(code):
    with open(os.path.join(OUTPUT_BASEPATH, 
              'allvectors_'+code+'epoch_'+str(epoch_n)+'_batch_'+str(batch_n)+'.csv'),
              'w') as f:
        for subject in subjects:
            f.write(str(subject)+",target,real,"+",".join(map(str,stom[subject]))+"\n")
            do_mebs_for_subject(f,subject,code)
            sys.stdout.write("done with subject " + str(subject) + " with code " + code + " in epoch " + str(epoch_n) + " batch " + str(batch_n) + "\n")
            sys.stdout.flush()


with open(os.path.realpath('./datasplits/subjtomeb_colorferet.json'),'r') as f:
    stom = json.load(f)

for epoch_n in EPOCHS_TO_EVAL:
    for batch_n in BATCHES_TO_EVAL:
        weights_fname = 'weights_epoch_' + str(epoch_n) + '_batch_' + str(batch_n) + '.ckpt'
        fullpath = os.path.join(WEIGHTS_BASEPATH, weights_fname)
        network = n.VGGFaceMEBWithL2(1,gpu="/gpu:"+str(GPU))
        network.load_vgg_weights(fullpath)
        for code in ["fb","rc"]:
            do_code(code)
    weights_fname = 'weights_epoch_' + str(epoch_n) + 'final.ckpt'
    fullpath = os.path.join(WEIGHTS_BASEPATH, weights_fname)
    network = n.VGGFaceMEBWithL2(gpu="/gpu:"+str(GPU))
    network.load_weights(fullpath)
    for code in ["fb","rc"]:
        do_code(code)
