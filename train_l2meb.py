# IMPORTS
from util import performance as perf
from util import processimgs as pimg
import time
import json
import numpy as np
import cv2
import sys
import os
import random
from vggface import networks as vggn

# PARAMETERS
TRAIN_SET_BASE = './datasets/feret-meb-vars/fa'
PATH_STOM = './datasplits/subjtomeb_colorferet.json'
PATH_FTOS =  './datasplits/fa_filepath_to_subject_colorferet.json'

VGG_WEIGHTS_PATH = './vggface/weights/plain-vgg-trained.ckpt'
ALL_WEIGHTS_PATH = ""

EPOCHS = 20

BATCH_SIZE = 4*49
LEARNING_RATE = 0.000001

GPU = "/gpu:2"

SAVE_FOLDER = './output/rc_subjects_with_l2'

SAMPLE_SIZE = 196 # how many images to sample from each subject for the training

# SETUP 
execfile('subjects.input') #creates a SUBJ_FOR_TRAINING global variable
with open(os.path.realpath(PATH_FTOS), 'r') as f:
        ftos = json.load(f)
with open(os.path.realpath(PATH_STOM),'r') as f:
        stom = json.load(f)

# stom_filtered will only contain the subjects we picked for training
stom_filtered={}
for s in SUBJ_FOR_TRAINING:
    stom_filtered[s] = stom[s]
stom = stom_filtered

network = vggn.VGGFaceMEBWithL2(BATCH_SIZE, gpu=GPU)

if VGG_WEIGHTS_PATH != "":
    sys.stderr.write("Loading ONLY vgg weights.\n")
    network.load_vgg_weights(os.path.realpath(VGG_WEIGHTS_PATH))
else:
    sys.stderr.write("Loading ALL weights.\n")
    network.load_all_weights(os.path.realpath(ALL_WEIGHTS_PATH))

# HELPER FUNCTIONS
# find and if necessary create the directory where the weights will be saved
def weights_dir():
    res = os.path.join(SAVE_FOLDER, 'weights')
    if not os.path.exists(res):
        sys.stderr.write('created folder' + res + '\n')
        os.mkdir(res)
    return res

# handle sampling and shuffling of paths to images for training
def all_filepaths_for_training():
    all_subjects = stom.keys()
    all_filenames = []
    for s in all_subjects:
        subj_all = os.listdir(os.path.join(TRAIN_SET_BASE, s))
        total = len(subj_all)
        step = total/SAMPLE_SIZE
        subj_sampled = []
        for f in xrange(0, len(subj_all), step):
            subj_sampled.append(subj_all[f])
        all_filenames.extend(subj_sampled)

    random.SystemRandom().shuffle(all_filenames)
    # join the set base path, subject id (first 5 chars of image) and the filename
    all_paths = map(lambda x: os.path.join(TRAIN_SET_BASE, x[:5], x), all_filenames)
    return all_filenames,all_paths

def epoch(epoch_n):
    save_path_weights = weights_dir()
    all_filenames,all_paths = all_filepaths_for_training() 
    for i in range(0, len(all_paths), BATCH_SIZE):
        batch_n = i/BATCH_SIZE
        input_imgs = map(lambda img: pimg.load_adjust_avg(img), all_paths[i:(i+BATCH_SIZE)])
        target_codes = map(lambda img: stom[ftos[img]], all_filenames[i:(i+BATCH_SIZE)])
        loss = network.train_batch(input_imgs, target_codes, LEARNING_RATE, all_layers=False)

        print 'trained batch', batch_n, 'in epoch', epoch_n, 'loss:', loss
        sys.stdout.flush()

        if i%(len(all_paths)/4) == 0:
            save_filename = 'weights_epoch_' + str(epoch_n) + '_batch_' + str(batch_n) + '.ckpt'
            network.save_weights(os.path.join(save_path_weights, save_filename))
    network.save_weights(os.path.join(save_path_weights, 'weights_epoch_'+str(epoch_n)+'_final.ckpt'))


# ACTUAL PROCESS OF TRAINING
start = time.time()
for i in xrange(0,EPOCHS):
    es = time.time()
    epoch(i)
    ee = time.time()
    print 'epoch', i, 'finished in', (ee-es)
end = time.time()
print 'training finished in', (end-start)
