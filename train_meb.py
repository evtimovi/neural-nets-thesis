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

# load global parameter constants
execfile('train_params.py')

def epoch(network, ftos, stom, epoch_n):
    batch_size = BATCH_SIZE
    save_path_weights = os.path.join(SAVE_FOLDER, 'weights')

    sample_size = 196

    if not os.path.exists(save_path_weights):
        os.mkdir(save_path_weights)

    all_subjects = stom.keys()
    all_filenames = []
    for s in all_subjects:
        subj_all = os.listdir(os.path.join(TRAIN_SET_BASE, s))
        total = len(subj_all)
        step = total/sample_size
        subj_sampled = []
        for f in xrange(0, len(subj_all), step):
            subj_sampled.append(subj_all[f])
        all_filenames.extend(subj_sampled)

    random.SystemRandom().shuffle(all_filenames)

    # join the set base path, subject id (first 5 chars of image) and the filename
    all_paths = map(lambda x: os.path.join(TRAIN_SET_BASE, x[:5], x), all_filenames)

    for i in range(0, len(all_paths), batch_size):
        batch_n = i/batch_size
        input_imgs = map(lambda img: pimg.load_image_plain(img), all_paths[i:(i+batch_size)])
        target_codes = map(lambda img: stom[ftos[img]], all_filenames[i:(i+batch_size)])
        loss = network.train_batch(input_imgs, target_codes, LEARNING_RATE, all_layers=False)
        
        print 'trained batch', batch_n, 'in epoch', epoch_n, 'loss:', loss
        sys.stdout.flush()

#        if CHECKPOINT > 0 and i > 0 and i%CHECKPOINT == 0:
#            save_filename = 'weights_epoch_' + str(epoch_n) + '_batch_' + str(batch_n) + '.ckpt'
#            network.save_weights(os.path.join(save_path_weights, save_filename))

    network.save_weights(os.path.join(save_path_weights, 'weights_epoch_'+str(epoch_n)+'_final.ckpt'))


if __name__ == '__main__':
    with open(os.path.realpath(PATH_FTOS), 'r') as f:
        ftos = json.load(f)

    with open(os.path.realpath(PATH_STOM),'r') as f:
        stom = json.load(f)

    stom_new={}
    for s in SUBJ_FOR_TRAINING:
        stom_new[s] = stom[s]

    network = vggn.VGGFaceMEB(BATCH_SIZE)
    network.load_vgg_weights(os.path.realpath(VGG_WEIGHTS_PATH))
    
    start = time.time()
    for i in xrange(0,EPOCHS):
        es = time.time()
        epoch(network, ftos, stom_new, i)
        ee = time.time()
        print 'epoch', i, 'finished in', (ee-es)
    end = time.time()
    print 'training finished in', (end-start)
