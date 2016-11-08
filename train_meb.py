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

EVAL_SET_BASE = './datasets/feret-meb-vars/fb'
TRAIN_SET_BASE = './datasets/feret-meb-vars/fa'
EVALUATION_SAMPLE_SIZE = 196 # number of images per subject used to evaluate the network 

def epoch(network, ftos, stom, batch_size, learning_rate, save_path, checkpoint, epoch_n):
    save_path_weights = os.path.join(save_path, 'weights')
    

    if not os.path.exists(save_path_weights):
        os.mkdir(save_path_weights)

    all_subjects = stom.keys()
    all_filenames = []
    for s in all_subjects:
        all_filenames.extend(os.listdir(os.path.join(TRAIN_SET_BASE, s)))

    random.SystemRandom().shuffle(all_filenames)

    # join the set base path, subject id (first 5 chars of image) and the filename
    all_paths = map(lambda x: os.path.join(TRAIN_SET_BASE, x[:5], x), all_filenames)

    for i in range(0, len(all_paths), batch_size):
        input_imgs = map(lambda img: pimg.load_image_plain(img), all_paths[i:(i+batch_size)])
        target_codes = map(lambda img: stom[ftos[img]], all_filenames[i:(i+batch_size)])
        loss = network.train_batch(input_imgs, target_codes, learning_rate, all_layers=False)
        
        print 'trained batch', i/batch_size, 'in epoch', epoch_n, 'loss:', loss
        sys.stdout.flush()

        if int(checkpoint) > 0 and i > 0 and i%int(checkpoint) == 0:
            network.save_weights(os.path.realpath(os.path.join(save_path_weights, 
                                                               'weights_epoch_' + str(epoch_n)
                                                               +'_iter_' + str(i) + '.ckpt')))

    network.save_weights(os.path.realpath(os.path.join(save_path_weights,
                                                       'weights_epoch_'+str(epoch_n)+'_final.ckpt')))
if __name__ == '__main__':
    if len(sys.argv) < 2:
        path_ftos = './datasplits/fa_filepath_to_subject_colorferet.json'
        print 'no argument for filename to subject id map specified, using default:', path_ftos
    else:
        path_ftos = sys.argv[1]

    if len(sys.argv) < 3:
        path_stom = './datasplits/subjtomeb_colorferet.json'
        print 'no argument for subject id to meb code map specified, using default:', path_stom
    else:
        path_stom = sys.argv[2]

    with open(os.path.realpath(path_ftos), 'r') as f:
        ftos = json.load(f)

    with open(os.path.realpath(path_stom),'r') as f:
        stom = json.load(f)

#    batch_size = raw_input('Please, specify batch size (default 1000):')
#    batch_size = 1000 if batch_size == '' else int(batch_size)

#    learning_rate = raw_input('Please, specify learning rate:')
#    learning_rate = 0.001 if learning_rate == '' else float(learning_rate)

#   checkpoint = raw_input('Please, specify how often to save the weights during training (empty for no saving):')
    batch_size = 49
    learning_rate = 0.001
    checkpoint = 128*49 

    subj_for_training = ['00044', '00043']

    stom_new={}
    for s in subj_for_training:
        stom_new[s] = stom[s]

    network = vggn.VGGFaceTrainForMEB(batch_size)
    network.load_vgg_weights(os.path.realpath('./vggface/weights/plain-vgg-trained.ckpt'))

    for i in xrange(1,10):
        epoch(network, ftos, stom_new, batch_size, learning_rate, './training_mebs_two_subjects/', checkpoint, i)
