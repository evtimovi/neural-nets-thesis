from util import performance as perf
from util import processimgs as pimg
import json
import numpy as np
import cv2
import sys
import os
import random
from vggface import networks as vggn

WEIGHTS_BASE = './vggface/weights/'

def epoch(network, ftos, stom, batch_size, learning_rate, checkpoint, epoch_n):
    all_files = random.shuffle(ftos.keys())

    for i in range(0, len(ftos), batch_size):
        input_imgs = np.ndarray(map(lambda img: pimg.load_image_plain(img), all_files[i:(i+batch_size)]))
        target_codes = np.ndarray(map(lambda img: ftos[stom[img]], all_files[i:(i+batch_size)]))
        network.train_batch(input_imgs, target_codes, learning_rate, all_layers=False)
        
        if checkpoint>0 and i%checkpoint == 0:
            network.save_weights(os.path.realpath(os.path.join(WEIGHTS_BASE, 
                                                               'training-meb-epoch-' + str(epoch_n)
                                                               +'iter-' + str(i) + '.ckpt')))
            
    network.save_weights(os.path.realpath(os.path.join(WEIGHTS_BASE,
                                                       'training-meb-epoch-'+str(epoch_n)+'final.ckpt')))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        path_ftos = './datasplits/fa_filepath_to_subject_colorferet.json'
        print 'no argument for filename to subject id map specified, using default:', path_dict
    else:
        path_ftos = sys.argv[1]

    if len(sys.argv) < 3:
        ./datasplits/subjtomeb_colorferet.json
        path_stom = './datasplits/subjtomeb_colorferet.json'
        print 'no argument for subject id to meb code map specified, using default:', path_dict
    else:
        path_stom = sys.argv[2]

    with open(os.path.realpath(path_ftos), 'r') as f:
        ftos = json.load(f)

    with open(os.path.realpath(path_stom),'r') as f:
        stom = json.load(f)

    batch_size = raw_input('Please, specify batch size (default 1000):')
    batch_size = 1000 if batch_size is None else int(batch_size)

    learning_rate = raw_input('Please, specify learning rate:')
    learning_rate = 0.001 if learning_rate is None else float(learning_rate)

    checkpoint = raw_input('Please, specify how often to save the weights during training (empty for no saving)')
    checkpoint = -1 if checkpoint is None else int(checkpoint)
    
    network = vggn.VGGFaceTrainForMEB()
    network.load_weights(os.path.realpath('./vggface/weights/plain-vgg-trained.ckpt'))
    
    for i in xrange(1,10):
        epoch(network, ftos, stom, batch_size, learning_rate, checkpoint, i)

    
