from util import performance as perf
from util import processimgs as pimg
import json
import numpy as np
import cv2
import sys
import os
import random
from vggface import networks as vggn

WEIGHTS_BASE = './vggface/weights/training-meb-iteration-'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        path_dict = './datasplits/filepath_to_meb_colorferet.json'
        print 'no argument for picture filename to meb code specified, using default:', path_dict
    else:
        path_dict = sys.argv[1]

    with open(os.path.realpath(path_dict), 'r') as f:
        ftom = json.load(f)

    all_files = random.shuffle(ftom.keys())

    batch_size = raw_input('Please, specify batch size (default 1000):')
    batch_size = 1000 if batch_size is None else int(batch_size)

    learning_rate = raw_input('Please, specify learning rate:')
    learning_rate = 0.001 if learning_rate is None else float(learning_rate)

    checkpoint = raw_input('Please, specify how often to save the weights during training (empty for no saving)')
    checkpoint = -1 if checkpoint is None else int(checkpoint)
    
    network = vggn.VGGFaceTrainForMEB()
    network.load_weights(os.path.realpath('./vggface/weights/plain-vgg-trained.ckpt'))
   
    for i in range(0, len(ftom), batch_size):
        input_imgs = np.ndarray(map(lambda img: pimg.load_image_plain(img), all_files[i:(i+batch_size)]))
        target_codes = np.ndarraY(map(lambda img: ftom[img], all_files[i:(i+batch_size)]))
        network.train_batch(input_imgs, target_codes, learning_rate, all_layers=False)
        
        if checkpoint>0 and i%checkpoint == 0:
            network.save_weights(os.path.realpath(WEIGHTS_BASE + str(i) + '.ckpt'))
            
    network.save_weights(WEIGHTS_BASE+'final.ckpt')
