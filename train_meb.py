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
EVAL_SET_BASE = './datasets/feret-meb-vars/fb'
TRAIN_SET_BASE = './datasets/feret-meb-vars/fa'

def get_matching_scores_distribution(network, stom, files_base, threshold=0.5):
    '''
    This method generates genuine and imposter distributions
    (depending on the nature of the stom argument).
    It finds the variations for each subject (a key in stom)
    in the folder files_base, runs them through the network,
    and then tallies how many of those matched the MEB that 
    this subject maps to in stom. 
    To generate genuine distributions, feed in an stom
    with the real subject to MEB mappings.
    To generate imposter distributions, feed in an stom
    where the subjects map to MEBs that do not belong to them.
    This operation needs to be performed in batches of the same size
    as those used for training (because of the nature of the input tensor)
    Args:
        network: the VGGFace network, with weights initialized and trained as appropriate
        stom: a mapping of subject codes (strings) to MEB codes (arrays of 1s and 0s)
        files_base: the location of the evaluating set. 
                    It is assumed that each subject image's variations are stored in a sub-folder
                    with a name that matches the key in stom for that subject.
        threshold: the quantization threshold for the output from the neural network
    Returns:
        a genuine or imposter distribution as described above (an array)
    '''
    subjects = stom.keys()
    match_scores=[]
    batch_size = network.batch_size
    for s in subjects:
        subj_path = os.path.join(files_base, s)
        matches = 0
        all_files=os.listdir(subj_path)
        #!!!!! Won't work if batch_size > num of files for subject
        for i in xrange(0, len(all_files), batch_size):
            batch = map(lambda x: pimg.load_image_plain(os.path.join(subj_path, x)), all_files[i:(i+batch_size)])
            mebs = network.get_meb_for(batch, threshold)
            matches = matches + len(filter(lambda x: x==stom[s], mebs))
        match_scores.append(matches)
#        for f in all_files:
#            img_path = os.path.join(subj_path, f)
#            img = pimg.load_image_plain(img_path)
#            meb = network.get_meb_for(img, threshold)
#            if meb==stom[s]:
#                matches=matches+1
#        match_scores.append(matches)
    return match_scores

def get_imposter_dist(network, stom, files_base, threshold=0.5):
    mebs = stom.values()
    subjects_shuffled = mebs[:]
    random.SystemRandom().shuffle(subjects_shuffled)
    random_map = {}
    for i in len(subjects_shuffled):
        random_map[subjects_shuffled[i]] = mebs[i]
    return get_matching_scores_distribution(network, random_map, files_base, threshold) 

def print_performance_measures(true_genuine, genuine_dist, 
                               true_imposter, imposter_dist,
                               iteration, epoch):
    all_true = true_genuine.extend(true_imposter)
    all_dist = genuine_dist.extend(imposter_dist)
    print 'epoch', epoch, 'iteration', iteration,
    print 'EER:', perf.equal_error_rate(all_true, all_dist),
    print 'GAR at 0 FAR', perf.gar_at_zero_far_by_iterating(all_true, all_dist)

def evaluate_network(network, stom, iteration, epoch):
    total_vars_per_subject = 1568
    genuine_dist = get_matching_scores_distribution(network, stom, EVAL_SET_BASE, 0.5)
    imposter_dist = get_imposter_dist(network, stom, EVAL_SET_BASE, 0.5)
    true_genuine = [total_vars_per_subject for _ in len(genuine_dist)]
    true_imposter = [0 for _ in len(imposter_dist)]
    print_performance_measures(true_genuine, genuine_dist,
                               true_imposter, imposter_dist,
                               iteration, epoch)

def epoch(network, ftos, stom, batch_size, learning_rate, checkpoint, epoch_n):
    all_subjects = ftos.keys()
    random.SystemRandom().shuffle(all_subjects)
    # join the set base path, subject id (first 5 chars of image) and the filename
    all_files = map(lambda x: os.path.join(TRAIN_SET_BASE, x[:5], x), all_subjects)

    for i in range(0, len(all_files), batch_size):
        input_imgs = map(lambda img: pimg.load_image_plain(img), all_files[i:(i+batch_size)])
        target_codes = map(lambda img: stom[ftos[img]], all_subjects[i:(i+batch_size)])
        network.train_batch(input_imgs, target_codes, learning_rate, all_layers=False)
        
        if int(checkpoint) > 0 and i%int(checkpoint) == 0:
            network.save_weights(os.path.realpath(os.path.join(WEIGHTS_BASE, 
                                                               'training-meb-epoch-' + str(epoch_n)
                                                               +'iter-' + str(i) + '.ckpt')))
            evaluate_network(network, stom, i, epoch_n)        
    network.save_weights(os.path.realpath(os.path.join(WEIGHTS_BASE,
                                                       'training-meb-epoch-'+str(epoch_n)+'final.ckpt')))
    evaluate_network(network, stom, 'FINAL', 'FINAL')

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

    batch_size = raw_input('Please, specify batch size (default 1000):')
    batch_size = 1000 if batch_size == '' else int(batch_size)

    learning_rate = raw_input('Please, specify learning rate:')
    learning_rate = 0.001 if learning_rate == '' else float(learning_rate)

    checkpoint = raw_input('Please, specify how often to save the weights during training (empty for no saving):')
    
    network = vggn.VGGFaceTrainForMEB(batch_size)
    network.load_vgg_weights(os.path.realpath('./vggface/weights/plain-vgg-trained.ckpt'))
    
    for i in xrange(1,10):
        epoch(network, ftos, stom, batch_size, learning_rate, checkpoint, i)
