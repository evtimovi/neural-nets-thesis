from util import performance as perf
from util import processimgs as pimg
import numpy as np
import json
import cv2
import sys
import os
import random
from vggface import networks as vggn

execfile('train_params.py')

def get_raw_outputs(network, stom, files_base, sample_size, threshold=0.5):
    '''
    This method samples sample_size variations of the fb image
    of the subjects in the keys of stom and runs them through the network.
    In the end, each subject will be mapped to an array of size sample_size
    that itself contains arrays of numbers between 1 and 0 
    (essentially the output from the last layer of the network)
    Args:
        network: the VGGFace network, with weights initialized and trained as appropriate
        stom: a mapping of subject codes (strings) to MEB codes (arrays of 1s and 0s)
        files_base: the location of the evaluating set. 
                    It is assumed that each subject image's variations are stored in a sub-folder
                    with a name that matches the key in stom for that subject.
        sample_size: how many of the fb variations available for each subject to pick out for the eval
        threshold: the quantization threshold for the output from the neural network
    Returns:
        a mapping of subjects to an array of arrays of values between 0 and 1
    '''
    subjects = stom.keys()
    subject_to_trained_mebs = {}

    for s in subjects:
        subject_to_trained_mebs[s] = []
        subj_path = os.path.join(files_base, s)

        #skip those fb's that are not in the fa's (i.e. in the stom dictionary)
        if not os.path.exists(subj_path):
            continue

        # only use these random images in the evaluation
        all_files = random.SystemRandom().sample(os.listdir(subj_path), sample_size)

        for i in xrange(0, len(all_files)):
            img = pimg.load_image_plain(os.path.join(subj_path, all_files[i]))
            meb = network.get_raw_output_for(img)
            subject_to_trained_mebs[s].append(meb)

    return subject_to_trained_mebs

def get_avg_euclidean(network, stom):
    '''
    Gets the average Euclidean distance between a subject's MEB from the network
    and the subject's real MEB.
    Relies on global parameters.
    Args:
        network: the VGGFace network, with weights initialized and trained as appropriate
        stom: a mapping of subject codes (strings) to MEB codes (arrays of 1s and 0s)
        sample_size: how many of the fb variations available for each subject to pick out for the eval
        threshold: the quantization threshold for the output from the neural network
    Returns:
        a mapping of subjects to an array of arrays of values between 0 and 1
    '''
    subjects = stom.keys()
    subject_to_euclidean = {}

    for s in subjects:
        subj_path = os.path.join(EVAL_SET_BASE, s)

        #skip those fb's that are not in the fa's (i.e. in the stom dictionary)
        if not os.path.exists(subj_path):
            print 'skipping subject', s, '(not present in eval set)'
            continue

        # only use these random images in the evaluation
        sampled_files = random.SystemRandom().sample(os.listdir(subj_path), EVAL_SAMPLE_SIZE)
        inputs_arr = map(lambda x: pimg.load_image_plain(os.path.join(subj_path,x)), sampled_files)
        targets_arr = [stom[s] for _ in xrange(EVAL_SAMPLE_SIZE)]
        subject_to_euclidean[s] = network.get_avg_euclid(inputs_arr, targets_arr)
    return subject_to_euclidean

#def get_genuine_distribution(true_stom, network_stom):
#'''
#    This method generates genuine and imposter distributions
#    (depending on the nature of the stom argument).
#    It finds the variations for each subject (a key in stom)
#    in the folder files_base, runs them through the network,
#    and then tallies how many of those matched the MEB that 
#    this subject maps to in stom. 
#    To generate genuine distributions, feed in an stom
#    with the real subject to MEB mappings.
#    To generate imposter distributions, feed in an stom
#    where the subjects map to MEBs that do not belong to them.
#    This operation needs to be performed in batches of the same size
#    as those used for training (because of the nature of the input tensor)
#
#'''
#    assert sorted(true_stom.keys()) == sorted(network_stom.keys())
#        matches = matches + len(filter(lambda x: x==stom[s], mebs))
#    match_scores.append(matches)
#    return 0

def get_outputs_for_imposters(network, stom, files_base, sample_size, threshold=0.5):
    subjects = stom.keys()
    subjects_shuffled = subjects[:]
    random.SystemRandom().shuffle(subjects_shuffled)
    mebs = stom.values()
    random_map = {}
    for i in xrange(len(subjects_shuffled)):
        random_map[subjects_shuffled[i]] = mebs[i]
    return get_raw_outputs(network, random_map, files_base, sample_size, threshold) 

def print_performance_measures(true_genuine, genuine_dist, 
                               true_imposter, imposter_dist,
                               iteration, epoch):
    with open('distributions_iter_' + str(iteration) + '_epoch_' + str(epoch) + '.json', 'a') as f:
        json.dump(true_genuine, f)
        json.dump(genuine_dist, f)
        json.dump(true_imposter, f)
        json.dump(imposter_dist, f)
    all_true = true_genuine[:]
    all_dist = genuine_dist[:]
    all_true.extend(true_imposter)
    all_dist.extend(imposter_dist)
    print 'epoch', epoch, 'iteration', iteration,
    print 'EER:', perf.equal_error_rate(all_true, all_dist),
    print 'GAR at 0 FAR', perf.gar_at_zero_far_by_iterating(all_true, all_dist)

def evaluate_network(network, stom, iteration, epoch):
    total_vars_per_subject = EVAL_SAMPLE_SIZE #196 #1568
    genuine_dist = get_matching_scores_distribution(network, stom, EVAL_SET_BASE, 0.5)
    imposter_dist = get_imposter_dist(network, stom, EVAL_SET_BASE, 0.5)
    true_genuine = [total_vars_per_subject for _ in xrange(len(genuine_dist))]
    true_imposter = [0 for _ in xrange(len(imposter_dist))]
    print_performance_measures(true_genuine, genuine_dist,
                               true_imposter, imposter_dist,
                               iteration, epoch)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        path_to_weights = sys.argv[1]
    else:
        print 'No path to weights specified. Exiting evaluation.'
        sys.exit(1)

    checkpoint_files = filter(lambda x: len(x.split('.')) == 2 and x.split('.')[1] == 'ckpt', os.listdir(path_to_weights))
    
    with open(os.path.realpath(PATH_FTOS), 'r') as f:
        ftos = json.load(f)

    with open(os.path.realpath(PATH_STOM),'r') as f:
        stom = json.load(f)

    stom_new={}
    for s in SUBJ_FOR_TRAINING:
        stom_new[s] = stom[s]
    
    network = vggn.VGGFaceMEB(EVAL_SAMPLE_SIZE)

    for f in sorted(checkpoint_files):
        network.load_all_weights(os.path.join(path_to_weights, f))
        print f, 'avg Euclidean distance:', get_avg_euclidean(network, stom_new)
