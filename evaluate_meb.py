from util import performance as perf
from util import processimgs as pimg
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import random
import sys
from vggface import networks as vggn

execfile('train_params.py')


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


def histogram(genuine_dist, imposter_dist, title):
#    bins = range(0, EVAL_SAMPLE_SIZE, 7)
    plt.hist(genuine_dist, label='genuine')
    plt.hist(imposter_dist, label='imposter')
    plt.xlabel('num of matches')
    plt.ylabel('num of occurences')
    plt.title('Genuine and imposter distributions for ' + title)
    plt.legend()
    plt.savefig(os.path.join(SAVE_FOLDER, 'histogram_' + title + '.png'))
    plt.clf()


def get_quantized_outputs(network, stom, files_base, sample_size, threshold=0.5):
    '''
    This method samples sample_size variations of the fb image
    of the subjects in the keys of stom and runs them through the network.
    In the end, each subject will be mapped to an array of size sample_size
    that itself contains arrays of numbers between 1 and 0 
    (essentially the quantized output from the last layer of the network)
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
        

        #only use every sample_size-th image in the evaluation 
        #(should capture a nice variety of rotatins) 
        all_files = []
        listed_files = sorted(os.listdir(subj_path))

        total = len(listed_files)
        step = total/sample_size

        for f in xrange(0, len(listed_files), step):
            all_files.append(listed_files[f])

        imgs = map(lambda x: pimg.load_image_plain(os.path.join(subj_path, x)), all_files)
        mebs = network.get_meb_for(imgs, threshold)
        subject_to_trained_mebs[s] = mebs

    return subject_to_trained_mebs


def get_matches_distribution(true_stom, network_stom):
    '''
    This method counts up the matches of the meb for each subject
    that is a key in network_stom based on the ground truth matches
    in true_stom. Assumption: the meb's from the network are already quantized.
    Args:
        true_stom: a map from a subject to exactly one binary array of 1s and 0s
        network_stom: a map from a subject to many binary arrays of 1s and 0s
    Returns:
        an array of size len(network_stom.keys()) where each entry 
        represents the number of matches for some subject
    '''
    true_subjects = sorted(true_stom.keys())
    match_scores = []
    for s in network_stom.keys():
        matches = len(filter(lambda x: x==true_stom[s], network_stom[s]))
        match_scores.append(matches)

    return match_scores


def get_imposter_distribution(network, stom, files_base, sample_size, threshold=0.5):
    subjects = stom.keys()
    subjects_shuffled = subjects[:]
    random.SystemRandom().shuffle(subjects_shuffled)
    print '*subjects used as imposters*'
    print ' '.join(subjects_shuffled)
    mebs = stom.values()
    random_map = {}
    for i in xrange(len(subjects_shuffled)):
        random_map[subjects_shuffled[i]] = mebs[i]
    matches = get_matches_distribution(random_map, get_quantized_outputs(network, random_map, files_base, sample_size, threshold)) 
    matches = map(lambda x: float(x)/float(EVAL_SAMPLE_SIZE), matches) 
    print '*raw matching scores for those subjects*'
    print ' '.join(map(lambda x: str(x), matches))
    return matches


def get_genuine_distribution(network, stom, files_base, sample_size, threshold=0.5):
    matches = get_matches_distribution(stom, get_quantized_outputs(network, stom, files_base, sample_size, threshold)) 
    matches =  map(lambda x: float(x)/float(EVAL_SAMPLE_SIZE), matches)
    print '*subjects used as genuines*'
    print ' '. join(stom.keys())
    print '*raw matching scores for genuines*'
    print ' '. join(map(lambda x: str(x), matches))
    return matches


def get_performance_measures(true_genuine, genuine_dist, 
                             true_imposter, imposter_dist):
    dist_path = os.path.join(SAVE_FOLDER, 'dist')

    if not os.path.exists(dist_path):
        os.mkdir(dist_path)

    all_true = true_genuine[:]
    all_dist = genuine_dist[:]
    all_true.extend(true_imposter)
    all_dist.extend(imposter_dist)

    eer = perf.equal_error_rate(all_true, all_dist)
    gar = perf.gar_at_zero_far_by_iterating(all_true, all_dist)

    return (eer, gar)


def evaluate_network(network, stom, weights_filename):
    genuine_dist = get_genuine_distribution(network, stom, EVAL_SET_BASE, EVAL_SAMPLE_SIZE, 0.5)
    imposter_dist = get_imposter_distribution(network, stom, EVAL_SET_BASE, EVAL_SAMPLE_SIZE, 0.5)

    true_genuine = [1 for _ in xrange(len(genuine_dist))]
    true_imposter = [0 for _ in xrange(len(imposter_dist))]

    return get_performance_measures(true_genuine, genuine_dist,
                               true_imposter, imposter_dist)

if __name__ == '__main__':
    path_to_weights = os.path.join(SAVE_FOLDER, 'weights')
    checkpoint_files = filter(lambda x: len(x.split('.')) == 2 and x.split('.')[1] == 'ckpt', os.listdir(path_to_weights))
    
    with open(os.path.realpath(PATH_FTOS), 'r') as f:
        ftos = json.load(f)

    with open(os.path.realpath(PATH_STOM),'r') as f:
        stom = json.load(f)

    stom_new={}
    for s in SUBJ_FOR_TRAINING:
        stom_new[s] = stom[s]
    
    network = vggn.VGGFaceMEB(EVAL_SAMPLE_SIZE)

    for f in sorted(checkpoint_files)[::-1]:
        network.load_all_weights(os.path.join(path_to_weights, f))
        eers = []
        gars = []
        for i in xrange(15):
            print '***********starting file', f, 'iteration', i, '***********'
            eer, gar = evaluate_network(network, stom_new, f)
            print '*eer for file', f, 'iteration', i, ':', eer
            print '*gar for file', f, 'iteration', i, ':', gar
            print '*********** end iteration ***********'
            sys.stdout.flush()
            eers.append(eer)
            gars.append(gar)
        eers = filter(lambda x: x is not None, eers)
        gars = filter(lambda x: x is not None, gars)
        print '****************** Final Results ******************'
        print 'eers (filtered)', eers, 'average: ', np.mean(eers), '+/-', np.std(eers)
        print 'gars (filtered)', gars, 'average: ', np.mean(gars), '+/-', np.std(gars)
