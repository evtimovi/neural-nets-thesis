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
        inputs_arr = map(lambda x: pimg.load_adjust_avg(os.path.join(subj_path,x)), sampled_files)
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
        subj_path = os.path.join(files_base, s)
        img_files = os.listdir(subj_path)

        # skip those fb's that are not in the fa's (i.e. in the stom dictionary)
        # and skip subjects without fb's 
        if (not os.path.exists(subj_path)) or (len(img_files) < sample_size):
            print 'skipping*subject*without*fa*', s
            sys.stdout.flush()
            sys.stderr.write("subject*" + s + "*has*no*fbs*\n")
            continue

        subject_to_trained_mebs[s] = []
        img_files = random.SystemRandom().sample(img_files, sample_size)

        for f in img_files:
            img = pimg.load_adjust_avg(os.path.join(subj_path, f))
            meb = network.get_raw_output_for([img,])
            subject_to_trained_mebs[s].append(map(lambda x: 1 if x > threshold else 0, meb))
    return subject_to_trained_mebs


def get_matches_distribution(true_stom, network_stom, label):
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
    subjects = network_stom.keys()

    match_scores = map(lambda s: str(float(network_stom[s].count(true_stom[s]))/float(EVAL_SAMPLE_SIZE)), subjects)

    print (label+'*subjects*'), ' '.join(subjects)
    print (label+'*scores*'), ' '.join(match_scores)
    sys.stdout.flush()

    return match_scores


def get_imposter_distribution(network, stom, files_base, sample_size, threshold=0.5):
    subjects = stom.keys()
    subjects_shuffled = subjects[:]
    random.SystemRandom().shuffle(subjects_shuffled)

    random_map = {}
    firstpart=[]
    secondpart=[]
    for s1, s2 in zip(subjects,subjects_shuffled):
        random_map[s2] = stom[s1]
        firstpart.append(s1)
        secondpart.append(s2)

    netout = get_quantized_outputs(network, random_map, files_base, sample_size, threshold)
    matches = get_matches_distribution(random_map, netout, "*imposters*") 

    print '*imposters*left*part*', ' '.join(firstpart)
    print '*imposters*right*part*', ' '.join(secondpart)
    print '*imposter*pair*scores*', ' '.join(matches)
    sys.stdout.flush()

    return matches


def get_genuine_distribution(network, stom, files_base, sample_size, threshold=0.5):
    netout = get_quantized_outputs(network, stom, files_base, sample_size, threshold)
    matches = get_matches_distribution(stom, netout, '*genuines*') 
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


def evaluate_network(network, stom, weights_filename, iteration):
    genuine_dist = get_genuine_distribution(network, stom, EVAL_SET_BASE, EVAL_SAMPLE_SIZE, 0.5)
    imposter_dist = get_imposter_distribution(network, stom, EVAL_SET_BASE, EVAL_SAMPLE_SIZE, 0.5)

#    histogram(genuine_dist, imposter_dist, weights_filename + '_' + str(iteration))

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
    
    network = vggn.VGGFaceMEB(1)

    for f in FILES:
        network.load_all_weights(os.path.join(VGG_WEIGHTS_PATH, f))
        eers = []
        gars = []
        for i in xrange(NUM_EVAL_ITERS):
            print '***********file*'+f+'*iteration*'+str(i)+'***********'
            sys.stdout.flush()
            eer, gar = evaluate_network(network, stom_new, f, i)
            print '*eer*for*file*'+f+'*iteration*'+str(i)+':', eer
            print '*gar*for*file*'+f+'*iteration*'+str(i)+':', gar
            print '***********end*iteration*'+str(i)+'*of*'+str(NUM_EVAL_ITERS)+'*for*file*'+f+'***********'
            sys.stdout.flush()
            eers.append(eer)
            gars.append(gar)
        eers = filter(lambda x: x is not None, eers)
        gars = filter(lambda x: x is not None, gars)
        print '******************Final*Results*for*file'+f+'******************'
        print 'eers (filtered)', eers, 'average: ', np.mean(eers), '+/-', np.std(eers)
        print 'gars (filtered)', gars, 'average: ', np.mean(gars), '+/-', np.std(gars)
