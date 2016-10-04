'''

This script is set up to compute to compute the ROC AUC score for a dataset of faces.
In particular, the following assumptions are made:
    1. There is a dataset in ../datasets/ that is contained in a folder. 
    That folder contains a certain number of subfolders each corresponding to 
    an individual person. Each person's folder contains one or more facial images of the person.
    This organization follows the organization of the LFW dataset.
    This is the first argument to the script

    2. There is a test file in ../datasets/ that contains lines of pairs of faces.
    Each pair contains the exact location of an image relative to the dataset root folder.
    The locations are separated by an empty space and there are no empty spaces in the path.
    The location of the file relative to ../datasets/ is the second argument to this script.

    3. An implementation of the VGGFace network is present in ./vggface/

For each input pair, the feature vector of each of the faces is computed and the Euclidean distance between the two is computed.
This generates a distribution of Euclidean distances which is then used to compute the area under the ROC curve and used as the final score.

'''
# importat note: numpy and cv2 must be imported before tensorflow because of a bug in tensorflow header loading
# this bug has been fixed, see here: https://github.com/tensorflow/tensorflow/issues/1924
import numpy as np
import cv2
from scipy.spatial.distance import euclidean 
from sklearn.metrics import roc_auc_score
import sys
import os
import vggface

def load_image(path):
    '''
    loads a single image, normalizes it and returns it as a NumPy array 
    '''
    img = cv2.imread(path)

    if img is None:
        raise Exception("Image at path " + path + " not found. Check path or numpy, cv2, tensorflow import order.")

    img2 = img.astype(np.float32)

    #subtract the average so that all pixels are close to the average
    img2 -= [129.1863,104.7624,93.5940]

    # have to resize it here to comply with VGGFace specs
    img2 = cv2.resize(img2, (224, 224))
    img2 = np.array([img2,])  
    return img2


def get_paths_from_args():
    '''
    parses the command-line arguments and returns a tuple
    of the form (path_to_single_image, path_to_test_file)
    '''
    # command line arguments expected will be:
    # position 0 is the file name
    # position 1: dataset path relative to datasets/
    # position 2: test file path relative to datasets/ 
    if len(sys.argv) < 3:
        print "Two command line arguments expected: dataset and test file."
        print "Found {0}".format(len(sys.argv)-1)
        print "Exiting..."
        sys.exit(1)

    DATASETS_BASE = './datasets/'

    # convert to absolute path for current operating system and location
    path_to_data = os.path.realpath(DATASETS_BASE + sys.argv[1])
    path_to_test = os.path.realpath(DATASETS_BASE + sys.argv[2])

    return path_to_data, path_to_test

def parse_test_file(path):
    '''
    parses the input test file
    The format of the file is expected to contain lines of pairs of image names
    separated by one or more whitespaces
    It returns an array of the pairs
    '''

    image_names = []
    delimiter = '\t'
    with open(path, 'r') as f:
        for line in f:
            line_split = line.split(delimiter)
            pair = (line_split[0].strip(), line_split[len(line_split)-1].strip())
            image_names.append(pair)
    return image_names
    

if __name__ == "__main__":

    (path_data, path_test) = get_paths_from_args()
    names_with_locations = parse_test_file(path_test)
    
    # initialize the network and load weights from a file
    # the VGGFace class takes care of sessions, etc and all the internal tensorflow stuff
    network = vggface.VGGFace()
    network.load_weights('./vggface/initial.ckpt')

    # this array will hold the distribution of distances for each of the 1000 pairs
    dist_distribution = []

    # go through each pair, get output vector and compute euclidean dist
    # then add that distance to the dist_distribution
    for (lhs, rhs) in names_with_locations:
        img1 = load_image(path_data + '/' + lhs)
        img2 = load_image(path_data + '/' + rhs) 
        v1 = network.get_l2_vector(img1)
        v2 = network.get_l2_vector(img2)
        dist = euclidean(v1, v2)
        dist_distribution.append(dist)

    # we will assume by specification that the first 500 pairs are the same person
    # and the second 500 are different people (i.e. we fed the network the faces of two different people)
    # that's our ground truth: 0 Euclidean dist between the vectors of the same people, 1 for different
    true = [0 for i in xrange(500)]
    true.extend([1 for i in xrange(500)])
    
    print 'The ROC AUC score is', roc_auc_score(true, dist_distribution)
