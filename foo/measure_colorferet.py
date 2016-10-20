# importat note: numpy and cv2 must be imported before tensorflow because of a bug in tensorflow header loading
# this bug has been fixed, see here: https://github.com/tensorflow/tensorflow/issues/1924
import numpy as np
import cv2
from scipy.spatial.distance import euclidean 
import sys
import os
from util import performance as p
from util import processimgs as pimg
from vggface import networks as vggn



def parse_test_file(path):
    '''
    parses the input test file
    The format of the file is expected to contain lines of pairs of image names
    separated by one or more whitespaces
    It returns an array of the pairs
    '''

    image_names = []
    delimiter = ' '
    with open(path, 'r') as f:
        for line in f:
            line_split = line.split(delimiter)
            pair = (line_split[0].strip(), line_split[len(line_split)-1].strip())
            image_names.append(pair)
    return map(lambda x: (os.path.realpath(x[0]),os.path.realpath(x[1])), image_names)
    

if __name__ == "__main__":
    path_test = os.path.realpath(sys.argv[1])
    out_file = os.path.realpath(sys.argv[2])
    names_with_locations = parse_test_file(path_test)
    
    # initialize the network and load weights from a file
    # the VGGFace class takes care of sessions, etc and all the internal tensorflow stuff
    network = vggn.VGGFaceVanilla()
    network.load_weights(os.path.realpath('./vggface/weights/initial.ckpt'))


    # this array will hold the distribution of distances for each of the 1000 pairs
    dist_distribution = []

    # go through each pair, get output vector and compute euclidean dist
    # then add that distance to the dist_distribution
    for (lhs, rhs) in names_with_locations:
        img1 = pimg.load_crop_adjust(lhs)
        img2 = pimg.load_crop_adjust(rhs) 
        v1 = network.get_l2_vector(img1)
        v2 = network.get_l2_vector(img2)
        dist = euclidean(v1, v2)
        dist_distribution.append(dist)

    # we will assume by specification that the first 500 pairs are the same person
    # and the second 500 are different people (i.e. we fed the network the faces of two different people)
    with open(out_file, 'w') as f:
        f.write('euclidean_dist=' + str(dist_distribution) + '\n')
