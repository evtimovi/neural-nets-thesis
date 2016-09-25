'''

This script is a cleaned-up version of get_feature_vectors.py

'''
# importat note: numpy and cv2 must be imported before tensorflow because of a bug in tensorflow header loading
# this bug has been fixed, see here: https://github.com/tensorflow/tensorflow/issues/1924
import numpy as np
import cv2
import sys
import os
import vggface
import tensorflow as tf

def load_image(path):
    '''
    The following method will load a single image and convert it to grayscale
    '''
    print '************loading image'

    img = cv2.imread('/home/evtimovi/neural-nets-thesis/datasets/lfw/Al_Gore/Al_Gore_0001.jpg')

    print '************loaded image'

    img2 = img.astype(np.float32)
    
    print '************image cast to float32'

    #subtract the average so that all pixels are close to the average
    img2 -= [129.1863,104.7624,93.5940]


    # have to resize it here to comply with VGGFace specs
    img2 = cv2.resize(img2, (224, 224))

    print '************image resiz done'

    img2 = np.array([img2,])  

    print '************after np.array([img2,]'

    return img2


def get_vector(path, network):
    # load faces
    img1 = load_image(path)
    
    print '************loaded image, now starting network eval'

    # extract features
    output1 = network.eval(feed_dict={x_image:img1})[0]

    print '************network eval completed'

    norm_output1 = output1/np.linalg.norm(output1,2)
    return norm_output1

def get_paths_from_args():
    '''
    This method parses the command-line arguments and returns a tuple
    of the form (path_to_single_image, path_to_test_file)
    '''
    # command line arguments expected will be:
    # position 0 is the file name
    # position 1: dataset path relative to datasets/
    # position 2: test file path relative to datasets/ 
    if len(sys.argv) != 3:
        print "Two command line arguments expected: dataset and test file."
        print "Found {0}".format(len(sys.argv)-1)
        print "Exiting..."
        sys.exit(1)

    DATASETS_BASE = '../datasets/'
    path_to_data = os.path.realpath(DATASETS_BASE + sys.argv[1])
    path_to_test = os.path.realpath(DATASETS_BASE + sys.argv[2])

    return path_to_data, path_to_test

def parse_test_file(path):
    '''
    This method parses the input test file
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
    names = parse_test_file(path_test)
  
    # set up tensorflow network
    # the shape of the placeholder is determined by the VGGFace specs
    # the image has to be resized to fit those
    x_image = tf.placeholder(tf.float32, shape=[1,224,224,3]) 

    # initialize session and set up variables for VGGFace 
    # (VGGFace setup provided by external script)
    sess = tf.InteractiveSession()
    network = vggface.VGGFace()
    network.load(sess, x_image); 

    print '************network loaded'

    # restore the variables in initial.ckpt into this sess object
    # this will essentially load the weights and biases
    saver = tf.train.Saver()

    print '************saver initiated'
    saver.restore(sess, "./vggface/initial.ckpt")
    print '************restore completed'

    print '************calling get vectors'
    print get_vector(path_data, network)
