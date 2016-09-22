'''

This script is a cleaned-up version of get_feature_vectors.py

'''
import vggface
import tensorflow as tf
import numpy as np
import cv2
import sys


def load_image(path):
    '''
    The following method will load a single image and convert it to grayscale
    '''
    img = cv2.imread(path)
    img2 = img.astype(np.float32)
    img2 -= [129.1863,104.7624,93.5940]
    img2 = np.array([img2,])  
    return img2


def get_vector(path, network_evaluated):
    # load faces
    img1 = load(path)
    
    # extract features
    output1 = network_evaluated.eval(feed_dict={x_image:img1})[0]
    norm_output1 = output1/np.linalg.norm(output1,2)
    return norm_output1



if __name__ == "__main__":
    # command line arguments expected will be:
    # position 0 is the file name
    # position 1: dataset path relative to this file
    # position 2: test file path relative to this file
    if len(sys.argv) != 3:
        print "Two command line arguments expected: dataset and test file."
        print "Found {0}".format(len(sys.argv)-1)
        print "Exiting..."
        sys.exit(1)

    path_to_data = sys.argv[1]
    path_to_test = sys.argv[2]
    
    # parse the input test file 
    # to an array of pairs of names
    image_names = []
    delimiter = '\t'
    with open(path_to_test, 'r') as f:
        for line in f:
            line_split = line.split(delimiter)
            pair = (line_split[0].strip(), line_split[len(line_split)-1].strip())
            image_names.append(pair)
   
    # set up tensorflow network
    x_image = tf.placeholder(tf.float32, shape=[1,250,250,3]) 

    # initialize session and set up variables for VGGFace 
    # (VGGFace setup provided by external script)
    sess = tf.InteractiveSession()
    network = vggface.VGGFace()

    # restore the variables in initial.ckpt into this sess object
    # this will essentially load the weights and biases
    saver = tf.train.Saver()
    saver.restore(sess, "./vggface/initial.ckpt")

    network_evaluated = network.network_eval(x_image)   
   
    print get_vector(path_to_data, network_evaluated)


