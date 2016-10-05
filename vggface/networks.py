'''
This module holds all useful implementations of networks based on 
VGG Face. You should put all child classes in here.
'''

#from parent import VGGFace
#import tensorflow as tf
import parent 

'''
This class inherits from the paretn VGGFace to build a 
vanilla VGGFace implementation meant only for evaluation.
'''

class VGGFaceVanilla(parent.VGGFace):

    def __init__(self):

        # initialize everything in the parent
        super(VGGFaceVanilla, self).__init__()

        # append an l2 layer
        self.layers.append(('l2','37',4096,True))

        # initialize everything else
        self._setup_network_variables()
        self.saver = parent.tf.train.Saver()
    
    def load_weights(self, path):
        '''
        This method initializes the network weights from 
        a .ckpt file to be found in path
        '''
        self.saver.restore(self.sess, path)

    def get_l2_vector(self, img): 
        '''
        This method runs img through the network to obtain
        its l2-embedded representation.
        '''
        # we appended the input placeholder to the beginning of the vars array
        # the actual variable there is in the second position
        x_image = self.vars[0][1]
        return self.get_output().eval(feed_dict={x_image:img})[0]
