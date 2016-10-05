'''
This module holds all useful implementations of networks based on 
VGG Face. You should put all child classes in here.
'''

import parent 
# note that tensorflow is imported in parent as tf,
# so it can be referred to as parent.tf

'''
This class inherits from the parent VGGFace to build 
and train a neural network that maps faces to MEB codes
(the MEB codes will be provided externally)
'''
class VGGFaceTrainForMEB(parent.VGGFace):
    def __init__(self):
         # initialize everything in the parent
        super(VGGFaceVanilla, self).__init__()

        # append an l2 layer
        self.layers.append(('l2','37',4096,True))

        # append a linear layer of 1024 neurons
        # for linear the syntax is:
        # (layer_type, layer_number, num_neurons, use_relu)
        self.layers.append(('linear', '41', 1024, True))

        # initialize everything else (including TF variables)
        self._setup_network_variables()
        self.saver = parent.tf.train.Saver()
    
    def train(inputs_arr, targets_arr,
              batch_size, learning_rate,
              checkpoint_step=100, all_layers=False):
        '''
        This method will allow you to train the network 
        based on the specified inputs and targets
        that will be partitioned in the specified batch_size
        The intention is to provide all inputs as an array in input_arr
        and all corresponding target outputs as another array in targets_arr
        The argument checkpoint_step allows you to set checkpoints every
        <checkpoint_step> number of training iterations (defaults to 100)
        The argument all_layers allows you to specify whether all layers
        of the network should be trained or just the last one.
        '''
        return null

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
