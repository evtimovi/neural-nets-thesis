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
    def __init__(self, keysize=256):
         # initialize everything in the parent
        super(VGGFaceVanilla, self).__init__()

        # append an l2 layer
        self.layers.append(('l2','37',4096,True))

        # append a linear layer of keysize neurons
        # for linear the syntax is:
        # (layer_type, layer_number, num_neurons, use_relu)
        self.layers.append(('linear', '41', keysize, True))

        # initialize everything else (including TF variables)
        self._setup_network_variables()

        # this will hold the MEB codes we are aiming to train to
        target_code = tf.placeholder(tf.float32, shape=[None,keysize])

        self.saver = parent.tf.train.Saver()
        self.weights_loaded = False

    def load_weights(self, path):
        '''
        This method initializes the network weights from 
        a .ckpt file to be found in path
        '''
        self.saver.restore(self.sess, path)
        self.weights_loaded = True

    
    def train_batch(self, inputs_arr, targets_arr,
                    learning_rate,
                    all_layers=False):
        '''
        This method will allow you to run one batch of training
        based on the specified inputs, targets, and learning rate.
        If all_layers is set to False, then only the layers
        involved in calculating an MEB code on top of 
        the VGG network will be trained.
        Otherwise, all layers in the network are trained from scratch.
        are trained. 
        This method does not handle saving checkpoints or partitioning
        in batches - it is assumed those will be done outside.
        Args:
            input_arr: a numpy array of all the inputs in the batch 
            targets_arr: a numpy array of the target meb codes corresponding 
        The intention is to provide all inputs for this batch
        as a numpy array in input_arr
        and all corresponding target outputs as another array in targets_arr
        The argument checkpoint_step allows you to set checkpoints every
        <checkpoint_step> number of training iterations (defaults to 100)
        The argument all_layers allows you to specify whether all layers
        of the network should be trained or just the last one.
        '''
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(target_code * tf.log(self.get_output()), reduction_indices=[1]))
        
        if all_layers:
            train_step =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        else:
            if not self.weights_loaded:
                print '***** Warning: The VGG weights are random, but only the last layer is being trained! *****' 
            var_to_train = filter(lambda v: v.name.startswith('linear_2'), tf.trainable_variables()) 
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, var_list=vars_to_train)

        train_step.run(feed_dict={x_image: inputs_arr, target_code: targets_arr})

        def save_weights(self, path):
            '''
            Uses the saver to save the session.
            To avoid confusion, please provide an absolute path
            by calling os.path.relpath(...) from the 
            calling script.
            '''
            self.saver.save(self.sess, path)

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
