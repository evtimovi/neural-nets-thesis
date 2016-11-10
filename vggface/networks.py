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
class VGGFaceMEB(parent.VGGFace):
    def __init__(self, batch_size, keysize=256):
         # initialize everything in the parent
        super(VGGFaceTrainForMEB, self).__init__(batch_size)

        # append a linear layer of keysize neurons
        # for linear the syntax is:
        # (layer_type, layer_number, num_neurons, activation_function)
        self.layers.append(('linear', '41', keysize, 'sigmoid'))

        # this will hold the MEB codes we are aiming to train to
        self.target_code = parent.tf.placeholder(parent.tf.float32, shape=[None,keysize])

        # initialize everything else (including TF variables)
        self._setup_network_variables()

        self.sess.run(parent.tf.initialize_variables(filter(lambda x: x.name.startswith('linear_3'),parent.tf.all_variables())))

        self.saver = parent.tf.train.Saver()
        # restorer is for VGG variables only
        # saver is for all variables
        self.restorer = parent.tf.train.Saver(filter(lambda x: not x.name.startswith('linear_3'),parent.tf.all_variables()))
        self.weights_loaded = False

    def load_vgg_weights(self, path):
        '''
        This method initializes the network weights from 
        a .ckpt file to be found in path
        '''
        self.restorer.restore(self.sess, path)
        self.weights_loaded = True

    def load_all_weights(self, path):
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
            targets_arr: a numpy array of the target meb codes
                         (indices should match avg_euclidean_mean up between input_arr and targets_arr)
            learning_rate: the learning rate to be used during training
            all_layers: allows you to specify whether all layers
                        of the network should be trained or just the last one.
                        defaults to False (only train the MEB layer)
        Returns:
            the cross entropy loss after training the batch
        '''

        cross_entropy = parent.tf.reduce_mean(-parent.tf.reduce_sum(self.target_code * parent.tf.log(self.get_output()), reduction_indices=[1]))
        
        if all_layers:
            train_step =  parent.tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        else:
            if not self.weights_loaded:
                print '***** Warning: The VGG weights are random, but only the last layer is being trained! *****' 
            vars_to_train = filter(lambda v: v.name.startswith('linear_3'), parent.tf.trainable_variables()) 
            train_step = parent.tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, var_list=vars_to_train)

        # vars at 0 holds the input placeholder in a tuple
        # the second entry in the tuple is the actual placeholder
        _, loss = self.sess.run([train_step, cross_entropy],
                                 feed_dict={self.vars[0][1]: inputs_arr, self.target_code: targets_arr})
        
        return loss

    def get_raw_output_for(self, img):
        '''
        This method runs the image img through the network
        and returns the output.
        '''
        x_image = self.vars[0][1]
        return self.get_output().eval(feed_dict={x_image:img})[0]

    def save_weights(self, path):
        '''
        Uses the saver to save the session.
        To avoid confusion, please provide an absolute path
        by calling os.path.relpath(...) from the 
        calling script.
        '''
        self.saver.save(self.sess, path)

    def get_meb_for(self, img, threshold=0.5): 
        '''
        This method runs img through the network to obtain
        an MEB code. The quantization is performed with the given
        threshold (default: 0.5)
        '''
        # we appended the input placeholder to the beginning of the vars array
        # the actual variable there is in the second position
        x_image = self.vars[0][1]
        output = self.get_output().eval(feed_dict={x_image:img})[0]
        return map(lambda x: 0 if x<threshold else 1, output)

    def get_avg_euclidaen(self, inputs_arr, targets_arr):
        euclidean_mean_op = parent.tf.reduce_mean(parent.tf.sqrt(parent.tf.reduce_sum(parent.tf.sub(self.get_output(), self.target_code))))
        _, avg_euclidean = self.sess.run([self.get_output(), euclidean_mean_op], 
                                         feed_dict={self.vars[0][1]: inputs_arr, self.target_code: targets_arr})
        return avg_euclidean

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
