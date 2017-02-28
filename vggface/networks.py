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
    def __init__(self, batch_size, gpu="/gpu:0", keysize=256, max_checkpoints=160):
        self.gpu = gpu
        with parent.tf.device(self.gpu):
             # initialize everything in the parent
            super(VGGFaceMEB, self).__init__(batch_size)

            # append a linear layer of keysize neurons
            # for linear the syntax is:
            # (layer_type, layer_number, num_neurons, activation_function)
            self.layers.append(('linear', '41', keysize, 'sigmoid'))

            # this will hold the MEB codes we are aiming to train to
            self.target_code = parent.tf.placeholder(parent.tf.float32, shape=[None,keysize])

            # initialize everything else (including TF variables)
            self._setup_network_variables()

            self.sess.run(parent.tf.initialize_variables(filter(lambda x: x.name.startswith('linear_3'),parent.tf.all_variables())))

            self.saver = parent.tf.train.Saver(max_to_keep=max_checkpoints)
            # restorer is for VGG variables only
            # saver is for all variables
            self.restorer = parent.tf.train.Saver(filter(lambda x: not x.name.startswith('linear_3'),parent.tf.all_variables()))
            self.weights_loaded = False

    def load_vgg_weights(self, path):
        '''
        This method initializes the network weights from 
        a .ckpt file to be found in path
        '''
        with parent.tf.device(self.gpu):
            self.restorer.restore(self.sess, path)
            self.weights_loaded = True

    def load_all_weights(self, path):
        with parent.tf.device(self.gpu):
            self.saver.restore(self.sess, path)
            self.weights_loaded = True
    
    def train_batch(self, inputs_arr, targets_arr,
                    learning_rate,
                    all_layers=False,
                    loss = 'euclidean'):
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
            loss: a string specifying the loss operation to use - can be 'logloss' or 'euclidean' (default: euclidean)
        Returns:
            the cross entropy loss after training the batch
        '''

        with parent.tf.device(self.gpu):
            y_target = self.target_code
            y_output = self.get_output()
        
            if loss == 'logloss':
                #cross_entropy
                ce_1 = parent.tf.mul(y_target, parent.tf.log(y_output))
                ce_2 = parent.tf.mul(parent.tf.sub(1.0, y_target), parent.tf.log(parent.tf.sub(1.0, y_output)))
                loss_op = -parent.tf.reduce_mean(parent.tf.reduce_sum(parent.tf.add(ce_1, ce_2), 1))
            elif loss == 'euclidean':
                #euclidean
                loss_op = parent.tf.reduce_mean(parent.tf.sqrt(parent.tf.reduce_sum(parent.tf.square(parent.tf.sub(y_output, y_target)), 1)))
        
            if all_layers:
                train_step =  parent.tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
            else:
                if not self.weights_loaded:
                    print '***** Warning: The VGG weights are random, but only the last layer is being trained! *****' 
                vars_to_train = filter(lambda v: v.name.startswith('linear_3'), parent.tf.trainable_variables()) 
                train_step = parent.tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op, var_list=vars_to_train)

            # vars at 0 holds the input placeholder in a tuple
            # the second entry in the tuple is the actual placeholder
            _, loss = self.sess.run([train_step, loss_op],
                                 feed_dict={self.vars[0][1]: inputs_arr, self.target_code: targets_arr})
        
            return loss

    def get_raw_output_for(self, img):
        '''
        This method runs the image img through the network
        and returns the output.
        '''
        with parent.tf.device(self.gpu):

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
        with parent.tf.device(self.gpu):

            # we appended the input placeholder to the beginning of the vars array
            # the actual variable there is in the second position
            x_image = self.vars[0][1]
            output = self.get_output().eval(feed_dict={x_image:img})[0]
            return map(lambda x: 0 if x<threshold else 1, output)

    def get_avg_euclid(self, inputs_arr, targets_arr):
        with parent.tf.device(self.gpu):

            euclidean_mean_op = parent.tf.reduce_mean(parent.tf.sqrt(parent.tf.reduce_sum(parent.tf.square(parent.tf.sub(self.get_output(), self.target_code)), 1)))
            _, avg_euclidean = self.sess.run([self.get_output(), euclidean_mean_op], 
                                         feed_dict={self.vars[0][1]: inputs_arr, self.target_code: targets_arr})
            return avg_euclidean

'''
This class inherits from the paretn VGGFace to build a 
vanilla VGGFace implementation meant only for evaluation.
'''
class VGGFaceVanilla(parent.VGGFace):

    def __init__(self, gpu):
        self.gpu = gpu
        with parent.tf.device(self.gpu):
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
        with parent.tf.device(self.gpu):
            self.saver.restore(self.sess, path)

    def get_l2_vector(self, img): 
        '''
        This method runs img through the network to obtain
        its l2-embedded representation.
        '''
        # we appended the input placeholder to the beginning of the vars array
        # the actual variable there is in the second position
        with parent.tf.device(self.gpu):
            x_image = self.vars[0][1]
            return self.get_output().eval(feed_dict={x_image:img})[0]

'''
This class inherits from the paretn VGGFace to build a 
vanilla VGGFace implementation meant only for evaluation.
The last layer of this network does not apply l2 
normalization.
'''
class VGGFaceVanillaNoL2(parent.VGGFace):

    def __init__(self, gpu):
        self.gpu = gpu
        with parent.tf.device(self.gpu):
            # initialize everything in the parent
            super(VGGFaceVanillaNoL2, self).__init__()

            # append an l2 layer
#            self.layers.append(('l2','37',4096,True))

            # initialize everything else
            self._setup_network_variables()
            self.saver = parent.tf.train.Saver()
    
    def load_weights(self, path):
        '''
        This method initializes the network weights from 
        a .ckpt file to be found in path
        '''
        with parent.tf.device(self.gpu):
            self.saver.restore(self.sess, path)

    def get_output_for_img(self, img): 
        '''
        This method runs img through the network to obtain
        the unnormalized output.
        '''
        # we appended the input placeholder to the beginning of the vars array
        # the actual variable there is in the second position
        with parent.tf.device(self.gpu):
            x_image = self.vars[0][1]
            return self.get_output().eval(feed_dict={x_image:img})[0]


'''
This network applies normalization
before passing to the MEB layer.
'''
class VGGFaceMEBWithL2(parent.VGGFace):
    def __init__(self, batch_size, gpu="/gpu:0", keysize=256, max_checkpoints=160):
        self.gpu = gpu
        with parent.tf.device(self.gpu):
            super(VGGFaceMEBWithL2, self).__init__(batch_size)
            self.layers.append(('l2','37'))

            # (layer_type, layer_number, num_neurons, activation_function)
            self.layers.append(('linear', '41', keysize, 'sigmoid'))

            # this will hold the MEB codes we are aiming to train to
            self.target_code = parent.tf.placeholder(parent.tf.float32, shape=[None,keysize])

            # initialize everything else (including TF variables)
            self._setup_network_variables()
            self.sess.run(parent.tf.initialize_variables(filter(lambda x: x.name.startswith('linear_3'),parent.tf.all_variables())))
            
            # restorer is for VGG variables only, saver is for all variables

            self.saver = parent.tf.train.Saver(max_to_keep=max_checkpoints)
            self.restorer = parent.tf.train.Saver(filter(lambda x: not x.name.startswith('linear_3'),parent.tf.all_variables()))
            self.weights_loaded = False

    def load_vgg_weights(self, path):
        '''
        This method initializes the network weights from 
        a .ckpt file to be found in path
        '''
        with parent.tf.device(self.gpu):
            self.restorer.restore(self.sess, path)
            self.weights_loaded = True

    def load_all_weights(self, path):
        with parent.tf.device(self.gpu):
            self.saver.restore(self.sess, path)
            self.weights_loaded = True
    
    def train_batch(self, inputs_arr, targets_arr,
                    learning_rate,
                    all_layers=False,
                    loss = 'euclidean'):
        with parent.tf.device(self.gpu):
            y_target = self.target_code
            y_output = self.get_output()
        
            if loss == 'logloss':
                #cross_entropy
                ce_1 = parent.tf.mul(y_target, parent.tf.log(y_output))
                ce_2 = parent.tf.mul(parent.tf.sub(1.0, y_target), parent.tf.log(parent.tf.sub(1.0, y_output)))
                loss_op = -parent.tf.reduce_mean(parent.tf.reduce_sum(parent.tf.add(ce_1, ce_2), 1))
            elif loss == 'euclidean':
                #euclidean
                loss_op = parent.tf.reduce_mean(parent.tf.sqrt(parent.tf.reduce_sum(parent.tf.square(parent.tf.sub(y_output, y_target)), 1)))
        
            if all_layers:
                train_step =  parent.tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op)
            else:
                if not self.weights_loaded:
                    print '***** Warning: The VGG weights are random, but only the last layer is being trained! *****' 
                vars_to_train = filter(lambda v: v.name.startswith('linear_3'), parent.tf.trainable_variables()) 
                train_step = parent.tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op, var_list=vars_to_train)

            # vars at 0 holds the input placeholder in a tuple
            # the second entry in the tuple is the actual placeholder
            _, loss = self.sess.run([train_step, loss_op],
                                 feed_dict={self.vars[0][1]: inputs_arr, self.target_code: targets_arr})
        
            return loss

    def get_raw_output_for(self, img):
        '''
        This method runs the image img through the network
        and returns the output.
        '''
        with parent.tf.device(self.gpu):

            x_image = self.vars[0][1]
            return self.get_output().eval(feed_dict={x_image:img})[0]

    def save_weights(self, path):
        '''
        Uses the saver to save the session.
        To avoid confusion, please provide an absolute path
        by calling os.path.relpath(...) from the calling script.
        '''
        self.saver.save(self.sess, path)

    def get_meb_for(self, img, threshold=0.5): 
        '''
        This method runs img through the network to obtain
        an MEB code. The quantization is performed with the given
        threshold (default: 0.5)
        '''
        with parent.tf.device(self.gpu):
            # we appended the input placeholder to the beginning of the vars array
            # the actual variable there is in the second position
            x_image = self.vars[0][1]
            output = self.get_output().eval(feed_dict={x_image:img})[0]
            return map(lambda x: 0 if x<threshold else 1, output)

    def get_avg_euclid(self, inputs_arr, targets_arr):
        with parent.tf.device(self.gpu):
            euclidean_mean_op = parent.tf.reduce_mean(parent.tf.sqrt(parent.tf.reduce_sum(parent.tf.square(parent.tf.sub(self.get_output(), self.target_code)), 1)))
            _, avg_euclidean = self.sess.run([self.get_output(), euclidean_mean_op], 
                                         feed_dict={self.vars[0][1]: inputs_arr, self.target_code: targets_arr})
            return avg_euclidean
