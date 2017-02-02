import tensorflow as tf

'''
This file builds the VGGFace Deep Convolutional Neural Network for face recognition
WITHOUT ITS LAST LAYER.

You should inherit from this class if you would like to add additional layers 
to the network. Note that the _setup_network_variables method will do
all necessary initialization of an input layer and convolutional, maxpooling,
softmax and l2 layers. You should add any custom initialization as an elif
in that method and just add a description in the subclass.

Note that the scope of tensorflow variables is managed by Tensorflow's
own mechanism for managing variable scope 
(see here: https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html)
and the object is constructed only so that we can save the representation of the network
in arrays that then allow us to automate the creation of multiple layers
without having to write each one out.

'''
class VGGFace(object):

    def __init__(self, batch_size=1):
        '''
        Initialize the layers array that holds a list of tuples describing each layer
        (e.g. whether it is convolutional or max pooling, strides, etc)
        '''
        self.batch_size = batch_size
        self.vars = []
        self.layers = []
        self._setup_layers_description()

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        
    def _setup_layers_description(self):
        '''
        This method fills up the self.layers variable
        with the architecture of the VGGFace network.
        It is only meant to be run as part of the initialization step.
        '''
        # (1): nn.SpatialConvolutionMM(3 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','1',3,3,3,64))
        # (3): nn.SpatialConvolutionMM(64 -> 64, 3x3, 1,1, 1,1)
        self.layers.append(('conv','3',3,3,64,64))
        # (5): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (6): nn.SpatialConvolutionMM(64 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','6',3,3,64,128))
        # (8): nn.SpatialConvolutionMM(128 -> 128, 3x3, 1,1, 1,1)
        self.layers.append(('conv','8',3,3,128,128))
        # (10): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (11): nn.SpatialConvolutionMM(128 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','11',3,3,128,256))
        # (13): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','13',3,3,256,256))
        # (15): nn.SpatialConvolutionMM(256 -> 256, 3x3, 1,1, 1,1)
        self.layers.append(('conv','15',3,3,256,256))
        # (17): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (18): nn.SpatialConvolutionMM(256 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','18',3,3,256,512))
        # (20): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','20',3,3,512,512))
        # (22): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','22',3,3,512,512))
        # (24): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (25): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','25',3,3,512,512))
        # (27): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','27',3,3,512,512))
        # (29): nn.SpatialConvolutionMM(512 -> 512, 3x3, 1,1, 1,1)
        self.layers.append(('conv','29',3,3,512,512))
        # (31): nn.SpatialMaxPooling(2,2,2,2)
        self.layers.append(('pool',2,2,2,2))
        # (32): nn.View
        # (33): nn.Linear(25088 -> 4096)
        self.layers.append(('linear','33',4096,'relu'))
        # (34): nn.ReLU
        # (35): nn.Dropout(0.500000)
        # (36): nn.Linear(4096 -> 4096)
        # (37): nn.ReLU
        self.layers.append(('linear','36',4096,'relu'))
        # (38): nn.Dropout(0.500000)
        # (39): nn.Linear(4096 -> 2622)
        # self.layers.append(('linear','39',2622,False))
        # (40): nn.SoftMax
        # self.layers.append(('softmax'))
    
    def _setup_network_variables(self):
        '''
        This method runs through and sets up the Tensorflow 
        variables as described in the self.layers variable
        It is only meant to be run as part of the initialization step.
        '''

        # first initialize the input layer
        x_image = tf.placeholder(tf.float32, shape=[self.batch_size,224,224,3]) 
        self.vars.append(('input', x_image, ['input', None]))

        # then initialize all following layers
        for layer in self.layers:
            name = self.get_unique_name_(layer[0])
            if layer[0] == 'conv':
                with tf.variable_scope(name) as scope:
                    h, w, c_i, c_o = layer[2],layer[3],layer[4],layer[5]
                    kernel = self.make_var('weights', shape=[h, w, c_i, c_o])
                    conv = tf.nn.conv2d(self.get_output(), kernel, [1]*4, padding='SAME')
                    biases = self.make_var('biases', [c_o])
                    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
                    relu = tf.nn.relu(bias, name=scope.name)
                    self.add_(name, relu,layer)
            elif layer[0] == 'pool':
                size,size,stride,stride = layer[1],layer[2],layer[3],layer[4]
                pool = tf.nn.max_pool(self.get_output(),
                                      ksize=[1, size, size, 1],
                                      strides=[1, stride, stride, 1],
                                      padding='SAME',
                                      name=name)
                self.add_(name, pool,layer)
            elif layer[0] == 'linear':
                num_out = layer[2]
                func = layer[3]
                with tf.variable_scope(name) as scope:
                    input = self.get_output()
                    input_shape = input.get_shape()
                    if input_shape.ndims==4:
                        dim = 1
                        for d in input_shape[1:].as_list():
                            dim *= d
                        feed_in = tf.reshape(input, [self.batch_size, dim])
                    else:
                        feed_in, dim = (input, int(input_shape[-1]))
                    weights = self.make_var('weights', shape=[dim, num_out])
                    biases = self.make_var('biases', [num_out])
                    
                    if func == 'relu':
                        op = tf.nn.relu
                    elif func == 'sigmoid':
                        op = tf.sigmoid
#                    else:
#                        op = tf.nn.xw_plus_b
#                    op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
                    fc = op(tf.nn.bias_add(tf.matmul(feed_in, weights), biases), name=scope.name)
                    self.add_(name, fc,layer)
            elif layer[0] == 'softmax':
                self.add_(name, tf.nn.softmax(self.get_output()),layer)
            elif layer[0] == 'l2':
                self.add_(name,tf.nn.l2_normalize(self.get_output(),0,),layer)

    # counts how many variables start with the same prefix
    # and then gives a string that's in the form prefix_number
    def get_unique_name_(self, prefix):
        id = sum(t.startswith(prefix) for t,_,_ in self.vars)+1
        return '%s_%d'%(prefix, id)

    def add_(self, name, var,layer):
        self.vars.append((name, var,layer))

    def get_output(self):
        return self.vars[-1][1]

    def make_var(self, name, shape):
        return tf.get_variable(name, shape)
