import parent

class VGGFaceWithL2(parent.VGGFace):
    def __init__(self, gpu):
        self.gpu = gpu
        with parent.tf.device(self.gpu):
            super(VGGFaceWithL2, self).__init__()

            #turn on normalization for the last VGG layer
            self.layers[-1] = (self.layers[-1][0], self.layers[-1][1], self.layers[-1][2], self.layers[-1][3], True)
            print self.layers[-1]
            self._setup_network_variables()
            self.saver = parent.tf.train.Saver()

    def load_weights(self, path):
        with parent.tf.device(self.gpu):
            self.saver.restore(self.sess, path)

    def forwardprop(self, img):
        with parent.tf.device(self.gpu):
            x_image = self.vars[0][1]
            return self.get_output().eval(feed_dict={x_image:img})[0]


class VGGFaceWithoutL2(parent.VGGFace):
    def __init__(self, gpu):
        self.gpu = gpu
        with parent.tf.device(self.gpu):
            super(VGGFaceWithoutL2, self).__init__()
            print self.layers[-1]
            self._setup_network_variables()
            self.saver = parent.tf.train.Saver()

    def load_weights(self, path):
        with parent.tf.device(self.gpu):
            self.saver.restore(self.sess, path)

    def forwardprop(self, img):
        with parent.tf.device(self.gpu):
            x_image = self.vars[0][1]
            return self.get_output().eval(feed_dict={x_image:img})[0]
