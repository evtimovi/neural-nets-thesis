from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
    # first arg specifies the type of data to be held
    # second arg specifies the shape, None = dim can be of any length
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    
    # x is a 1 by 784  matrix of input values
    # W is a 784 by 10 matrix of weights
    # b is a 10 by 1 matrix of biases
    # W*x = 1 by 10 matrix of unbiased outputs
    # W*x + b is the actual final output (plus biases) as a 10-bit one-hot vector
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None,10]) #"target" output

    # note that the y_ is supposed to be the target output, not the actual one (y)
    # I was missing an underscore here
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i%100 == 0:
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    
