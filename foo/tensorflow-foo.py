import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[4,3])
y = tf.placeholder(tf.float32, shape=[4,3])
euclidean_mean_op = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x, y)), 1)))

input_x = [[5,5,5], [5,5,5], [5,5,5], [5,5,5]]
input_y = [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]

print sess.run(euclidean_mean_op, feed_dict={x: input_x, y: input_y})
