import tensorflow as tf
import scipy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

n_input = 2400 # 30 * 80
n_x = 80
n_y = 30
n_out = 2 # Steer and Speed
dropout = 0.75 

x = tf.placeholder(tf.float32, shape = [None,30,80])
#x_image = tf.reshape(x, shape = [-1,n_input])
x_image = tf.reshape(x, shape = [-1, 30, 80, 1])

y_ = tf.placeholder(tf.float32, shape = [None, n_out])
keep_prob = tf.placeholder(tf.float32)

#Conv 1
W_conv1 = weight_variable([10, 10, 1, 24])
b_conv1 = bias_variable([24])

h_conv1 = tf.nn.tanh(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tf.nn.max_pool(W_conv1, ksize=[1, 3, 8 , 1],
                         strides=[1, 3, 8, 1], padding='SAME')
#Conv 2
W_conv2 = weight_variable([5, 5, 24, 48])
b_conv2 = bias_variable([48])

h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Flaten for the FCL 1
h_pool_flat = tf.reshape(h_pool2, [-1, 960], name = 'flatten')


##FCL 1
#W_fc1 = weight_variable([24, 100])
#b_fc1 = bias_variable([100])

#h_fc1 = tf.nn.tanh(tf.matmul(x_image, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##FCL 2
#W_fc2 = weight_variable([100, 50])
#b_fc2 = bias_variable([50])

#h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
W_fc3 = weight_variable([960, 25])
b_fc3 = bias_variable([25])

h_fc3 = tf.nn.tanh(tf.matmul(h_pool_flat, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#FCL 4
W_fc4 = weight_variable([25, n_out])
b_fc4 = bias_variable([n_out])
 
y = tf.tanh(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

