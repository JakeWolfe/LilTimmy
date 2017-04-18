import tensorflow as tf
import scipy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

n_input = 2400 # 30 * 80
n_out = 2 # Steer and Speed
dropout = 0.75 

x = tf.placeholder(tf.float32, shape = [None,30,80])
x_image = tf.reshape(x, shape = [-1,n_input])


y_ = tf.placeholder(tf.float32, shape = [None, n_out])
keep_prob = tf.placeholder(tf.float32)

#FCL 1
W_fc1 = weight_variable([n_input, 100])
b_fc1 = bias_variable([100])

h_fc1 = tf.nn.tanh(tf.matmul(x_image, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#FCL 2
W_fc2 = weight_variable([100, 50])
b_fc2 = bias_variable([50])

h_fc2 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

#FCL 3
W_fc3 = weight_variable([50, 25])
b_fc3 = bias_variable([25])

h_fc3 = tf.nn.tanh(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
#FCL 4
W_fc4 = weight_variable([25, n_out])
b_fc4 = bias_variable([n_out])
 
y = tf.tanh(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

