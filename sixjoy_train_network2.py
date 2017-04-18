# Comments with J: are by Jake Wolfe. Comments with C: are by Chris Kott
"""
Reference:
OpenCV Python Neural Network Autonomous RC Car
https://github.com/hamuchiwa/AutoRCCar
Adapted from mlp_training.py
"""

import tensorflow as tf
import numpy as np
import sixjoy_model2
import scipy
import cv2

print('Start Training Process')
print('Load Data')

# J: TODO make this inputable to allow for testing
num_out = 2
image_length = 2400

# J: Setup the NN
LOGDIR = './trained_network'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.subtract(sixjoy_model2.y_, sixjoy_model2.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

xs = []
y1s = []
y2s = []

#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

total_lines = 0

#read data.txt
with open("training_data/data.txt") as f:
	for line in f:
		xs.append("training_images/" + line.split()[0])
        #the paper by Nvidia uses the inverse of the turning radius,
        #but steering wheel angle is proportional to the inverse of turning radius
        #so the steering wheel angle in radians is used as the output
		y1s.append(float(line.split()[1]))
		y2s.append(float(line.split()[2]))
		total_lines += 1
		print "Total_lines Open: ", total_lines

print "Done Open"
x_out = []
y1_out = []
y2_out = []

for i in range(0, total_lines):
	x_out.append(cv2.imread(xs[i],0))
	y1_out.append(y1s[i])
	y2_out.append(y2s[i])
	print "Images Read: ", (i+1)

print np.array(x_out).shape
train_step.run(feed_dict={sixjoy_model2.x: x_out, sixjoy_model2.y_: [y1_out,y2_out] , sixjoy_model2.keep_prob:0.8})

print('Saving')

filename = saver.save(sess, )
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
filename = saver.save(sess, checkpoint_path)




# J: TODO add testing computation
# J: Might add timing information

