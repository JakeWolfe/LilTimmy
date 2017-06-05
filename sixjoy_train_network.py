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
import os
import data_handler

print('Start Training Process')

# J: TODO make this inputable to allow for testing
num_out = 2
image_length = 2400

# J: Setup the NN
LOGDIR = './trained_network/model.chkpt'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

#loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=sixjoy_model2.y_, logits=sixjoy_model2.y))
loss = tf.reduce_mean(
	tf.square(tf.subtract(sixjoy_model2.y_, sixjoy_model2.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

epochs = 50
batch_size = 100
print ("Training Images Count: %d" % (data_handler.num_train_images))

print("Running %d Epochs with a Batch Size of %d" % (epochs, batch_size))
for epoch in range(epochs):
	for i in range(int(data_handler.num_train_images/batch_size)):
		xs, ys = data_handler.LoadTrainBatch(batch_size)
		train_step.run(feed_dict={sixjoy_model2.x: xs, sixjoy_model2.y_: ys , sixjoy_model2.keep_prob:0.8})
		if (i % 10 == 0):
			xs, ys = data_handler.LoadValBatch(batch_size)
			loss_value = loss.eval(feed_dict={sixjoy_model2.x: xs, sixjoy_model2.y_: ys , sixjoy_model2.keep_prob:1.0})
			print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))
			
		if (i % batch_size == 0):
			if not os.path.exists(LOGDIR):
				os.makedirs(LOGDIR)
			checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
			filename = saver.save(sess, checkpoint_path)
			print("Model saved in file: %s" % filename)	


