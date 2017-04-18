import cv2
import random
import numpy as np

xs = []
y1s = []
y2s = []

num_out = 2
#points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

total_lines = 0

#read data.txt
with open("training_data/data.txt") as f:
	for line in f:
		xs.append("training_images/" + line.split()[0])
		y1s.append(float(line.split()[1]))
		y2s.append(float(line.split()[2]))
		total_lines += 1
		

print "Done Open"

num_images = len(xs)

#Shuffle the data
c = list(zip(xs, y1s, y2s))
random.shuffle(c)
xs, y1s, y2s = zip(*c)

ys = np.column_stack((y1s, y2s))

train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
	global train_batch_pointer
	x_out = []
	y_out = []

	for i in range(0, batch_size):
	
		x_out.append(np.divide(cv2.imread(train_xs[(train_batch_pointer + i) % num_train_images], 0), 255.0))
		y_out.append(train_ys[(train_batch_pointer + i) % num_train_images])
		
	train_batch_pointer += batch_size
	return x_out, y_out

def LoadValBatch(batch_size):
	global val_batch_pointer
	x_out = []
	y_out = []

	for i in range(0, batch_size):
	
		x_out.append(np.divide(cv2.imread(val_xs[(val_batch_pointer + i) % num_val_images], 0), 255.0))
		y_out.append(val_ys[(val_batch_pointer + i) % num_val_images])
		
	val_batch_pointer += batch_size
	return x_out, y_out