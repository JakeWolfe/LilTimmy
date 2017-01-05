# Comments with J: are by Jake Wolfe. Comments with C: are by Chris Kott
"""
Reference:
OpenCV Python Neural Network Autonomous RC Car
https://github.com/hamuchiwa/AutoRCCar
Adapted from mlp_training.py
"""

import cv2
import numpy as np
import glob

print 'Start Training Process'
print 'Load Data'

# J: TODO make this inputable to allow for testing
hidden_size = 40

# J: Load data, get the length of each array
file_names = glob.glob('training/*.npz')
values_temp = np.load(file_names[0])
image_length = values_temp[]

del values_temp

# J: Start the matrix to start storing values
training_images = np.zeros((1,image_length))
training_labels = np.zeros((1, 4), 'float')

for unloaded_file in file_names:
	with np.load(unloaded_file) as loaded_file:
	
		# J: Assume the file has two sets of data, image and label
		image_temp = data['image']
		label_temp = data['label']
	
	# J: Assume all files are of the same length
	training_images = np.vstack((training_images, image_temp))
	training_labels = np.vstack((training_labels, label_temp))

# J: Filter 0's from first row. Byproduct from using np.zeros
images = training_images[1:, :]
train_lables = training_labels[1:, :]

# J: Make the NN
layer_sizes = np.int32([image_length, hidden_size, 4])

# J: Code 'borrowed' from reference
# J: TODO understand all parts of borrowed code
model = cv2.ANN_MLP()
model.create(layer_sizes)
criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 0.0001)
criteria2 = (cv2.TERM_CRITERIA_COUNT, 100, 0.001)
params = dict(term_crit = criteria,
               train_method = cv2.ANN_MLP_TRAIN_PARAMS_BACKPROP,
               bp_dw_scale = 0.001,
               bp_moment_scale = 0.0 )

print 'Start Training'

iterations_response = model.train(images, train_lables, None, params = params)
print 'Took ', iterations_response, ' iterations'

# J: Save to file
model.save('trained_NN/NN.xml')

# J: TODO add testing computation
# J: Might add timing information


