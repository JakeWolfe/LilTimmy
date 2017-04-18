# Comments with J: are by Jake Wolfe. Comments with C: are by Chris Kott
"""
Reference:
PiCamera Advanced Recipes
http://picamera.readthedocs.io/en/release-1.11/recipes2.html
OpenCV Python Neural Network Autonomous RC Car
https://github.com/hamuchiwa/AutoRCCar
Adapted from rc_driver.py
"""
import threading
import serial
import cv2
import numpy as np
import math
import io
import picamera
import time
import tensorflow as tf
import sixjoy_model2

# J: Controler for RC car
num_out = 2
hidden_size = 40
image_length = 2400

forward_speed = 1700
reverse_speed = 1284
throttle_center = 1476
right_val = 1700
left_val = 1100
steering_center = 1410

class RCControl(object):

	def __init__(self):
		self.ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
		self.ser.write(str(throttle_center)+'t,')
		self.ser.write(str(steering_center)+'s,')
		self.ser.flush()
		print 'Control init'
		
	def steer(self, raw_data):
	
		if raw_data.size == 1:	
			if ((abs(raw_data) <= 1)):
				#y_micro = throttle_center + int((forward_speed-throttle_center) * raw_data)
				x_micro = steering_center + int(400 * raw_data)
				self.ser.write(str(forward_speed)+'t,')
				self.ser.write(str(x_micro)+'s,')
				self.ser.flush()
				time.sleep(0.01)
			else:
				self.stop()
				
		if raw_data[0].size == 2:
			if (((abs(raw_data[0][0])) <= 1) & (abs(raw_data[0][1]) <= 1)):
				y_micro = throttle_center + int((forward_speed-throttle_center) * raw_data[0][1])
				x_micro = steering_center + int(400 * raw_data[0][0])
				self.ser.write(str(y_micro)+'t,')
				self.ser.write(str(x_micro)+'s,')
				self.ser.flush()
				time.sleep(0.01)
			else:
				self.stop()

	def stop(self):
		self.ser.write(str(throttle_center)+'t,')
		self.ser.write(str(steering_center)+'s,')
		self.ser.flush()

# J: Controler for the NN
class NeuralNetwork(object):
	
	def __init__(self):

		print 'Load trained Neural Network'
		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, "./trained_network/model.chkpt/model.ckpt")
	
	def predict(self, image):
	
		resp = sixjoy_model2.y.eval(feed_dict={sixjoy_model2.x: [image], sixjoy_model2.keep_prob: 1})
		return resp

class Driver(object):

	def __init__(self):

		self.model = NeuralNetwork()
		self.rc_car = RCControl()
		self.handle()
		

	def handle(self):
		
		try:
			# J: Refer to training_capture.py for how this is running
			with picamera.PiCamera() as camera:
				print 'Start Camera'
				camera.resolution = (80, 60)
				camera.framerate = 60
				time.sleep(1)
				start = time.time()
				stream = io.BytesIO()
				curr_time = time.time()
				frames = 0
				
				print 'Start Predicting'

				# J: Predict for each image
				for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True):

					# J: Frame counter
					frames += 1

					# J: Turn picture into a Matrix
					data = np.fromstring(stream.getvalue(), dtype=np.uint8)
					image = cv2.imdecode(data, 0) # J: The 0 defines grayscale
					
					# J: Use bottom half (May change value depending on horizon)
					half_image = image[30:60, :]
					
					# J: Predict
					prediction = self.model.predict(half_image)
					print prediction
					self.rc_car.steer(prediction)

					print 'New Image at time', curr_time - start, 'at frame', frames, 'going', prediction
					
					curr_time = time.time()
					if curr_time - start > 10:
						print 'Done Running'
						break
					stream.truncate()
					stream.seek(0)

				# J: Once I find a resonable way to break, clean up
				self.rc_car.stop()

		except KeyboardInterrupt:
			self.rc_car.stop()
			print
			print 'Stopped'
			pass
				
if __name__ == '__main__':
	Driver()
