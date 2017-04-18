from triangula.input import SixAxisResource,SixAxis
import cv2
import io
import numpy as np
import picamera
import serial
import time
import os

# Connect to serial port
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)
	
def collect_image():
		
	# Initialize frame counters
	saved_frame = 0
	dropped_frame = 0
	total_frame = 0
	x_pressed = False
	global joystick

	e1 = cv2.getTickCount()
	image_array = np.zeros((1, 2400))
	raw_array = np.zeros((1,2))
	
	try:
		frame = 1
		
		# Initalize camera
		with picamera.PiCamera() as camera:
			print 'Start Camera'
			camera.resolution = (80, 60)
			camera.framerate = 60
			time.sleep(2)
			start = time.time()
			stream = io.BytesIO()
			curr_time = time.time()	
			
			print 'Start collecting images...'
			
			with SixAxisResource() as joystick:
				for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True):
					
					buttons_pressed = joystick.get_and_clear_button_press_history()
					if buttons_pressed & 1 << SixAxis.BUTTON_CROSS:
						x_pressed = True
					
					# Construct numpy array from image data stream
					data = np.fromstring(stream.getvalue(), dtype=np.uint8)
					# Decode the image from the array
					image = cv2.imdecode(data, 0)
					# Select lower half of image
					roi = image[30:60, :]
					# Save streamed images
					cv2.imwrite('training_images/frame{:>05}.jpg'.format(1),image)
					# Reshape data into one dimensional array
					temp_array = roi.reshape(1,2400).astype(np.float32)
					# Update frame counters
					frame +=1 
					total_frame += 1
					getLatestStatus()
					print raw_data
					# Save images to stack based on data input
					if x_pressed:
						print 'Quit...'
						break
					elif (abs(raw_data[0]) > 0.1):
						image_array = np.vstack((image_array, temp_array))
						raw_array = np.vstack((raw_array, raw_data))
						saved_frame += 1
					
					stream.seek(0)		
					stream.truncate()
				
		# Save training images
		train = image_array[1:,:]		
		train_raw = raw_array[1:,]
		# Scan path for existing files and create new file name
		i = 0
		while os.path.exists("training_data/sixjoy_test%s.npz" % i):
			i += 1
		# Save training images as numpy file
		np.savez('training_data/sixjoy_test%s.npz' % i, train=train, train_raw=train_raw)
		
		e2 = cv2.getTickCount()
		time0 = (e2-e1)/cv2.getTickFrequency()
		print 'Streaming duration: ', time0
		print train.shape
		print train_raw.shape
		print 'Total frames: ', total_frame
		print 'Saved frames: ', saved_frame
		print 'Dropped frames: ', total_frame - saved_frame
	except KeyboardInterrupt:
		pass
		
# Grab only the most recent serial data
def getLatestStatus():
	global raw_data
	x = joystick.axes[2].corrected_value()
	y = joystick.axes[3].corrected_value()
	y_micro = 1476 + int((1750-1476) * y)
	x_micro = 1400 + int(500 * x)
	ser.write(str(y_micro)+'t,')
	ser.write(str(x_micro)+'s,')
	ser.flush()
	time.sleep(0.05)
	raw_data = [x,y]

collect_image()