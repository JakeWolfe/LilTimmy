from triangula.input import SixAxisResource,SixAxis
import cv2
import io
import numpy as np
import picamera
import serial
import time
import os

#CK Connect to serial port
ser = serial.Serial('/dev/ttyS0', 9600, timeout=1)

forward_speed = 1680
reverse_speed = 1284
throttle_center = 1476
right_val = 1700
left_val = 1100
steering_center = 1404


total_saved_frames = 0

class QuitButtonPressed(Exception):
	pass
	
def getLatestStatus():
	global raw_data
	x = joystick.axes[2].corrected_value()
	y = joystick.axes[1].corrected_value()
	raw_data = [x,y]
	driveCar()

#CK Send control data to car
def driveCar():
	y_micro = throttle_center + int((forward_speed-throttle_center) * raw_data[1])
	x_micro = steering_center + int(400 * raw_data[0])
	ser.write(str(y_micro)+'t,')
	ser.write(str(x_micro)+'s,')
	ser.flush()
	time.sleep(0.01) #CK Edit if you have issues with control stuttering.
	
def resetCar():  #CK Reset RC car controls to center, so as to avoid it driving off wildly
	ser.write(str(throttle_center)+'t,'+str(steering_center)+'s,')
	ser.flush()

def collect_images():
		
	#CK Initialize frame counters
	saved_frame = 0
	dropped_frame = 0
	total_frame = 0
	global joystick
	global total_saved_frames
	
	image_array = np.zeros((1,2400))
	raw_array = np.zeros((1,2))

	e1 = cv2.getTickCount()
	
	try:
		frame = 1
				
		print 'Capture Initiated'
			
		for foo in camera.capture_continuous(stream, 'jpeg', use_video_port = True, resize=(80,60)):
			
			buttons_pressed = joystick.get_and_clear_button_press_history()
			if buttons_pressed & 1 << SixAxis.BUTTON_CROSS:
				print 'Capture Ended'
				break
			elif buttons_pressed & 1 << SixAxis.BUTTON_CIRCLE:
				raise QuitButtonPressed
			#CK Update frame counters
			frame +=1 
			total_frame += 1
			getLatestStatus()
				
			#CK Save images to stack based on data input
			if (abs(raw_data[1]) > 0.1):
				#CK Construct numpy array from image data stream
				data = np.fromstring(stream.getvalue(), dtype=np.uint8)
				#CK Decode the image from the array
				image = cv2.imdecode(data, 0)
				#CK Select lower half of image
				roi = image[30:60, :]
				#CK Reshape into one dimensional array
				temp_array = roi.reshape(1,2400).astype(np.float32)
				#CK Stack images and control data
				image_array = np.vstack((image_array, temp_array))
				raw_array = np.vstack((raw_array, raw_data))
				saved_frame += 1
				#print raw_data #CK Enable to see the data being recorded
						
			stream.seek(0)		
			stream.truncate()
		
		resetCar()
		print 'RC Car Stopped'
		e2 = cv2.getTickCount()
		time0 = (e2-e1)/cv2.getTickFrequency()
		total_saved_frames += saved_frame
		print 'Capture duration: ', time0
		print 'Total frames: ', total_frame
		print 'Saved frames: ', saved_frame
		print 'Dropped frames: ', total_frame - saved_frame
			
		file_index = 0
		while os.path.exists("training_images/frame%s.jpg" % file_index):
			file_index += 1
		#CK Open data file, format data, and save images and data
		dataFile = open('training_data/data.txt', 'a')		
		for num in range(0,saved_frame):
			temp_image = image_array[num+1].reshape(30,80)
			cv2.imwrite("training_images/frame%s.jpg" % (file_index+num), temp_image)
			dataFile.write('frame%s.jpg' % (file_index + num) + ' ' + str(raw_array[num+1][0]) + ' ' + str(raw_array[num+1][1])+ '\n')
		dataFile.close()
		
	except KeyboardInterrupt:
		resetCar()
		print
		print 'KeyboardInterrupt, Capture Aborted'
		pass
	except QuitButtonPressed:
		resetCar()
		print 'O Button Pressed, Capture Aborted'
		pass
	
	
try:
	with SixAxisResource() as joystick:
				
		#CK Initalize camera
		with picamera.PiCamera() as camera:
			print 'Initializing Camera'
			camera.resolution = (1640,922)
			camera.framerate = 40
			time.sleep(2)
			start = time.time()
			stream = io.BytesIO()
			curr_time = time.time()
				
			print 'Initalizing SixAxis controller'
			getLatestStatus()
			print 'Press square button to start recording, or Triangle to quit'
			while 1:
				getLatestStatus()
				time.sleep(0.09)
				buttons_pressed = joystick.get_and_clear_button_press_history()
				if buttons_pressed & 1 << SixAxis.BUTTON_SQUARE:
					print 'Press X to end capture, or O to abort'
					collect_images()
					print 'Press square button to start capture, or Triangle to exit'
				elif buttons_pressed & 1 << SixAxis.BUTTON_TRIANGLE:
					raise QuitButtonPressed
			
except IOError:
		resetCar()
		print 'Unable to find a SixAxis controller, Press center PS button to sync'
except KeyboardInterrupt:
	resetCar()
	print
	print 'KeyboardInterrupt, Capture Aborted'
	pass
except QuitButtonPressed:
	resetCar()
	print 'Triangle button pressed, Exiting capture session'
	print 'Total number of saved frames: ' + str(total_saved_frames)
	

	
