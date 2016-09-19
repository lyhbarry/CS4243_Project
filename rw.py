""" 
<Read/write module>

Executes all reading and writing of videos.

"""

import cv2
import numpy as np


def read(vid_name):
	""" Read video frames
	
	Args:
		vid_name (string): Name of video file to be read

	Returns:
		frames (ndarray): Ndarray of all frames read from video

	"""
	vid = cv2.VideoCapture(vid_name)
	frames_list = []

	while True:
	    ret, frame = vid.read()		# Read frame by frame
	    frames_list.append(frame)	# Appends frames read from video

	    # Terminates when all frames are read
	    if ret == False:
	    	break

	# Release capture once all frames are read and appended
	vid.release()

	# Converts array to ndarray
	frames_array = np.asarray(frames_list)

	return frames_array


def write():
	"""
	TODO: Write function
	"""
	return None


