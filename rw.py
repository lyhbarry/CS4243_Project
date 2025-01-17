"""
<Read/write module>

Executes all reading and writing of videos.
"""
import cv2
import re
import numpy as np

from random import randint
from stats_generator import get_distance

font = cv2.FONT_HERSHEY_PLAIN	# Font-type

def read(vid_name):
	"""
	Purpose:
		Read video frames
	Args:
		vid_name (string): Name of video file to be read
	Returns:
		frames (ndarray): Ndarray of all frames read from video
	"""
	vid = cv2.VideoCapture(vid_name)
	frames_list = []

	ret = True
	while ret:
		ret, frame = vid.read()  # Read frame by frame
		if frame is not None:
			frames_list.append(frame)  # Appends frames read from video
			#
			# # Terminates when all frames are read
			# if ret is False:
			#     break

	fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)

	# Release capture once all frames are read and appended
	vid.release()

	# Converts array to ndarray
	frames = np.asarray(frames_list)

	return frames, fps


def write(filename, frames, fps):
	# type: (string, ndarray, int) -> void
	(h, w) = frames[0].shape[:2]
	cv2.VideoWriter(filename + ".avi", cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'),\
				    fps, (w, h))


def concatenate_output():
	vid = cv2.VideoCapture("output_1.avi")

	# Get properties of video
	w = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	h = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = int(vid.get(cv2.cv.CV_CAP_PROP_FPS))
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

	fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	out = cv2.VideoWriter("final_output.avi", fourcc, fps, (w, h))

	for i in range(frame_count):
		ret, frame = vid.read()
		out.write(frame)

	vid = cv2.VideoCapture("output_2.avi")
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	for i in range(frame_count):
		ret, frame = vid.read()
		out.write(frame)

	vid = cv2.VideoCapture("output_3.avi")
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	for i in range(frame_count):
		ret, frame = vid.read()
		out.write(frame)

	vid = cv2.VideoCapture("output_4.avi")
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	for i in range(frame_count):
		ret, frame = vid.read()
		out.write(frame)

	vid = cv2.VideoCapture("output_5.avi")
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	for i in range(frame_count):
		ret, frame = vid.read()
		out.write(frame)

	vid = cv2.VideoCapture("output_6.avi")
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	for i in range(frame_count):
		ret, frame = vid.read()
		out.write(frame)

	vid = cv2.VideoCapture("output_7.avi")
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	for i in range(frame_count):
		ret, frame = vid.read()
		out.write(frame)

	out.release()



def draw_original_vid(frame):
	""" Draw original video frames """

	# Section label
	cv2.putText(frame, 'Original Video', (5, 15), font, 1, (255,255,255), 1, cv2.CV_AA)


def draw_full_court(frame):
	""" Draw full court frames """

	# Section Label
	cv2.putText(frame, 'Full Court Video', (5, 15), font, 1, (255,255,255), 1, cv2.CV_AA)


def draw_top_down(frame, feature_pos):
	""" Draw top-down view frames """

	# Section label
	cv2.putText(frame, 'Top Down View', (5, 15), font, 1, (255,255,255), 1, cv2.CV_AA)

	# Volleyball court
	cv2.rectangle(frame, (76, 30), (556, 270), (133, 35, 160), 2)		# Court outline
	cv2.line(frame, (316, 20), (316, 280), (255, 255, 255), 2)			# Net
	cv2.rectangle(frame, (306, 10), (326, 20), (9, 237, 233), -1)		# Net-pole 1
	cv2.rectangle(frame, (306, 290), (326, 280), (9, 237, 233), -1)		# Net-pole 2

	# Player A1
	cv2.circle(frame, (feature_pos['a1_u'], feature_pos['a1_v']), 10, (229,85,59), -1)
	cv2.putText(frame, 'A1', (feature_pos['a1_u'], feature_pos['a1_v']), font, 1.5, (255,255,255), 1, cv2.CV_AA)

	# Player A2
	cv2.circle(frame, (feature_pos['a2_u'], feature_pos['a2_v']), 10, (229,85,59), -1)
	cv2.putText(frame, 'A2', (feature_pos['a2_u'], feature_pos['a2_v']), font, 1.5, (255,255,255), 1, cv2.CV_AA)

	# Player B1
	cv2.circle(frame, (feature_pos['b1_u'], feature_pos['b1_v']), 10, (120,229,110), -1)
	cv2.putText(frame, 'B1', (feature_pos['b1_u'], feature_pos['b1_v']), font, 1.5, (255,255,255), 1, cv2.CV_AA)

	# Player B2
	cv2.circle(frame, (feature_pos['b2_u'], feature_pos['b2_v']), 10, (120,229,110), -1)
	cv2.putText(frame, 'B2', (feature_pos['b2_u'], feature_pos['b2_v']), font, 1.5, (255,255,255), 1, cv2.CV_AA)

	# Volleyball
	cv2.circle(frame, (feature_pos['ball_u'], feature_pos['ball_v']), 5, (73,156,244), -1)
	cv2.putText(frame, 'Ball', (feature_pos['ball_u'], feature_pos['ball_v']), font, 1.5, (255,255,255), 1, cv2.CV_AA)


def draw_stats(frame, dist):
	""" Draw statistics frames """

	# Section label
	cv2.putText(frame, 'Statistics', (5, 15), font, 1, (255,255,255), 1, cv2.CV_AA)

	# Statistics table
	cv2.rectangle(frame, (5, 30), (627, 270), (255, 255, 255), 1)
	cv2.line(frame, (316, 30), (316, 270), (255, 255, 255), 1)
	cv2.line(frame, (5, 75), (627, 75), (255, 255, 255), 1)
	cv2.line(frame, (5, 165), (627, 165), (255, 255, 255), 1)

	# Team A statistics
	cv2.putText(frame, 'Team A', (25, 65), font, 1.5, (229,85,59), 2, cv2.CV_AA)

	# Player A1
	cv2.putText(frame, 'Player A1', (25, 105), font, 1.25, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Distance run: ' + str("{0:.2f}".format(dist['a1'])) + 'm', (25, 125), font, 1, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Jump count: ' + '0', (25, 145), font, 1, (255,255,255), 1, cv2.CV_AA)

	# Player A2
	cv2.putText(frame, 'Player A2', (25, 195), font, 1.25, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Distance run: ' + str("{0:.2f}".format(dist['a2'])) + 'm', (25, 215), font, 1, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Jump count: ' + '0', (25, 235), font, 1, (255,255,255), 1, cv2.CV_AA)

	# Team B statistics
	cv2.putText(frame, 'Team B', (341, 65), font, 1.5, (120,229,110), 2, cv2.CV_AA)

	# Player B1
	cv2.putText(frame, 'Player B1', (341, 105), font, 1.25, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Distance run: ' + str("{0:.2f}".format(dist['b1'])) + 'm', (341, 125), font, 1, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Jump count: ' + '0', (341, 145), font, 1, (255,255,255), 1, cv2.CV_AA)

	# Player B2
	cv2.putText(frame, 'Player B2', (341, 195), font, 1.25, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Distance run: ' + str("{0:.2f}".format(dist['b2'])) + 'm', (341, 215), font, 1, (255,255,255), 1, cv2.CV_AA)
	cv2.putText(frame, 'Jump count: ' + '0', (341, 235), font, 1, (255,255,255), 1, cv2.CV_AA)


def output_generator(vid_name):
	""" Generates output video """
	default_vid = cv2.VideoCapture('input/beachVolleyball1.mov')
	vid = cv2.VideoCapture(vid_name)

	vid_num = int(re.search(r'\d+', vid_name).group())

	# Get properties of video
	w = int(default_vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	h = int(default_vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	fps = int(vid.get(cv2.cv.CV_CAP_PROP_FPS))
	frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

	if fps > 60:
		fps = 60
	# Define the codec and create VideoWriter object
	fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')
	out = cv2.VideoWriter('output_' + str(vid_num) + '.avi', fourcc, fps, (w * 2, h * 2))

	dist = {}
	dist['a1'] = 0
	dist['a2'] = 0
	dist['b1'] = 0
	dist['b2'] = 0

	print '\n[BEGIN] - Generating Output'
	for i in range(frame_count):
		ret, vid_frame = vid.read()

		if (i + 1) % 50 == 0:
			print 'Processing ', i + 1, '/', frame_count, ' frames...'

		vid_frame = cv2.resize(vid_frame, (w, h))

		# Test top-down output
		feature_pos = {'a1_u': randint(0, 316), 'a1_v': randint(0, 300),
					   'a2_u': randint(0, 316), 'a2_v': randint(0, 300),
					   'b1_u': randint(316, 632), 'b1_v': randint(0, 300),
					   'b2_u': randint(316, 632), 'b2_v': randint(0, 300),
					   'ball_u': randint(0, 632), 'ball_v': randint(0, 300)
					  }

		# Initialise output frames
		full_court_frame = cv2.imread('6_test/warped_' + '{:04}'.format(i) + \
									  '.jpg')
		full_court_frame = cv2.resize(full_court_frame, (w, h))
		top_down_frame = np.zeros(vid_frame.shape, np.uint8)
		stats_frame = np.zeros(vid_frame.shape, np.uint8)

		# Draw output section (original, fullcourt, topdown, statistics)
		draw_original_vid(vid_frame)
		draw_full_court(full_court_frame)
		draw_top_down(top_down_frame, feature_pos)

		a1_start_pos = np.asarray([feature_pos['a1_u'], feature_pos['a1_v']])
		a2_start_pos = np.asarray([feature_pos['a2_u'], feature_pos['a2_v']])
		b1_start_pos = np.asarray([feature_pos['b1_u'], feature_pos['b1_v']])
		b2_start_pos = np.asarray([feature_pos['b2_u'], feature_pos['b2_v']])

		if i == 0:
			a1_prev_pos = a1_start_pos
			a2_prev_pos = a2_start_pos
			b1_prev_pos = b1_start_pos
			b2_prev_pos = b2_start_pos
		# Updates distance every 10 frames
		if i!= 0 and i % 10 == 0:
			a1_end_pos = np.asarray([feature_pos['a1_u'],\
									 feature_pos['a1_v']])
			a2_end_pos = np.asarray([feature_pos['a2_u'],\
									 feature_pos['a2_v']])
			b1_end_pos = np.asarray([feature_pos['b1_u'],\
									 feature_pos['b1_v']])
			b2_end_pos = np.asarray([feature_pos['b2_u'],\
									 feature_pos['b2_v']])
			dist['a1'] += get_distance(vid_num, a1_prev_pos, a1_end_pos)
			dist['a2'] += get_distance(vid_num, a2_prev_pos, a2_end_pos)
			dist['b1'] += get_distance(vid_num, b1_prev_pos, b1_end_pos)
			dist['b2'] += get_distance(vid_num, b2_prev_pos, b2_end_pos)
			a1_prev_pos = np.asarray([feature_pos['a1_u'],\
									  feature_pos['a1_v']])
			a2_prev_pos = np.asarray([feature_pos['a2_u'],\
									  feature_pos['a2_v']])
			b1_prev_pos = np.asarray([feature_pos['b1_u'],\
									  feature_pos['b1_v']])
			b2_prev_pos = np.asarray([feature_pos['b2_u'],\
									  feature_pos['b2_v']])
		draw_stats(stats_frame, dist)

		# Concatenate output videos
		top_half = np.hstack((vid_frame, full_court_frame))
		btm_half = np.hstack((top_down_frame, stats_frame))
		full_vid = np.vstack((top_half, btm_half))

		# Write frame to video
		out.write(full_vid)

		cv2.imshow('Output Video', full_vid)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	print 'Processing ', frame_count, '/', frame_count, ' frames...'
	print '[COMPLETED] - Generating Output'

	# Release capture once all frames are read and appended
	vid.release()
	out.release()
	cv2.destroyAllWindows()