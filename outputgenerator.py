"""
Author: Rodson Chue

Output Generator
"""

import cv2
import numpy as np
import rw
from stats_generator import get_distance


def output_writer(vid_name):
    """ Generates output an output writer """
    default_vid = cv2.VideoCapture(vid_name)
    vid = cv2.VideoCapture(vid_name)

    # Get properties of video
    w = int(default_vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    h = int(default_vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.cv.CV_CAP_PROP_FPS))
    frame_count = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')

    out = cv2.VideoWriter('output.avi', fourcc, fps, (w * 2, h * 2))
    return out

def draw_stats_alt(frame, dist):
	""" Draw statistics frames """
	font = cv2.FONT_HERSHEY_PLAIN	# Font-type

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

def draw_top_down_alt(frame, feature_pos):
    """ Draw top-down view frames """
    font = cv2.FONT_HERSHEY_PLAIN	# Font-type
    court_color = (133, 35, 160)
    net_color = (255, 255, 255)
    net_pole_color = (9, 237, 233)
    a_side_color = (229,85,59)
    b_side_color = (120,229,110)
    ball_color = (73,156,244)
    dim_w, dim_h, _ = frame.shape

    # Section label
    cv2.putText(frame, 'Top Down View', (5, 15), font, 1, (255,255,255), 1, cv2.CV_AA)

    # Volleyball court
    cv2.rectangle(frame, (76, 30), (556, 270), court_color, 2)		# Court outline
    cv2.line(frame, (316, 20), (316, 280), net_color, 2)			# Net
    cv2.rectangle(frame, (306, 10), (326, 20), net_pole_color, -1)		# Net-pole 1
    cv2.rectangle(frame, (306, 290), (326, 280), net_pole_color, -1)		# Net-pole 2

    # Based on the definition of the volleyball court above
    court_width_p = float(556-76)
    court_height_p = float(270-30)

    # Player A1
    if('a1' in feature_pos.keys()):
        w, h = feature_pos['a1']
        pos = (int((w*court_width_p)+76), int((h*court_height_p)+30))
        cv2.circle(frame, pos, 10, a_side_color, -1)
        cv2.putText(frame, 'A1', pos, font, 1.5, (255,255,255), 1, cv2.CV_AA)

    # Player A2
    if('a2' in feature_pos.keys()):
        w, h = feature_pos['a2']
        pos = (int((w*court_width_p)+76), int((h*court_height_p)+30))
        cv2.circle(frame, pos, 10, a_side_color, -1)
        cv2.putText(frame, 'A2', pos, font, 1.5, (255,255,255), 1, cv2.CV_AA)

    # Player B1
    if('b1' in feature_pos.keys()):
        w, h = feature_pos['b1']
        pos = (int((w*court_width_p)+76), int((h*court_height_p)+30))
        cv2.circle(frame, pos, 10, b_side_color, -1)
        cv2.putText(frame, 'B1', pos, font, 1.5, (255,255,255), 1, cv2.CV_AA)

    # Player B2
    if('b2' in feature_pos.keys()):
        w, h = feature_pos['b2']
        pos = (int((w*court_width_p)+76), int((h*court_height_p)+30))
        cv2.circle(frame, pos, 10, b_side_color, -1)
        cv2.putText(frame, 'B2', pos, font, 1.5, (255,255,255), 1, cv2.CV_AA)

    # Volleyball
    if('ball' in feature_pos.keys()):
        w, h = feature_pos['ball']
        pos = (int((w*court_width_p)+76), int((h*court_height_p)+30))
        cv2.circle(frame, pos, 5, ball_color, -1)
        cv2.putText(frame, 'Ball', pos, font, 1.5, (255,255,255), 1, cv2.CV_AA)



def generate_frame(orig_frame, full_court_frame, outputDim, feature_pos, feature_pos_prev, feature_dist):
    """ Generates an output frame based on input frame and existing information """
    """
    feature_pos is a dictionary where K: name of feature, V: coord position
    feature_pos_prev is a dictionary where K: name of feature, V: previous coord position
    feature_dist is a dictionary where K: name of feature, V: distance so far for that feature
    """
    w, h, _ = outputDim

    # Each component only takes up 1/4 of the entire frame (1/2 in each dimension)
    vid_frame = cv2.resize(orig_frame, (h, w))
    full_court_frame = cv2.resize(full_court_frame, (h, w))
    top_down_frame = np.zeros(outputDim, np.uint8)
    stats_frame = np.zeros(outputDim, np.uint8)

    # Compute statistics
    for k, v in feature_dist.items():
        feature_dist[k] += get_distance(np.asarray(feature_pos[k]), np.asarray(feature_pos_prev[k]))

    # Text labels for each frame
    rw.draw_original_vid(vid_frame)
    rw.draw_full_court(full_court_frame)
    draw_top_down_alt(top_down_frame, feature_pos)
    rw.draw_stats(stats_frame, feature_dist)


    # Concatenate output videos
    top_half = np.hstack((vid_frame, full_court_frame))
    btm_half = np.hstack((top_down_frame, stats_frame))
    full_vid_frame = np.vstack((top_half, btm_half))
    return full_vid_frame
