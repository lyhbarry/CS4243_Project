""" 
<Read/write module>

Executes all reading and writing of videos.

"""
import cv2
import numpy as np

from random import randint


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
    vid = cv2.VideoWriter(filename + ".avi", cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'),  fps, (w, h))


def output_generator(vid_name):
	vid = cv2.VideoCapture(vid_name)

	w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(vid.get(cv2.CAP_PROP_FPS))

	font = cv2.FONT_HERSHEY_PLAIN

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter('output.avi', fourcc, fps, (w * 2, h * 2))

	while True:	
		# Test top-down static output
		test_pos = {'a1_u': randint(0, 632), 'a1_v': randint(0, 300),
					'a2_u': randint(0, 632), 'a2_v': randint(0, 300),
					'b1_u': randint(0, 632), 'b1_v': randint(0, 300),
					'b2_u': randint(0, 632), 'b2_v': randint(0, 300),
					'ball_u': randint(0, 632), 'ball_v': randint(0, 300)
					}		
		ret1, vid_frame = vid.read()		# Read frame by frame

		# Terminates when all frames are read
		if ret1 == False:
			break
		
		# Initialise output frames		
		fullcourt_frame = np.zeros(vid_frame.shape, np.uint8)
		topdown_frame = np.zeros(vid_frame.shape, np.uint8)
		stats_frame = np.zeros(vid_frame.shape, np.uint8)
		
		# Draw original video
		cv2.putText(vid_frame, 'Original Video', (5, 15), font, 1, (255,255,255), 1, cv2.LINE_AA)  

		# Draw full-court results
		# TODO: Draw feature points
		cv2.putText(fullcourt_frame, 'Full Court Video', (5, 15), font, 1, (255,255,255), 1, cv2.LINE_AA)	
		
		# Draw top-down projection results
		# TODO: Draw player position
		cv2.putText(topdown_frame, 'Top Down View', (5, 15), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.rectangle(topdown_frame, (50, 30), (582, 270), (133, 35, 160), 2)
		cv2.line(topdown_frame, (316, 20), (316, 280), (255, 255, 255), 2)
		cv2.rectangle(topdown_frame, (306, 10), (326, 20), (9, 237, 233), -1)
		cv2.rectangle(topdown_frame, (306, 290), (326, 280), (9, 237, 233), -1)
		cv2.circle(topdown_frame, (test_pos['a1_u'], test_pos['a1_v']), 10, (229,85,59), -1)
		cv2.putText(topdown_frame, 'A1', (test_pos['a1_u'], test_pos['a1_v']), font, 1.5, (255,255,255), 1, cv2.LINE_AA)    
		cv2.circle(topdown_frame, (test_pos['a2_u'], test_pos['a2_v']), 10, (229,85,59), -1)
		cv2.putText(topdown_frame, 'A2', (test_pos['a2_u'], test_pos['a2_v']), font, 1.5, (255,255,255), 1, cv2.LINE_AA)    
		cv2.circle(topdown_frame, (test_pos['b1_u'], test_pos['b1_v']), 10, (120,229,110), -1)
		cv2.putText(topdown_frame, 'B1', (test_pos['b1_u'], test_pos['b1_v']), font, 1.5, (255,255,255), 1, cv2.LINE_AA)    
		cv2.circle(topdown_frame, (test_pos['b2_u'], test_pos['b2_v']), 10, (120,229,110), -1)
		cv2.putText(topdown_frame, 'B2', (test_pos['b2_u'], test_pos['b2_v']), font, 1.5, (255,255,255), 1, cv2.LINE_AA)
		cv2.circle(topdown_frame, (test_pos['ball_u'], test_pos['ball_v']), 10, (73,156,244), -1)
		cv2.putText(topdown_frame, 'Ball', (test_pos['ball_u'], test_pos['ball_v']), font, 1.5, (255,255,255), 1, cv2.LINE_AA)  

		# Draw statistics for video
		cv2.putText(stats_frame, 'Statistics', (5, 15), font, 1, (255,255,255), 1, cv2.LINE_AA)
		cv2.rectangle(stats_frame, (5, 30), (627, 270), (255, 255, 255), 1)
		cv2.line(stats_frame, (316, 30), (316, 270), (255, 255, 255), 1)
		cv2.line(stats_frame, (5, 75), (627, 75), (255, 255, 255), 1)
		cv2.line(stats_frame, (5, 165), (627, 165), (255, 255, 255), 1)	
		cv2.putText(stats_frame, 'Team A', (25, 65), font, 1.5, (229,85,59), 2, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Player 1', (25, 105), font, 1.25, (255,255,255), 1, cv2.LINE_AA)
		cv2.putText(stats_frame, 'Distance run: ' + '0' + 'm', (25, 125), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Jump count: ' + '0', (25, 145), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Player 2', (25, 195), font, 1.25, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Distance run: ' + '0' + 'm', (25, 215), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Jump count: ' + '0', (25, 235), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Team B', (341, 65), font, 1.5, (120,229,110), 2, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Player 1', (341, 105), font, 1.25, (255,255,255), 1, cv2.LINE_AA)
		cv2.putText(stats_frame, 'Distance run: ' + '0' + 'm', (341, 125), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Jump count: ' + '0', (341, 145), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Player 2', (341, 195), font, 1.25, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Distance run: ' + '0' + 'm', (341, 215), font, 1, (255,255,255), 1, cv2.LINE_AA)    
		cv2.putText(stats_frame, 'Jump count: ' + '0', (341, 235), font, 1, (255,255,255), 1, cv2.LINE_AA)    

		# Concatenate output videos
		top_half = np.hstack((vid_frame, fullcourt_frame))
		btm_half = np.hstack((topdown_frame, stats_frame))
		full_vid = np.vstack((top_half, btm_half))
		out.write(full_vid)
			
		cv2.imshow('videos', full_vid)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release capture once all frames are read and appended
	vid.release()
	out.release()
	cv2.destroyAllWindows()	