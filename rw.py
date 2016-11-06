""" 
<Read/write module>

Executes all reading and writing of videos.

"""


import cv2
import numpy as np


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
