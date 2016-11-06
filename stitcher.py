"""
Author: Ng Jun Wei

Stitcher module

Handles the full court panoramic view of the game.
"""

import numpy as np
import homography as hg
import cv2


def generate_corresponding_pairs(points1, points2):
    # type: ((int, int)[], (int, int)[]) -> ((int, int), (int, int))[]
    """
    Purpose:
        Generate the list of pairs of corresponding points between two consecutive list of points.
    Parameters:
        points1 - the list of feature points corresponding to frame #1
        points2 - the list of feature points corresponding to frame #2
    Returns:
        a list of pairs of corresponding feature points for frame #1 and #2
    """
    # TODO: perform correspondence check.

    # TODO: remove stub value.
    pairs = []
    return pairs


def generate_homography_matrix_list(points):
    # type: ((int, int)[]) -> np.matrix[]
    """
    Purpose:
        Generate the list of corresponding homography matrix to map one frame to the initial point of view.
    Parameters:
        points - the list of list of feature points
    Returns:
        a list of homography matrices
    """
    matrices = []

    # adding an identity matrix as the first homography matrix
    # this is because the first frame will be used as the anchor frame
    matrices.append(np.eye(3))

    prev = points[0]

    # for each two consecutive sets of points, generate the homography matrix
    for i in range(1, len(points)):
        curr = points[i]

        pairs = generate_corresponding_pairs(prev, curr)
        points_curr = []
        points_prev = []
        for j in range(len(pairs)):
            points_curr.append(pairs[j][0])
            points_prev.append(pairs[j][1])
        homo_mat = hg.find_homography(points_curr, points_prev)

        matrices.append(homo_mat)

        prev = curr
    return matrices


def get_output_size(h, w, homo_mat):
    # type: (int, int, np.matrix) -> int, int, int, int
    """
    Purpose:
        Calculates the new size of the output frame.
    Parameters:
        h - original frame height
        w - original frame width
        homo_mat - corresponding homography matrix
    Returns:
        left - left-most position of the output
        top - top-most position of the output
        right - right-most position of the output
        bottom - bottom-most position of the output
    """

    # TODO: explain how size matrix works.
    size_mat = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])

    size_mat = np.dot(homo_mat, size_mat)

    left = np.amin(size_mat[0] / size_mat[2])
    top = np.amin(size_mat[1] / size_mat[2])
    right = np.amax(size_mat[0] / size_mat[2])
    bottom = np.amax(size_mat[1] / size_mat[2])

    return left, top, right, bottom


def create_full_court_stitch(frames, points, path):
    # type: (ndarray[], ((int, int)[])[]) -> void
    """
    Purpose:
        Create the list of frames corresponding to the full court view of the game.
    Parameters:
        frames - list of video frames
        points - list of feature points in the video
        path - file path to store output JPG files to
    Returns:
        void
    Usage example:
        import stitcher as st

        frames = <get frames of video using rw>
        points = <get feature points using feature tracker(detector?)>
        st.create_full_court_stitch(frames, points, '6_working/pano')
    """

    # step 1: generate list of homography matrices between each frames
    matrices = generate_homography_matrix_list(points)

    temp = np.eye(3)

    homos = []
    positions = []
    for i in range(len(matrices)):
        temp = np.dot(temp, matrices[i])
        homos.append(temp)

        (h, w) = frames[i].shape[:2]
        l, t, r, b = get_output_size(h, w, temp)
        positions.append((l, t, r, b))
    positions = np.array(positions, np.int32)

    offset_left = np.amin(positions[:, 0])
    offset_top = np.amin(positions[:, 1])
    offset_right = np.amin(positions[:, 2])
    offset_bottom = np.amin(positions[:, 3])

    max_w = int(round(abs(offset_left) + offset_right))
    max_h = int(round(abs(offset_top) + offset_bottom))

    for i in range(len(frames)):
        # step 2: calculate the homography matrix to use
        homo = np.array(homos[i], copy=True)
        # setting left offset for current frame after homography
        homo[0, 2] += abs(offset_left)
        # setting top offset for current frame after homography
        homo[1, 2] += abs(offset_top)

        # step 3: generate the output frame
        output = cv2.warpPerspective(frames[i], homo, max_w, max_h)

        # step 4: save the output frame to a specified destination
        cv2.imwrite(path + 'pano_' + '{:04}'.format(i) + '.jpg', output)
