"""
Author: Ng Jun Wei

Stitcher module

Handles the full court panoramic view of the game.
"""

import numpy as np
import homography as hg
import cv2
import math
import os


def generate_corresponding_pairs(points1, points2, threshold=5):
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
    pairs = []

    for i in range(len(points1)):
        for point in points2[:]:
            x2 = math.pow(abs(points1[i][0] - point[0]), 2)
            y2 = math.pow(abs(points1[i][1] - point[1]), 2)
            if math.sqrt(x2 + y2) < threshold:
                pairs.append((points1[i], point))
                points2.remove(point)

    return pairs


def eucl_dist(p1, p2):
    # type: ((int, int), (int, int)) -> float

    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def find_correspondence(pts1, pts2):
    # type: ((int, int)[], (int, int)[]) -> ((int, int)[], (int, int)[])
    """
    Purpose:
        Find corresponding points between two sets of points.
    Parameters:
        pts1 - the first set of points
        pts2 - the second set of points
    Returns:
        ret1 - the list of corresponding points from the first set
        ret2 - the list of corresponding points from the second set
    """
    corrs = {}
    for item in pts1:
        corrs[item] = []
        for item2 in pts2:
            if eucl_dist(item2, item) < 5:
                corrs[item].append(item2)
    corrs = {k: v for k, v in corrs.items() if len(v) > 0}
    ret1 = []
    ret2 = []
    for item in corrs:
        ret1.append(item)
        ret2.append(corrs[item][0])
    return ret1, ret2


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

        # pairs = generate_corresponding_pairs(prev, curr)
        #
        # points_curr = []
        # points_prev = []
        # for j in range(len(pairs)):
        #     points_curr.append(pairs[j][0])
        #     points_prev.append(pairs[j][1])
        # homo_mat = hg.find_homography(points_curr, points_prev)

        # pts_prev, pts_curr = find_correspondence(prev, curr)
        pts_prev = points[i - 1]
        pts_curr = points[i]

        acc = 0
        if len(pts_prev) >= 4 and len(pts_curr) >= 4:
            if len(pts_prev) is len(pts_curr):
                for x in range(len(pts_prev)):
                    acc += eucl_dist(pts_prev[x], pts_curr[x])
        if acc < len(pts_prev):
            homo_mat = np.eye(3)
        else:
            # homo_mat = hg.find_homography(pts_curr, pts_prev)
            homo_mat = cv2.findHomography(np.array(pts_curr, np.float32), np.array(pts_prev, np.float32))
            homo_mat = np.matrix(homo_mat[0])

        print "i =", i
        print "pts_prev:", pts_prev
        print "pts_curr:", pts_curr
        print "homo_mat:\n", homo_mat
        print "-------------------------------------------------"

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

    # From http://stackoverflow.com/questions/22220253/cvwarpperspective-only-shows-part-of-warped-image:
    # H * S = [[h00 h01 h02]  * [[0 w w 0]
    #          [h10 h11 h12]     [0 0 h h]
    #          [h20 h21 h22]]    [1 1 1 1]]
    #
    #       = [[h02     w * h00 + h02   w * h00 + h * h01 + h02     h * h01 + h02]
    #          [h12     w * h10 + h12   w * h10 + h * h11 + h12     h * h11 + h12]
    #          [h22     w * h20 + h12   w * h20 + h * h21 + h22     h * h21 + h22]]

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
    if not os.path.exists(path):
        os.makedirs(path)

    # step 1: generate list of homography matrices between each frames
    matrices = generate_homography_matrix_list(points)

    temp = np.eye(3)

    homos = []
    positions = []
    for i in range(len(matrices)): # changed from frame
        temp = np.dot(temp, matrices[i])
        homos.append(temp)

        (h, w) = frames[i].shape[:2]
        l, t, r, b = get_output_size(h, w, temp)
        positions.append([l, t, r, b])
    positions = np.array(positions)

    offset_left = np.amin(positions[:, 0])
    offset_top = np.amin(positions[:, 1])
    offset_right = np.amin(positions[:, 2])
    offset_bottom = np.amin(positions[:, 3])

    max_w = int(round(abs(offset_left) + offset_right))
    max_h = int(round(abs(offset_top) + offset_bottom))

    print "offset_left:", offset_left
    print "offset_top:", offset_top
    print "offset_right:", offset_right
    print "offset_bottom:", offset_bottom
    print "max_w:", max_w
    print "max_h:", max_h

    for i in range(len(homos)): # changed from frame
        # step 2: calculate the homography matrix to use
        homo = np.array(homos[i], copy=True)
        # setting left offset for current frame after homography
        # homo[0, 2] += abs(offset_left)
        # setting top offset for current frame after homography
        # homo[1, 2] += abs(offset_top)

        cv2.imshow('frame', frames[i])

        # step 3: generate the output frame
        # warped = cv2.warpPerspective(frames[i], homo, (max_w, max_h), flags=cv2.INTER_NEAREST)
        warped = cv2.warpPerspective(frames[i], homo, (800, 500), flags=cv2.INTER_NEAREST)
        cv2.imshow('warped', warped)
        cv2.waitKey(1)

        print "showing i:", i

        if i is 0:
            output = warped
        else:
            grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            # obtain the mask of the new frame
            _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
            # obtain the inverse of the mask
            mask_inv = cv2.bitwise_not(mask)
            # mask out the regions that are not updated
            roi = cv2.bitwise_and(warped, warped, mask=mask)
            # mask out the area to be updated
            background = cv2.bitwise_and(output, output, mask=mask_inv)
            # place the updated regions into the ongoing background
            output = cv2.add(background, roi)

        # step 4: save the output frame to a specified destination
        cv2.imwrite(path + 'pano_' + '{:04}'.format(i) + '.jpg', output)

    cv2.destroyAllWindows()
