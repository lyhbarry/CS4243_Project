"""
Author: Ng Jun Wei

Homography module
"""

import numpy as np


''' Constructs the matrix
Args:
    pts_src (list<tuple<int, int>>): List of points from the source image
    pts_dst (list<tuple<int, int>>): List of points from the destination image
Returns:
    (np.matrix): A matrix to be solved using SVD
'''
def construct_m(pts_p, pts_c):
    A = []
    for i in range(len(pts_p)):
        up = pts_p[i][0]
        vp = pts_p[i][1]
        uc = pts_c[i][0]
        vc = pts_c[i][1]
        A.append([up, vp, 1, 0, 0, 0, -(uc * up), -(uc * vp), -uc])
        A.append([0, 0, 0, up, vp, 1, -(vc * up), -(vc * vp), -vc])
    return np.matrix(A)


''' Calculates the homography matrix between two images
Args:
    pts_src (list<tuple<int, int>>): List of points from the source image
    pts_dst (list<tuple<int, int>>): List of points from the destination image
Returns:
    homo_mat (np.matrix): The homography matrix
'''
def find_homography(pts_src, pts_dsc):
    m = construct_m(pts_src, pts_dsc)

    # note that V is already transposed
    _, _, V = np.linalg.svd(m, full_matrices=True)

    # the last column of V is the homography matrix
    homo_mat = np.matrix(np.reshape(V[-1], (3, 3)))
    # setting all super small values to zero
    homo_mat[abs(homo_mat) < 1e-14] = 0
    # normalising the matrix such that h33 to 1
    homo_mat = homo_mat / homo_mat[-1, -1]

    return homo_mat
