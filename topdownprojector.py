"""
Author: Rodson Chue

Top Down Projector

Contains functions that helps map points on an image plane,
to an imaginary top-down plane.

See: computeHomographyToCourt(), toPlaneCoordinates()

<obsolete>
Contains functions that converts camera pixels positions
into top-down coordinates, corrected using camera's intrinsic
and extrinsic parameters
</obsolete>
"""


import numpy as np
import homography as hg


def computeHomographyToCourt(ptsOnImgPlane):
    """
    Computes homography from an image plane to the court's image plane (Top down)
    returns homography H such that:

    Top-down coord = H * image plane coord
    and top-down coord = [a*u, a*v, a]

    The interior of the court is defined to be between [0, 1] on both u, v axis

    ptsOnImgPlane should be a dictionary with the following keys:
    0 - TopLeft
    1 - TopMid
    2 - TopRight
    3 - BottomLeft
    4 - BottomMid
    5 - BottomRight

    Court looks like the following:

    +---------------> +ve u-direction
    | 0----1----2
    | |         |
    | 3----4----5 << corners of the court
    V
    +ve v-direction
    """
    # Need minimum 4 points
    if(len(ptsOnImgPlane)<4):
        return None

    pts_src = []
    pts_dst = []
    ptsAvailable = ptsOnImgPlane.keys()
    if(0 in ptsAvailable):
        pts_src.append(ptsOnImgPlane[0])
        pts_dst.append([0., 0.])
    if(1 in ptsAvailable):
        pts_src.append(ptsOnImgPlane[1])
        pts_dst.append([0.5, 0.])
    if(2 in ptsAvailable):
        pts_src.append(ptsOnImgPlane[2])
        pts_dst.append([1., 0.])
    if(3 in ptsAvailable):
        pts_src.append(ptsOnImgPlane[3])
        pts_dst.append([0., 1.])
    if(4 in ptsAvailable):
        pts_src.append(ptsOnImgPlane[4])
        pts_dst.append([0.5, 1.])
    if(5 in ptsAvailable):
        pts_src.append(ptsOnImgPlane[5])
        pts_dst.append([1., 1.])
    H = hg.find_homography(pts_src, pts_dst)
    return H

def normalizeImgVector(uvVec):
    """
    Given a vector such that [a*u, a*v, a]
    returns the normalized vector [u, v, 1]
    """
    if(uvVec[2]==0):
        return uvVecs
    uvVec[:] = uvVec[:]/uvVec[2]
    return uvVec

def toPlaneCoordinates(pts, H, normalize=True):
    """
    Given a homography H, maps a list of points pts
    into their corresponding points based on H.
    """
    plane_pts = []
    for pt in pts:
        plane_pt = np.dot(H, pt)
        if(normalize):
            plane_pt = normalizeImgVector(plane_pt)

        plane_pts.append(plane_pt)
    return plane_pts

def toPlaneCoordinates2D(pts, H, normalize=True):
    """
    Given a homography H, maps a list of points pts
    into their corresponding points based on H.
    """
    plane_pts = []
    for pt in pts:
        vec = np.asarray([pt[0], pt[1], 1])
        plane_pt = np.dot(H, vec).transpose()
        if(normalize):
            plane_pt = normalizeImgVector(plane_pt)

        plane_pts.append(plane_pt[:2])
    return plane_pts

############# Any code after this point can be considered obsolete #################
def computeRay(pixel, windowSize, camIntrinsic, camExtrinsic):
    """
    Computes the ray-trace vector for a particular pixel
    Given the window size and camera parameters

    Input Parameters:
        pixel:
        The (x,y) coordinate on the window
        windowSize:
        (u,v) size of the window
        camIntrinsic:
        The 3x3 camera intrinsic matrix
        [fx, 0, cx]
        [0 ,fy, cy]
        [0 , 0,  1]
        camExtrinsic:
        Rotation and tranlation matrix combined
        [R T], where R is the rotation matrix, T is the translation


    Returns: ray
        A pair of (numpy) 3-tuples:
        The first of the pair is the unit direction vector
        The second of the pair is one such point on the ray,
        usually the camera.
        E.g ( (0.5,0.5,0.), (0.,0.,0.) ) is a ray that passes through
        (0,0,0) and has a direction vector (0.5,0.5,0.5)

    """
    print "Not implemented yet!"
    return None

def computeIntersect(ray, plane):
    """
    Computes the intersection between the ray and plane

    Input Parameters:
        ray:
        A pair of (numpy) 3-tuples:
        The first of the pair is the unit direction vector
        The second of the pair is one such point on the ray,
        usually the camera.
        E.g ( (0.5,0.5,0.), (0.,0.,0.) ) is a ray that passes through
        (0,0,0) and has a direction vector (0.5,0.5,0.5)

        plane:
        A pair of (numpy) 3-tuples:
        The first of the pair is the unit normal vector of the plane
        The second of the pair is one such point on the ray.

        E.g A plane of z=0 for all points (thus intersects the origin),
        could be ( (0.,0.,1.), (0.,0.,0.) )

    Returns: intersect
        a (numpy) 3-tuple vector.
        (x,y,z) representing the point of intersection
        NOTE: returns None (null) if the ray and plane are parallel

    """

    rayDir = ray[0]
    rayP = ray[1]
    planeN = plane[0]
    planeP = plane[1]

    # Check if Line and plane are parallel
    denom = np.dot(rayDir, planeN)
    if (denom == 0): return None

    d = np.dot((planeP - rayP), planeN) / denom
    intersect = (d * rayDir) + rayP

    return intersect
