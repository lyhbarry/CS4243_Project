"""
Top Down Projector

Contains functions that converts camera pixels positions
into top-down coordinates, corrected using camera's intrinsic
and extrinsic parameters
"""
import numpy as np

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
