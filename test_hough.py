"""
Author: Ng Jun Wei

Testing Hough

[Testing] For intersections to get court points of interest.
TODO: Next step is to find how to manually match points correspondence.
"""
import cv2
import numpy as np
import math


def find_intersections(frame, offset):
    # type: (np.array, [int, int, int, int]) -> np.array
    
    (h, w) = frame.shape[:2]

    offset_left = offset[0]
    offset_top = offset[1]
    offset_right = offset[2]
    offset_bottom = offset[3]

    temp = np.array(frame, copy=True)
    temp = temp[offset_top:h-offset_bottom, offset_left:w-offset_right]
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    cv2.imshow('edges', edges)

    hough = np.array(frame, copy=True)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    points = []
    mc = []

    if lines is not None and len(lines) > 0:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho

            x1 = int(x0 + 1000 * (-b) + offset_left)
            y1 = int(y0 + 1000 * a + offset_top)
            x2 = int(x0 - 1000 * (-b) + offset_left)
            y2 = int(y0 - 1000 * a + offset_top)

            points.append(((x1, y1), (x2, y2)))
            cv2.line(hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

        def intersect(line1, line2):
            # type: (((int, int), (int, int)), ((int, int), (int, int))) -> (int, int)
            
            # y = mx + c
            # m = (y2 - y1)/(x2 - x1)
            if line1[1][0] - line1[0][0] is not 0:
                m1 = (line1[1][1] - line1[0][1])/float(line1[1][0] - line1[0][0])
            else:
                m1 = None
            if line2[1][0] - line2[0][0] is not 0:
                m2 = (line2[1][1] - line2[0][1])/float(line2[1][0] - line2[0][0])
            else:
                m2 = None
            if m1 is not None and m2 is not None and m1 != m2:
                # c = y - mx
                c1 = line1[0][1] - (m1 * line1[0][0])
                c2 = line2[0][1] - (m2 * line2[0][0])
                # x = (c2 - c1)/(m1 - m2)
                # y = (m1 * ((c2 - c1)/(m1 - m2))) + c1
                x = (c2 - c1)/float(m1 - m2)
                y = (m1 * ((c2 - c1)/float(m1 - m2))) + c1
                return int(round(x)), int(round(y))
            else:
                return None, None

        #def intersect(line1, line2):
        #    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        #    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        #
        #    def det(j, k):
        #        return j[0] * k[1] - j[1] * k[0]
        #
        #    div = det(xdiff, ydiff)
        #    if div is not 0:
        #        d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
        #        x = det(d, xdiff) / div
        #        y = det(d, ydiff) / div
        #        return x, y
        #    else:
        #        return None, None

    intersects = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x, y = intersect(points[i], points[j])
            if x is not None and y is not None:
                if 0 <= x <= w and 0 <= y <= h:
                    intersects.append((x, y))

    def eucl_dist(p1, p2):
        # type: ((int, int), (int, int)) -> float
        
        return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))

    def group(points):
        # type: ((int, int)[]) -> (int, int)[]
        
        dic = {}
        if len(points) > 0:
            dic[points[0]] = []
            points.remove(points[0])
        
        while len(points) > 0:
            for item in dic:
                for i in xrange(len(points) - 1, -1, -1):
                    if eucl_dist(item, points[i]) < 10:
                        dic[item].append(points[i])
                        points.remove(points[i])
            if len(points) > 0:
                dic[points[0]] = []
                points.remove(points[0])
        ret = []
        for item in dic:
            xsum = item[0]
            ysum = item[1]
            for p in dic[item]:
                xsum += p[0]
                ysum += p[1]
            ret.append((int(round(xsum / float(len(dic[item]) + 1))), int(round(ysum / float(len(dic[item]) + 1)))))
        return ret
    intersects = group(intersects)
    
    marked = np.array(frame, copy=True)
    for i in range(len(intersects)):
        print "intersects at", intersects[i]
        cv2.circle(marked, intersects[i], 5, (255, 0, 0), 2)

    cv2.imshow('hough', hough)
    cv2.imshow('marked', marked)
    cv2.waitKey(0)

    return np.array(intersects)

import rw

frames1, fps1 = rw.read('input/beachVolleyball1.mov')
img = frames1[0]
find_intersections(img, [0, img.shape[0]/4, 0, 0])
cv2.destroyAllWindows()
