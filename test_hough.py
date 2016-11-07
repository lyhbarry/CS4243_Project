"""
Author: Ng Jun Wei

Testing Hough

[Testing] For intersections to get court points of interest.
"""
import cv2
import numpy as np


def find_intersections(frame, offset):
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
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(j, k):
                return j[0] * k[1] - j[1] * k[0]

            div = det(xdiff, ydiff)
            if div is not 0:
                d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
                x = det(d, xdiff) / div
                y = det(d, ydiff) / div
                return x, y
            else:
                return None, None

    intersects = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x, y = intersect(points[i], points[j])
            if x is not None and y is not None:
                if 0 <= x <= w and 0 <= y <= h:
                    # print "x:", x, ", y:", y
                    intersects.append((x, y))

    marked = np.array(frame, copy=True)
    for i in range(len(intersects)):
        cv2.circle(marked, intersects[i], 5, (255, 0, 0), 2)

    cv2.imshow('hough', hough)
    cv2.imshow('marked', marked)
    cv2.waitKey(1)

    return np.array(intersects)

import rw

frames1, fps1 = rw.read('input/beachVolleyball1.mov')
for img in frames1:
    find_intersections(img, [0, img.shape[0]/4, 0, 0])
cv2.destroyAllWindows()
frames1 = None

frames2, fps2 = rw.read('input/beachVolleyball2.mov')
for img in frames2:
    find_intersections(img, [0, img.shape[0]/4, 0, 0])
cv2.destroyAllWindows()
frames2 = None

frames3, fps3 = rw.read('input/beachVolleyball3.mov')
for img in frames3:
    find_intersections(img, [0, img.shape[0]/4, 0, 0])
cv2.destroyAllWindows()
frames3 = None

frames4, fps4 = rw.read('input/beachVolleyball4.mov')
for img in frames4:
    find_intersections(img, [0, img.shape[0]/4, 0, 0])
cv2.destroyAllWindows()
frames4 = None

frames5, fps5 = rw.read('input/beachVolleyball5.mov')
for img in frames5:
    find_intersections(img, [0, 0, 0, 0])
cv2.destroyAllWindows()
frames5 = None

frames6, fps6 = rw.read('input/beachVolleyball6.mov')
for img in frames6:
    find_intersections(img, [0, 0, 0, 0])
cv2.destroyAllWindows()
frames6 = None

frames7, fps7 = rw.read('input/beachVolleyball7.mov')
for img in frames7:
    find_intersections(img, [0, 0, 0, 0])
cv2.destroyAllWindows()
frames7 = None
