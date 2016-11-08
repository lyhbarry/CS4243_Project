import cv2
import numpy as np
import os

class Stitcher:
    fd = None
    video_writer = None

    def __init__(self, threshold=300):
        self.fd = cv2.SURF(threshold)


    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, dsc = self.fd.detectAndCompute(gray, None)
        return kp, dsc

    '''Finds strong corresponding features in the two given vectors.'''
    def match_flann(self, desc1, desc2, r_threshold=0.06):
        # Adapted from <http://stackoverflow.com/a/8311498/72470>.

        # Build a kd-tree from the second feature vector.
        FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
        flann = cv2.flann_Index(desc2, {'algorithm': FLANN_INDEX_KDTREE, 'trees': 4})

        # For each feature in desc1, find the two closest ones in desc2.
        (idx2, dist) = flann.knnSearch(desc1, 2, params={})  # bug: need empty {}

        # Create a mask that indicates if the first-found item is sufficiently
        # closer than the second-found, to check if the match is robust.
        mask = dist[:, 0] / dist[:, 1] < r_threshold

        # Only return robust feature pairs.
        idx1 = np.arange(len(desc1))
        pairs = np.int32(zip(idx1, idx2[:, 0]))
        return pairs[mask]


    def match_correspondences(self, kp1, dsc1, kp2, dsc2):
        match = self.match_flann(dsc1, dsc2)

        points1 = []
        points2 = []

        for i in range(0, len(match)):
            points1.append(kp1[match[i][0]].pt)
            points2.append(kp2[match[i][1]].pt)

        points1 = np.array(points1)
        points2 = np.array(points2)

        return points1, points2

    def calculate_offset(self, src_size, dst_size, homography):
        # dst_offset is of format [x, y]
        dst_offset = homography[0:2, 2]

        # size is of format (h, w)
        size = ((src_size[1] + int(abs(dst_offset[0]))), (src_size[0] + int(abs(dst_offset[1]))))

        return size, dst_offset

    '''
    Args:
        frames (list of ndarray): list of frames in the video
        fps (int)               : fps count of the video
        area (array)            : area to use for monitoring of keypoints and descriptors
        disreg (int)            : start index of frames to disregard for monitoring
        index (int)             : working index of video
    '''
    def do_main(self, frames, fps, area, disreg, index):
        image_prev = frames[0]
        # area in format of [left top right bottom]
        monitor_left = 0
        monitor_top = 0
        monitor_right = frames[0].shape[1]
        monitor_bottom = frames[0].shape[0]

        if area[0] is not None:
            monitor_left = int(area[0])
        if area[1] is not None:
            monitor_top = int(area[1])
        if area[2] is not None:
            monitor_right = int(area[2])
        if area[3] is not None:
            monitor_bottom = int(area[3])

        image_prev = image_prev[monitor_top:monitor_bottom, monitor_left:monitor_right]

        kp_prev, dsc_prev = self.extract_features(image_prev)

        homo_prev = np.eye(3)

        # in the form (left top right bottom)
        offset_list = []
        homo_list = []

        offset_list.append([0, 0, frames[0].shape[1], frames[0].shape[0]])
        homo_list.append(homo_prev)

        frames_count = len(frames)
        if disreg is not None and disreg < len(frames):
            frames_count = disreg

        for i in range(1, frames_count):
            image = frames[i]
            image = image[monitor_top:monitor_bottom, monitor_left:monitor_right]

            kp, dsc = self.extract_features(image)

            d_prev = cv2.drawKeypoints(image_prev, kp_prev)
            d = cv2.drawKeypoints(image, kp)

            cv2.imshow("prev", d_prev)
            cv2.imshow("curr", d)
            cv2.waitKey(1)

            points1, points2 = self.match_correspondences(kp_prev, dsc_prev, kp, dsc)

            if len(points1) >= 4 and len(points2) >= 4:
                (homo, _) = cv2.findHomography(points2, points1, cv2.RANSAC)

                homo = np.dot(homo_prev, homo)

                homo_list.append(homo)

                (h, w) = frames[i].shape[:2]
                size_mat = np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])

                # From http://stackoverflow.com/questions/22220253/cvwarpperspective-only-shows-part-of-warped-image:
                # H * S = [[h00 h01 h02]  * [[0 w w 0]
                #          [h10 h11 h12]     [0 0 h h]
                #          [h20 h21 h22]]    [1 1 1 1]]
                #
                #       = [[h02     w * h00 + h02   w * h00 + h * h01 + h02     h * h01 + h02]
                #          [h12     w * h10 + h12   w * h10 + h * h11 + h12     h * h11 + h12]
                #          [h22     w * h20 + h12   w * h20 + h * h21 + h22     h * h21 + h22]]

                size_mat = np.dot(homo, size_mat)

                left = np.amin(size_mat[0] / size_mat[2])
                top = np.amin(size_mat[1] / size_mat[2])
                right = np.amax(size_mat[0] / size_mat[2])
                bottom = np.amax(size_mat[1] / size_mat[2])

                offset_list.append([left, top, right, bottom])

            image_prev = image
            kp_prev, dsc_prev = kp, dsc
            homo_prev = homo
        cv2.destroyAllWindows()

        # left top right bottom
        offset_list = np.array(offset_list)
        output = []

        offset_left = np.amin(offset_list[:, 0])
        offset_top = np.amin(offset_list[:, 1])
        offset_right = np.amax(offset_list[:, 2])
        offset_bottom = np.amax(offset_list[:, 3])

        max_w = int(round(abs(offset_left) + offset_right))
        max_h = int(round(abs(offset_top) + offset_bottom))

        if not os.path.exists(str(index) + '_test/'):
            os.makedirs(str(index) + '_test/')

        for i in range(len(offset_list)):
            temp_homo = np.array(homo_list[i], copy=True)
            temp_homo[0, 2] += abs(offset_left)
            temp_homo[1, 2] += abs(offset_top)

            print str(i) + ") temp_homo:\n", temp_homo
            print "\tsize: (" + str(max_h) + ", " + str(max_w) + ")"
            print "\toffset:", offset_list[i]

            warped = cv2.warpPerspective(frames[i], temp_homo, (max_w, max_h), flags = cv2.INTER_NEAREST)
            print "----------------------------------------------------------------------------------------------------"
            output.append(warped)

            if i is 0:
                frame = warped
            else:
                grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                roi = cv2.bitwise_and(warped, warped, mask=mask)
                overlay = cv2.bitwise_and(frame, frame, mask=mask_inv)
                frame = cv2.add(overlay, roi)
                cv2.imshow("output", frame)
                cv2.waitKey(1)

            cv2.imwrite(str(index) + '_test/warped_' + '{:04}'.format(i) + ".jpg", frame)
            print "Processed", i

        print "[NOTE] Check out \"" + str(index) + "_test\" folder for output files."

        cv2.destroyAllWindows()

        if False:
            if self.video_writer is None:
                vid_name = str(index) + '_actual.avi'
                self.video_writer = cv2.VideoWriter(vid_name, cv2.cv.CV_FOURCC('D', 'I', 'V', 'X'), fps, (max_w, max_h))

            frame = output[0]
            self.video_writer.write(frame)

            for i in range(1, len(output)):
                (h1, w1) = output[i].shape[:2]

                for w in range(w1):
                    for h in range(h1):
                        if output[i][h, w, 0] > 20 or output[i][h, w, 1] > 20 or output[i][h, w, 2] > 20:
                            frame[h, w] = output[i][h, w]

                self.video_writer.write(frame)
                print "Processed", i

            self.video_writer.release()
            self.video_writer = None
