import outputgenerator as og
import homography as hg
import topdownprojector as tdp
import numpy as np
import cv2
import re
import feature_tracker_2 as ft2
import rw

from archival_stitcher import Stitcher

font = cv2.FONT_HERSHEY_PLAIN   # Font-type


#Wrapper function for the per-frame loop logic
def frame_loop(vid, writer, vid_frame, static_points, static_corr_points, next_static_corr_points, active_points, mapping, entityPos, entityDist, entityPos_prev, static_lk_params, player_lk_params, new_feature_points, features_to_retrack, player_retrack_frames, new_retrack_player_points, players_to_retrack, video_number):
    #Update feature locations
    old_number_of_static = len(static_points)
    gray_previous_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
    player_retrack_frame = player_retrack_frames.pop(0)

    entityPos_list = []
    for i in range(1, frame_count):
        ret, vid_frame = vid.read()
        outputDim = vid_frame.shape
        gray_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)

        full_court_frame = cv2.imread(str(video_number) + '_test/warped_' + '{:04}'.format(i) + '.jpg')
        full_court_frame = cv2.resize(full_court_frame, (outputDim[0], outputDim[1]))

        """
        gray_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
        gray_old_frame = cv2.cvtColor(prev_vid_frame, cv2.COLOR_BGR2GRAY)
        static_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, np.array(static_points, dtype=np.float32), None, **static_lk_params)
        static_points = static_points.tolist()
        active_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, np.array(active_points, dtype=np.float32), None, **player_lk_params)
        active_points = active_points.tolist()
        """
        static_points, new_feature_points, features_to_retrack, static_corr_points, next_static_corr_points = ft2.track_static_points(gray_previous_frame, gray_frame, static_points, static_lk_params, new_feature_points, features_to_retrack, i, video_number, old_number_of_static, static_corr_points, next_static_corr_points)
        active_points, players_to_retrack, new_retrack_player_points, player_retrack_frame, player_retrack_frames = ft2.track_players(gray_previous_frame, gray_frame, active_points, player_lk_params, new_retrack_player_points, player_retrack_frame, player_retrack_frames, players_to_retrack, i, video_number)
        #print players_to_retrack
        #gray_old_frame = gray_img.copy()
        #print active_points
        #Map to top-down

        H = hg.find_homography(static_points, static_corr_points)
        top_down_points = tdp.toPlaneCoordinates2D(active_points, H, normalize=False)

        for k, v in mapping.items():
            entityPos_prev[k] = entityPos[k]
            entityPos[k] = top_down_points[v]

        # Without any player position info, Using original frame as full court frame
        # This part should be changed to update
        output_frame = og.generate_frame(video_number, vid_frame, full_court_frame, outputDim, entityPos, entityPos_prev, entityDist)

        #Additional step, mark all the points
        counter = 0
        for s_u, s_v in static_points:
            pos = (int(s_u), int(s_v))
            cv2.circle(output_frame, pos, 10, (0,0,255), -1)
            cv2.putText(output_frame, 'o'+str(counter), pos, font, 1.5, (255,255,255), 1, cv2.CV_AA)
            counter+=1
        counter=0
        for a_u, a_v in active_points:
            pos = (int(a_u), int(a_v))
            cv2.circle(output_frame, pos, 10, (255,0,0), -1)
            cv2.putText(output_frame, '+'+str(counter), pos, font, 1.5, (255,255,255), 1, cv2.CV_AA)
            counter+=1


        writer.write(output_frame)
        cv2.imshow("sample", output_frame)
        cv2.waitKey(10)

        static_points = static_points.reshape(-1,1,2)
        active_points = active_points.reshape(-1,1,2)
        #Update step
        gray_previous_frame = gray_frame.copy()
        # prev_vid_frame = vid_frame[:]
        # ret, vid_frame = vid.read()
        entityPos_list.append(entityPos)
        #return ret, vid_frame, prev_vid_frame, static_points, static_corr_points, active_points, entityPos, entityPos_prev
    return entityPos_list
#Model to use for LK tracker, using Ben's model
static_lk_params = dict(winSize  = (10,10),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
player_lk_params = dict(winSize  = (5,5),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Video files to run
# Does not work for video 4 and 5
sources = [
           'input/beachVolleyball1.mov',
           'input/beachVolleyball2.mov',
           'input/beachVolleyball3.mov',
           # 'input/beachVolleyball4.mov',
           # 'input/beachVolleyball5.mov',
           'input/beachVolleyball6.mov',
           'input/beachVolleyball7.mov'
           ]

for src in sources:
    writer = og.output_writer(src)

    vid = cv2.VideoCapture(src)

    video_number = int(re.search(r'\d+', src).group())

    # Comment out this section to not run the stitching
    frames, fps = rw.read(src)
    stitcher = Stitcher()
    stitcher.do_main(frames, fps, [None, None, None, frames[0].shape[0] / 2], None, video_number)

    ret, vid_frame = vid.read()
    #prev_vid_frame = vid_frame[:]
    outputDim = vid_frame.shape

    entityDist = {'a1' : 0., 'a2' : 0.,\
                  'b1' : 0., 'b2' : 0.}
    static_points, active_points, new_feature_points, frame_count, player_retrack_frames, players_to_retrack, new_retrack_player_points, features_to_retrack, static_corr_points, next_static_corr_points = ft2.get_video_initial_points(video_number)
    """
    static_points = [
        [294, 84],
        [440, 136],
        [39, 142],
        [478, 116],
        [349, 69]]
    """
    first_frame_static = ft2.restructure_array(static_points)
    """
    static_corr_points = [
        [0.5, -0.1], #Umpire-side pole
        [0., 1.], #Umpire-side right corner
        [0.5, 1.1], #Non-umpire-side pole
        [1., -0.3], #Flag guy's foot
        [0.5, -0.5]] #Person sitting by london 2012 text
    """
    """
    static_corr_points = [
        [0.5, -0.1],
        [1.0, 0.5],
        [0.5, 1.1],
        [0.5, 1.0]]
    """
    """
    active_points = [
        [492, 236], #'a1', player serving
        [207, 114], #'a2', player near net
        [111, 79], #'b1', opposite nearer bottom
        [165, 68], #'b2', opposite nearer top
        [498, 187]] #'ball', expected to be hard to track
    """
    first_frame_players = ft2.restructure_array(active_points)
    #Map to top-down
    H = hg.find_homography(first_frame_static, static_corr_points)
    #alpha = compute_alpha(H, static_corr_points)
    top_down_points = tdp.toPlaneCoordinates2D(first_frame_players, H, normalize=False)

    mapping = {'b1':0, 'b2':1, 'a1':2, 'a2':3}#, 'ball':4}
    entityPos = {}
    entityPos_prev = {}

    for k, v in mapping.items():
        entityPos_prev[k] = top_down_points[v]
        entityPos[k] = top_down_points[v]

    entityPos_list = frame_loop(vid, writer, vid_frame, static_points, static_corr_points, next_static_corr_points, active_points, mapping, entityPos, entityDist, entityPos_prev, static_lk_params, player_lk_params, new_feature_points, features_to_retrack, player_retrack_frames, new_retrack_player_points, players_to_retrack, video_number)
    #entityPos_list.append(entityPos)

    vid.release()
    writer.release()
    cv2.destroyAllWindows()
