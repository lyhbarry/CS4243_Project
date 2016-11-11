import outputgenerator as og
import homography as hg
import topdownprojector as tdp
import numpy as np
import cv2


font = cv2.FONT_HERSHEY_PLAIN   # Font-type


#Wrapper function for the per-frame loop logic
def frame_loop(vid, writer, vid_frame, prev_vid_frame, static_points, static_corr_points, active_points, mapping, entityPos, entityPos_prev, static_lk_params, player_lk_params):
    #Update feature locations
    gray_img = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)
    gray_old_frame = cv2.cvtColor(prev_vid_frame, cv2.COLOR_BGR2GRAY)
    static_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, np.array(static_points, dtype=np.float32), None, **static_lk_params)
    static_points = static_points.tolist()
    active_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, np.array(active_points, dtype=np.float32), None, **player_lk_params)
    active_points = active_points.tolist()

    #Map to top-down
    H = hg.find_homography(static_points, static_corr_points)
    top_down_points = tdp.toPlaneCoordinates2D(active_points, H, normalize=False)

    for k, v in mapping.items():
        entityPos_prev[k] = entityPos[k]
        entityPos[k] = top_down_points[v]

    # Without any player position info, Using original frame as full court frame
    # This part should be changed to update
    output_frame = og.generate_frame(vid_frame, vid_frame, outputDim, entityPos, entityPos_prev, entityDist)

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

    #Update step
    prev_vid_frame = vid_frame[:]
    ret, vid_frame = vid.read()
    return ret, vid_frame, prev_vid_frame, static_points, static_corr_points, active_points, entityPos, entityPos_prev

#Model to use for LK tracker, using Ben's model
static_lk_params = dict(winSize  = (10,10),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
player_lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Here is a sample run of how it works
src = 'input/beachVolleyball1.mov'
writer = og.output_writer(src)

vid = cv2.VideoCapture(src)

ret, vid_frame = vid.read()
prev_vid_frame = vid_frame[:]
outputDim = vid_frame.shape

entityDist = {'a1' : 0., 'a2' : 0.,\
              'b1' : 0., 'b2' : 0.}

static_points = [
    [294, 84],
    [440, 136],
    [39, 142],
    [478, 116],
    [349, 69]]
static_corr_points = [
    [0.5, -0.1], #Umpire-side pole
    [0., 1.], #Umpire-side right corner
    [0.5, 1.1], #Non-umpire-side pole
    [1., -0.3], #Flag guy's foot
    [0.5, -0.5]] #Person sitting by london 2012 text
active_points = [
    [492, 236], #'a1', player serving
    [207, 114], #'a2', player near net
    [111, 79], #'b1', opposite nearer bottom
    [165, 68], #'b2', opposite nearer top
    [498, 187]] #'ball', expected to be hard to track

#Map to top-down
H = hg.find_homography(static_points, static_corr_points)
top_down_points = tdp.toPlaneCoordinates2D(active_points, H, normalize=False)

mapping = {'b1':0, 'b2':1, 'a1':2, 'a2':3, 'ball':4}
entityPos = {}
entityPos_prev = {}

for k, v in mapping.items():
    entityPos_prev[k] = top_down_points[v]
    entityPos[k] = top_down_points[v]

entityPos_list = []
while ret:
    ret, vid_frame, prev_vid_frame, static_points, static_corr_points, active_points, entityPos, entityPos_prev = frame_loop(vid, writer, vid_frame, prev_vid_frame, static_points, static_corr_points, active_points, mapping, entityPos, entityPos_prev, static_lk_params, player_lk_params)
    entityPos_list.append(entityPos)

vid.release()
writer.release()
cv2.destroyAllWindows()
