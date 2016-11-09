import cv2
import cv2.cv as cv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def retrack_players(new_player_points, player_to_retrack, new_retrack_player_points, img, print_frame_number):

    print_frame_number = get_frame_grid(img, print_frame_number)
    new_retrack_player_point = new_retrack_player_points[0]
    new_retrack_player_points = np.delete(new_retrack_player_points,[0], axis=0)
    new_player_points[player_to_retrack][0] = new_retrack_player_point 

    return new_player_points, new_retrack_player_points, print_frame_number


def check_for_large_movement_of_static_points(new_static_points, err, st):

    for k,err in enumerate(err):
        if err > 15.0:
            print "DELETING LOL"
            print err
            print new_static_points[k]
            new_static_points = np.delete(new_static_points, [k], axis=0)
            err = np.delete(err, [k], axis=0)
            st = np.delete(st, [k], axis=0)
    return new_static_points, err, st        

def check_static_points(number_of_static, old_number_of_static, good_new, img, print_frame_number, new_feature_points, frame_count):
    if (number_of_static < old_number_of_static):
        print_frame_number = get_frame_grid(img, print_frame_number)
        print frame_count
        new_feature = new_feature_points[0]
        print new_feature
        new_feature_points = np.delete(new_feature_points,[0], axis=0)
        print "BEN SOON OVER HERE"
        print new_feature_points
        good_new = np.append(good_new, new_feature, axis=0)
        print good_new
        good_new = good_new.astype(np.float32)
    return good_new, new_feature_points, print_frame_number    

def get_frame_grid(img, print_frame_number):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.savefig('frame_' + str(print_frame_number) + '.jpg')
    plt.close()

    print_frame_number = print_frame_number + 1

    return print_frame_number

def get_video_initial_points(video_number):

    """
    The coordinates declared are static points on the video which are inside the frame at all times
    """
    if video_number == 1:
        # 1st video static points
        
        static_points = np.zeros((4,1,2))
        
        static_points[0][0] = [309, 77] #good  bottom of umpire pole
        static_points[1][0] = [353, 193] # good middle of court
        static_points[2][0] = [39, 142] # good bottom of left pole
        static_points[3][0] = [88, 144] # left side of the court
        static_points = static_points.astype(np.float32)
        
        # 1st video player points
        # players are ordered from left to right in the first frame
        player_points = np.zeros((4,1,2))
        player_points[0][0] = [94, 80]
        player_points[1][0] = [179, 66]
        player_points[2][0] = [226, 106]
        player_points[3][0] = [498, 236]
        player_points = player_points.astype(np.float32)
        print static_points
        
        new_feature_points = np.zeros((2,1,2))


        player_retrack_frames = [373, 420, 580, 620, 680]
        players_to_retrack = [3, 3, 3, 1, 0]
        new_retrack_player_points = np.zeros((5,1,2))
        new_retrack_player_points[0][0] = [328, 209]
        new_retrack_player_points[1][0] = [367, 224]
        new_retrack_player_points[2][0] = [373, 173]
        new_retrack_player_points[3][0] = [271, 115]
        new_retrack_player_points[4][0] = [297, 32]


        frame_count = 746
        
    elif video_number == 2:    
        #2nd video static points
        static_points = np.zeros((4,1,2))
        static_points[0][0] = [535, 142]
        static_points[1][0] = [574, 82]
        static_points[2][0] = [531, 89] 
        static_points[3][0] = [580, 128]
        static_points = static_points.astype(np.float32)
        """
        #2nd video player points
        player_points = np.zeros((4,1,2))
        player_points[0][0] = p0[60] # this side / left side
        player_points[1][0] = p0[27] # other side /right side
        player_points[2][0] = p0[29] # other side / right side
        player_points[3][0] = p0[4] # other side / left side
        player_points = player_points.astype(np.float32)
        """
        new_feature_points = np.zeros((2,1,2))

    elif video_number == 3:    
        # 3rd video static points
        static_points = np.zeros((4,1,2))
        static_points[0][0] = [170, 141] #nearer pole
        static_points[1][0] = [163, 182]
        static_points[2][0] = [160, 232]
        static_points[3][0] = [493, 281]
        #static_points[3][0] = [238, 232]
        static_points = static_points.astype(np.float32)
        """
        # 3rd video player points
        player_points = np.zeros((3,1,2))
        player_points[0][0] = p0[-12] # guy bending over
        player_points[1][0] = p0[9] # guy serving
        player_points[2][0] = p0[31] # guy serving
        player_points = player_points.astype(np.float32)
        """

        new_feature_points = np.zeros((6,1,2))
        new_feature_points[0][0] = [551, 190]
        new_feature_points[1][0] = [26, 190]
        new_feature_points[2][0] = [37, 116]
        new_feature_points[3][0] = [627, 136]
        new_feature_points[4][0] = [94, 180]
        new_feature_points[5][0] = [396, 276]
    

    elif video_number == 5:
        # 5th video static points
        
        static_points = np.zeros((5,1,2))
        #static_points[0][0] = p0[-8]  #corner of the podium of the umpire
        #static_points[1][0] = p0[12]  #bottom left coorner of the net
        static_points[0][0] = [455, 164]
        static_points[1][0] = [203, 138]
        static_points[2][0] = [483, 168]
        static_points[3][0] = [102, 258]
        static_points[4][0] = [202, 173]
        static_points = static_points.astype(np.float32)
        """
        player_points = np.zeros((6,1,2))
        player_points[0][0] = p0[35] # other side/ right guy      needs work
        player_points[1][0] = p0[24] # other side/ left guy       good
        player_points[2][0] = p0[33] # other side / left guy      good
        player_points[3][0] = p0[42] # this side / further away   needs work
        player_points[4][0] = p0[19] # this side /closer guy      good
        player_points[5][0] = p0[21] # this side / closer guy     good
        player_points = player_points.astype(np.float32)
        """

        
        new_feature_points = np.zeros((2,1,2))
        new_feature_points[0][0] = [131, 230]
        new_feature_points[1][0] = [446, 231]
        new_feature_points = new_feature_points.astype(np.float32)

    elif video_number == 6:
        # 6th video static points
        
        static_points = np.zeros((4,1,2))
        static_points[0][0] = [442, 190] #umpire pole
        static_points[1][0] = [562, 284] #top left corner of five sign, but might move down due to ball
        static_points[2][0] = [557, 164]
        static_points[3][0] = [590, 300]
        static_points = static_points.astype(np.float32)
        """
        player_points = np.zeros((1,1,2))
        """
    elif video_number == 7:
        # 7th video static points
        
        static_points = np.zeros((4,1,2))
        static_points[0][0] = [76, 176] # umpire
        static_points[1][0] = [214, 189] # five sign top right corner
        static_points[2][0] = [66, 207] # pole nearer to the camera
        static_points[3][0] = [197,334]
        static_points = static_points.astype(np.float32)
        """
        player_points = np.zeros((1,1,2))
        player_points[0][0] = p0[21] # right side, bending
        player_points = player_points.astype(np.float32)
        """
    return static_points, player_points, new_feature_points, frame_count, player_retrack_frames, players_to_retrack, new_retrack_player_points

def read_video_file(file_name, video_number):

    # this is to keep track of the frames that are copied for our reference whenever a point goes off screen
    print_frame_number = 0

    cap = cv2.VideoCapture(file_name)
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
    
    # read the first frame of video and convert to grayscale
    _,old_frame = cap.read()

    gray_old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0,255,(100,3))

    #parameters to detect and track feature points
    static_lk_params = dict(winSize  = (10,10),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    player_lk_params = dict(winSize  = (5,5),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict( maxCorners = 200, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
    new_feature_params = dict(maxCorners = 10, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
    #mask = np.zeros_like(old_frame)
    
    static_points, player_points, new_feature_points, frame_count, player_retrack_frames, players_to_retrack, new_retrack_player_points = get_video_initial_points(video_number)

    #number of static points and player points
    number_of_static = len(static_points)
    #number_of_player = len(player_points)
    old_number_of_static = len(static_points)
    #old_number_of_player = len(player_points)

    print_frame_number = get_frame_grid(old_frame, print_frame_number)

    player_retrack_frame = player_retrack_frames.pop(0)
    player_to_retrack = players_to_retrack.pop(0)
    print(player_retrack_frames)
    print(players_to_retrack)
    for i in range(1,frame_count):
        print i
        _,img = cap.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        new_static_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, static_points, None, **static_lk_params)
        
        new_player_points, st_2, err_2 = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, player_points, None, **player_lk_params)
        #print err_2[3]
        new_static_points, err, st = check_for_large_movement_of_static_points(new_static_points, err, st)
        
        if (i == player_retrack_frame):
            new_player_points, new_retrack_player_points, print_frame_number = retrack_players(new_player_points, player_to_retrack, new_retrack_player_points, img, print_frame_number)

            if ((len(players_to_retrack) > 0) and (len(player_retrack_frames) > 0)):
                player_retrack_frame = player_retrack_frames.pop(0)
                player_to_retrack = players_to_retrack.pop(0)

            else:
                player_retrack_frame = 0

        # here the st==1 means that the point is found in the next frame
        # so it means that it asks for the "good" points from the
        # calcOpticalFlowPyrLK function
        good_new = new_static_points[st==1]
        good_old = static_points[st==1]

        # here we compare number of found static to the previous number of static
        # points to see if we have lost any points
        number_of_static = len(good_new)
        
        good_new, new_feature_points, print_frame_number = check_static_points(number_of_static, old_number_of_static, good_new, img, print_frame_number, new_feature_points, frame_count)

        # supposed to regain same number of points
        number_of_static = len(good_new)
        
        for j,(new,old) in enumerate(zip(good_new,good_old)):
            
            a,b = new.ravel()
            c,d = old.ravel()
            #print "HERE"
            #print a,b,c,d,j, (color[j].tolist())
            #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(img,(a,b),5,color[j].tolist(),-1)
        
        good_new_2 = new_player_points[st_2==1]
        good_old_2 = player_points[st_2==1]

        for i,(new,old) in enumerate(zip(good_new_2,good_old_2)):
            a,b = new.ravel()
            c,d = old.ravel()
            #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(img,(a,b),5,color[i].tolist(),-1)
        
        #image = cv2.add(img,mask)
        
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        gray_old_frame = gray_img.copy()

        static_points = good_new.reshape(-1,1,2)
        player_points = good_new_2.reshape(-1,1,2)

def main():
    file_name = "input/beachVolleyball1.mov"
    video_number = int(raw_input('Enter video number: '))
    read_video_file(file_name, video_number)


if __name__ == "__main__":
    main()