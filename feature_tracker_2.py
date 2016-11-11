import cv2
import cv2.cv as cv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def track_players(gray_old_frame, gray_img, player_points, player_lk_params, new_retrack_player_points, player_retrack_frame, player_retrack_frames, players_to_retrack, frame_number, video_number):
    #print frame_number
    new_player_points, st_2, err_2 = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, player_points, None, **player_lk_params)
    #print new_player_points
    if (frame_number == player_retrack_frame):
        player_to_retrack = players_to_retrack.pop(0)
        print player_retrack_frames
        #if ((video_number == 3))
        # if ((video_number == 3) and (frame_number == 33)):
        #     new_player_points = np.append(new_player_points, [[[0,0]]], axis=0)
        #     st_2 = np.append(st_2, [[1]], axis=0)
        # if ((video_number == 3) and (frame_number == 69)):
        #     new_player_points = np.append(new_player_points, [[[0,0]]], axis=0)
        #     st_2 = np.append(st_2, [[1]], axis=0)    
        new_player_points, new_retrack_player_points = retrack_players(new_player_points, player_to_retrack, new_retrack_player_points)
        
        if ((len(players_to_retrack) > 0) and (len(player_retrack_frames) > 0)):
            player_retrack_frame = player_retrack_frames.pop(0)

        else:
            player_retrack_frame = 0

    #good_new_2 = new_player_points[st_2==1]
    good_new_2 = np.zeros((4,2))
    print good_new_2
    good_new_2[0] = new_player_points[0][0]
    good_new_2[1] = new_player_points[1][0]
    good_new_2[2] = new_player_points[2][0]
    good_new_2[3] = new_player_points[3][0]
    good_new_2 = good_new_2.astype(np.float32)
    # cv2.imshow('frame',img)
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
    # gray_old_frame = gray_img.copy()

    return good_new_2, players_to_retrack, new_retrack_player_points, player_retrack_frame, player_retrack_frames

def track_static_points(gray_old_frame, gray_img, static_points, static_lk_params, new_feature_points, features_to_retrack, frame_number, video_number, old_number_of_static, static_corr_points, next_static_corr_points):

    new_static_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, static_points, None, **static_lk_params)

    new_static_points, new_feature_points, features_to_retrack, static_corr_points, next_static_corr_points = check_for_large_movement_of_static_points(new_static_points, st, err, new_feature_points, features_to_retrack, static_corr_points, next_static_corr_points)

    good_new = new_static_points[st==1]
    good_old = static_points[st==1]

    number_of_static = len(good_new)

    #special case for video 2
    #if ((frame_number))
    # if ((frame_number == 256) and (video_number == 1)):
    #     new_static_points[3][0] = [342, 109]
    #     static_corr_points[3] = [0.75, 0.0]
    #     print static_corr_points
    # if ((frame_number == 305) and (video_number == 1)):
    #     new_static_points[3][0] = [469, 163] 
    #     static_corr_points[3] = [1.0, 0.0]
    #     print static_corr_points
    if ((frame_number == 132) and (video_number == 2)):
        new_static_points[1][0] = [85, 167]
        static_corr_points[1] = next_static_corr_points[0]
        next_static_corr_points = next_static_corr_points[1:]
    if ((frame_number == 372) and (video_number == 2)):
        new_static_points[0][0] = [277, 239]  
        static_corr_points[0] = next_static_corr_points[0]
        next_static_corr_points = next_static_corr_points[1:]
    if ((frame_number == 72) and (video_number == 4)):
        new_static_points[2][0] = [475, 276]    
    if ((frame_number == 106) and (video_number == 4)):
        new_static_points[1][0] = [520, 268]
        new_static_points[3][0] = [458, 160]  
    if ((frame_number == 156) and (video_number == 4)):
        new_static_points[3][0] = [154, 167]                  

    new_static_points, new_feature_points, features_to_retrack, st, static_corr_points, next_static_corr_points = check_static_points(number_of_static, old_number_of_static, new_static_points, new_feature_points, features_to_retrack, st, static_corr_points, next_static_corr_points)
    good_new = new_static_points[st==1]

    return good_new, new_feature_points, features_to_retrack, static_corr_points, next_static_corr_points

def restructure_array(array):
    new_array = []
    for i in range(len(array)):
        print array[i]
        coordinates = array[i][0]
        new_array.append(coordinates)
    return new_array    

def retrack_players_2(new_player_points, player_to_retrack, new_retrack_player_points,img, print_frame_number):
    #print_frame_number = get_frame_grid(img, print_frame_number)
    print "RETRACKING"
    new_retrack_player_point = new_retrack_player_points[0]
    new_retrack_player_points = np.delete(new_retrack_player_points,[0], axis=0)
    new_player_points[player_to_retrack][0] = new_retrack_player_point 
    new_player_points = new_player_points.astype(np.float32)
    return new_player_points, new_retrack_player_points, print_frame_number

def retrack_players(new_player_points, player_to_retrack, new_retrack_player_points):
    #print_frame_number = get_frame_grid(img, print_frame_number)
    print "RETRACKING"
    new_retrack_player_point = new_retrack_player_points[0]
    new_retrack_player_points = np.delete(new_retrack_player_points,[0], axis=0)
    new_player_points[player_to_retrack][0] = new_retrack_player_point 
    new_player_points = new_player_points.astype(np.float32)
    return new_player_points, new_retrack_player_points#, print_frame_number

def check_for_large_movement_of_static_points_2(new_static_points, st, err, new_feature_points, features_to_retrack, print_frame_number, img):

    for k,(error, status) in enumerate(zip(err,st)):

        if ((error > 15.0) and (status[0] == 1)):
            print "DELETING LOL"

            print_frame_number = get_frame_grid(img, print_frame_number)
            feature_to_retrack = features_to_retrack.pop(0) 
            new_feature = new_feature_points[0]
            new_feature_points = np.delete(new_feature_points,[0], axis=0)
            new_static_points[feature_to_retrack][0] = new_feature
            #err = np.delete(err, [k], axis=0)
            #st = np.delete(st, [k], axis=0)
    return new_static_points, new_feature_points, features_to_retrack, print_frame_number

def check_for_large_movement_of_static_points(new_static_points, st, err, new_feature_points, features_to_retrack, static_corr_points, next_static_corr_points):
    for k,(error, status) in enumerate(zip(err,st)):

        if ((error > 15.0) and (status[0] == 1)):
            print "DELETING LOL"

            #print_frame_number = get_frame_grid(img, print_frame_number)
            feature_to_retrack = features_to_retrack.pop(0) 
            new_feature = new_feature_points[0]
            new_feature_points = np.delete(new_feature_points,[0], axis=0)
            new_static_points[feature_to_retrack][0] = new_feature
            static_corr_points[feature_to_retrack] = next_static_corr_points[0]
            next_static_corr_points = next_static_corr_points[1:]
            #err = np.delete(err, [k], axis=0)
            #st = np.delete(st, [k], axis=0)
    return new_static_points, new_feature_points, features_to_retrack, static_corr_points, next_static_corr_points#, print_frame_number

def check_static_points_2(number_of_static, old_number_of_static, new_static_points, img, print_frame_number, new_feature_points, features_to_retrack, st):

    if (number_of_static < old_number_of_static):
        print_frame_number = get_frame_grid(img, print_frame_number)
        feature_to_retrack = features_to_retrack.pop(0)

        new_feature = new_feature_points[0]
        new_feature_points = np.delete(new_feature_points,[0], axis=0)
        new_static_points[feature_to_retrack][0] = new_feature
        new_static_points = new_static_points.astype(np.float32)
        st[feature_to_retrack] = [1]

    return new_static_points, new_feature_points, features_to_retrack, print_frame_number, st 


def check_static_points(number_of_static, old_number_of_static, new_static_points, new_feature_points, features_to_retrack, st, static_corr_points, next_static_corr_points):
    if (number_of_static < old_number_of_static):
        #print_frame_number = get_frame_grid(img, print_frame_number)
        feature_to_retrack = features_to_retrack.pop(0)

        new_feature = new_feature_points[0]
        new_feature_points = np.delete(new_feature_points,[0], axis=0)
        new_static_points[feature_to_retrack][0] = new_feature

        static_corr_points[feature_to_retrack] = next_static_corr_points[0]
        next_static_corr_points = next_static_corr_points[1:]
        new_static_points = new_static_points.astype(np.float32)
        st[feature_to_retrack] = [1]
        #print static_corr_points

    return new_static_points, new_feature_points, features_to_retrack, st, static_corr_points, next_static_corr_points   

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
        #static_points[3][0] = [438, 136]
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
        
        new_feature_points = np.zeros((1,1,2))
        new_feature_points[0][0] = None
        features_to_retrack = []
        # new_feature_points[0][0] = [163, 230]
        # features_to_retrack = [3]

        player_retrack_frames = [373, 420, 580, 620, 680]
        players_to_retrack = [3, 3, 3, 1, 0]
        new_retrack_player_points = np.zeros((5,1,2))
        new_retrack_player_points[0][0] = [328, 209]
        new_retrack_player_points[1][0] = [367, 224]
        new_retrack_player_points[2][0] = [373, 173]
        new_retrack_player_points[3][0] = [271, 115]
        new_retrack_player_points[4][0] = [297, 32]

        static_corr_points = [
            [0.5, -0.1],
            [1.0, 0.5],
            [0.5, 1.1],
            [0.5, 1.0]] 
        next_static_corr_points = []

        frame_count = 746
        
    elif video_number == 2:    
        #2nd video static points
        static_points = np.zeros((4,1,2))
        static_points[0][0] = [210, 109]
        static_points[1][0] = [275, 79]
        static_points[2][0] = [574, 82] 
        static_points[3][0] = [580, 128]
        static_points = static_points.astype(np.float32)
        
        
        #2nd video player points
        player_points = np.zeros((4,1,2))
        player_points[0][0] = [177, 237]
        player_points[1][0] = [424, 200]
        player_points[2][0] = [381, 87]
        player_points[3][0] = [478, 92]
        player_points = player_points.astype(np.float32)
        
        new_feature_points = np.zeros((7,1,2))
        new_feature_points[0][0] = [254, 245]
        new_feature_points[1][0] = [147, 161]
        new_feature_points[2][0] = [147, 142]
        new_feature_points[3][0] = [17, 202]
        new_feature_points[4][0] = [130, 138]
        new_feature_points[5][0] = [530, 286]
        new_feature_points[6][0] = [106, 136]
        features_to_retrack = [0, 0, 1, 1, 1, 2, 3]

        
        player_retrack_frames = [66, 75, 105, 132, 155, 165, 184, 190, 198, 225, 232, 339, 350, 370, 477, 505, 523, 540, 552, 590, 620]
        players_to_retrack = [0, 0, 0, 2, 1, 3, 1, 0, 2, 2, 0, 2, 0, 0, 0, 3, 0, 0, 2, 3, 1]
        new_retrack_player_points = np.zeros((21,1,2))
        new_retrack_player_points[0][0] = [194, 250]
        new_retrack_player_points[1][0] = [187, 205]
        new_retrack_player_points[2][0] = [188, 178]
        new_retrack_player_points[3][0] = [366, 139]
        new_retrack_player_points[4][0] = [302, 250]
        new_retrack_player_points[5][0] = [480, 144]
        new_retrack_player_points[6][0] = [311, 261]
        new_retrack_player_points[7][0] = [254, 229]
        new_retrack_player_points[8][0] = [331, 134]
        new_retrack_player_points[9][0] = [297, 139]
        new_retrack_player_points[10][0] = [249, 206]
        new_retrack_player_points[11][0] = [305, 84]
        new_retrack_player_points[12][0] = [300, 82]
        new_retrack_player_points[13][0] = [297, 111]
        new_retrack_player_points[14][0] = [357, 150]
        new_retrack_player_points[15][0] = [376, 116]
        new_retrack_player_points[16][0] = [321, 124]
        new_retrack_player_points[17][0] = [312, 104]
        new_retrack_player_points[18][0] = [347, 124]
        new_retrack_player_points[19][0] = [376, 124]
        new_retrack_player_points[20][0] = [505, 240]
        
        static_corr_points = [
            [0.5, -0.1],
            [1.0, -0.3],
            [1.3, 1.0],
            [0.5, 1.0]]
        next_static_corr_points = [
            [0.0, 0.5],
            [0.5, -0.1],
            [0.5, -0.3],
            [1.0, -0.3],
            [0.0, 0.0],
            [0.0, 0.5],
            [0.5, -0.3],
            [0.0, 1.0],
            [0.5, -0.1]]  
        
        frame_count = 649


    elif video_number == 3:    
        # 3rd video static points
        static_points = np.zeros((4,1,2))
        static_points[0][0] = [170, 141] # umpire's right shoe
        static_points[1][0] = [163, 182] # base of further pole
        static_points[2][0] = [160, 232] # top of nearer pole / center of court?
        static_points[3][0] = [493, 281] # right lower corner of court
        #static_points[3][0] = [238, 232]
        static_points = static_points.astype(np.float32)
        
        #3rd video player points
        player_points = np.zeros((4,1,2))
        player_points[0][0] = [485, 226]
        player_points[1][0] = [200, 240]
        player_points[2][0] = [0, 0]  # off-screen in initial frame
        player_points[3][0] = [0, 0]  # off-screen in initial frame
        player_points = player_points.astype(np.float32)
        

        new_feature_points = np.zeros((6,1,2))
        new_feature_points[0][0] = [551, 190]
        new_feature_points[1][0] = [26, 190]
        new_feature_points[2][0] = [37, 116]
        new_feature_points[3][0] = [627, 136]
        new_feature_points[4][0] = [94, 180]
        new_feature_points[5][0] = [396, 276]
        
        features_to_retrack = [3, 3, 3, 3, 2, 2]

        player_retrack_frames = [33, 69, 246, 255, 283, 308]
        players_to_retrack = [2, 3, 2, 3, 2, 0]
        new_retrack_player_points = np.zeros((6,1,2))
        new_retrack_player_points[0][0] = [5, 267]
        new_retrack_player_points[1][0] = [158, 212]
        new_retrack_player_points[2][0] = [217, 261]
        new_retrack_player_points[3][0] = [184, 217]
        new_retrack_player_points[4][0] = [208, 231]
        new_retrack_player_points[5][0] = [414, 183]

        static_corr_points = [
            [0.5, -0.5],
            [0.5, -0.3],
            [0.5, 0.5],
            [1.0, 1.0]]

        next_static_corr_points = [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, -0.5],
            [1.0, -0.3],
            [0.25, 0.0],
            [0.75, 1.0]]     

        frame_count = 462


    elif video_number == 4:    
        # 3rd video static points
        static_points = np.zeros((4,1,2))
        static_points[0][0] = [137, 140]
        static_points[1][0] = [246, 274]
        static_points[2][0] = [310, 150]
        static_points[3][0] = [346, 80]
        #static_points[3][0] = [238, 232]
        static_points = static_points.astype(np.float32)
        """
        #3rd video player points
        player_points = np.zeros((4,1,2))
        player_points[0][0] = [485, 226]
        player_points[1][0] = [200, 240]
        player_points[2][0] = [0, 0]  # off-screen in initial frame
        player_points[3][0] = [0, 0]  # off-screen in initial frame
        player_points = player_points.astype(np.float32)
        """

        new_feature_points = np.zeros((5,1,2))
        new_feature_points[0][0] = [514, 153]
        new_feature_points[1][0] = [482, 282]
        new_feature_points[2][0] = [210, 166]
        new_feature_points[3][0] = [506, 167]
        new_feature_points[4][0] = [160, 167]

        features_to_retrack = [2, 0, 1, 2, 3]
        """
        player_retrack_frames = [33, 69, 246, 255, 283, 308]
        players_to_retrack = [2, 3, 2, 3, 2, 0]
        new_retrack_player_points = np.zeros((6,1,2))
        new_retrack_player_points[0][0] = [5, 267]
        new_retrack_player_points[1][0] = [158, 212]
        new_retrack_player_points[2][0] = [217, 261]
        new_retrack_player_points[3][0] = [184, 217]
        new_retrack_player_points[4][0] = [208, 231]
        new_retrack_player_points[5][0] = [414, 183]
        """
        frame_count = 462


    elif video_number == 5:
        # 5th video static points
        
        static_points = np.zeros((4,1,2))
        #ordered left to right on first frame
        static_points[0][0] = [102, 258]
        #static_points[1][0] = [203, 138]
        static_points[1][0] = [202, 173]
        static_points[2][0] = [455, 164]
        static_points[3][0] = [483, 168]
        static_points = static_points.astype(np.float32)
        
        player_points = np.zeros((4,1,2))
        #player_points[0][0] = [250, 324]
        #player_points[0][0] = [333, 324] # USA1
        player_points[0][0] = [333, 240]
        player_points[1][0] = [310, 170] # USA2 
        player_points[2][0] = [286, 141] # Backguy_left
        player_points[3][0] = [347, 141] # Backguy_right
        player_points = player_points.astype(np.float32)
        
        
        new_feature_points = np.zeros((3,1,2))
        new_feature_points[0][0] = [131, 230]
        new_feature_points[1][0] = [446, 231]
        new_feature_points[2][0] = [7, 256]
        new_feature_points = new_feature_points.astype(np.float32)
        #features_to_retrack = [0, 0, 0]
        features_to_retrack = [0, 0, 0]
        
        player_retrack_frames = [150, 165]
        players_to_retrack = [0, 1]
        new_retrack_player_points = np.zeros((2,1,2))
        new_retrack_player_points[0][0] = [339, 306]
        new_retrack_player_points[1][0] = [310, 187]
        #new_retrack_player_points[1][0] = [339, 310]
        #new_retrack_player_points[1][0] = [342, 155]
        
        frame_count = 1170

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
    return static_points, player_points, new_feature_points, frame_count, player_retrack_frames, players_to_retrack, new_retrack_player_points, features_to_retrack, static_corr_points, next_static_corr_points
    #return static_points, new_feature_points, features_to_retrack
    #return static_points, new_feature_points

def read_video_file(file_name, video_number):

    # this is to keep track of the frames that are copied for our reference whenever a point goes off screen
    print_frame_number = 0

    cap = cv2.VideoCapture(file_name)
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
    print frame_count
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
    print_frame_number = get_frame_grid(old_frame, print_frame_number)

    static_points, player_points, new_feature_points, frame_count, player_retrack_frames, players_to_retrack, new_retrack_player_points, features_to_retrack = get_video_initial_points(video_number)
    #static_points, new_feature_points, features_to_retrack = get_video_initial_points(video_number)
    #number of static points and player points
    number_of_static = len(static_points)
    #number_of_player = len(player_points)
    old_number_of_static = len(static_points)
    #old_number_of_player = len(player_points)
    
    player_retrack_frame = player_retrack_frames.pop(0)
    player_to_retrack = players_to_retrack.pop(0)
    print(player_retrack_frames)
    print(players_to_retrack)
    
    #player_retrack_frame = 0

    for i in range(1,frame_count):
        print i
        _,img = cap.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        new_static_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, static_points, None, **static_lk_params)

        new_player_points, st_2, err_2 = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, player_points, None, **player_lk_params)
        print new_player_points
        new_static_points, new_feature_points, features_to_retrack, print_frame_number = check_for_large_movement_of_static_points_2(new_static_points, st, err, new_feature_points, features_to_retrack, print_frame_number, img)

        
        if (i == player_retrack_frame):

            if ((video_number == 3) and (i == 33)):
                new_player_points = np.append(new_player_points, [[[0,0]]], axis=0)
                st_2 = np.append(st_2, [[1]], axis=0)
            if ((video_number == 3) and (i == 69)):
                new_player_points = np.append(new_player_points, [[[0,0]]], axis=0)
                st_2 = np.append(st_2, [[1]], axis=0)    
            new_player_points, new_retrack_player_points, print_frame_number = retrack_players_2(new_player_points, player_to_retrack, new_retrack_player_points, img, print_frame_number)
            
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
        
        #if ((i == 256) and (video_number == 1)):
        #    new_static_points[3][0] = [342, 109]
            #static_corr_points[4] = [0.75, 0.0]
            #print static_corr_points
        #if ((i == 305) and (video_number == 1)):
        #    new_static_points[3][0] = [469, 163] 
            #static_corr_points[4] = [1.0, 0.0]
        if ((i == 132) and (video_number == 2)):
            new_static_points[1][0] = [85, 167]
        if ((i == 372) and (video_number == 2)):
            new_static_points[0][0] = [277, 239]  

        if ((i == 72) and (video_number == 4)):
            new_static_points[2][0] = [475, 276]    
        if ((i == 106) and (video_number == 4)):
            new_static_points[1][0] = [520, 268]
            new_static_points[3][0] = [458, 160]  
        if ((i == 156) and (video_number == 4)):
            new_static_points[3][0] = [154, 167]                  

        new_static_points, new_feature_points, features_to_retrack, print_frame_number, st = check_static_points_2(number_of_static, old_number_of_static, new_static_points, img, print_frame_number, new_feature_points, features_to_retrack, st)
        good_new = new_static_points[st==1]
        
        for j,(new,old) in enumerate(zip(good_new,good_old)):
            
            a,b = new.ravel()
            c,d = old.ravel()
            #print "HERE"
            #print a,b,c,d,j, (color[j].tolist())
            #cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(img,(a,b),5,color[j].tolist(),-1)
        
        good_new_2 = new_player_points[st_2==1]
        #good_old_2 = player_points[st_2==1]

        for i,new in enumerate(good_new_2):
            a,b = new.ravel()
            #c,d = old.ravel()
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
    file_name = "input/beachVolleyball3.mov"
    video_number = int(raw_input('Enter video number: '))
    read_video_file(file_name, video_number)


if __name__ == "__main__":
    main()