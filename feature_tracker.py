import cv2
import cv2.cv as cv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def read_video_file(file_name, video_number):
    cap = cv2.VideoCapture(file_name)
    frame_count = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
    
    # read the first frame of video and convert to grayscale
    _,old_frame = cap.read()
    gray_old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    color = np.random.randint(0,255,(100,3))

    #parameters to detect and track feature points
    static_lk_params = dict(winSize  = (10,10),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    player_lk_params = dict(winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
    mask = np.zeros_like(old_frame)
    
    # this function finds A LOT of features 
    p0 = cv2.goodFeaturesToTrack(gray_old_frame, mask = None, **feature_params)
    print p0

    # images to find coordinates of feature points better
    plt.figure()
    plt.imshow(gray_old_frame, cmap="gray")
    plt.savefig('first_gray_frame.jpg')
    plt.close()
    plt.imshow(old_frame)
    plt.savefig('first_color_frame.jpg')
    plt.close()

    first_frame_features = p0
    for i,(first_frame_feature) in enumerate(zip(first_frame_features)):
        #print first_frame_feature[0][0]
        x_coor = first_frame_feature[0][0][0]
        y_coor = first_frame_feature[0][0][1]

        if 80.0 < x_coor < 100.0:
            print i
            print x_coor
            print y_coor
            cv2.circle(gray_old_frame,(x_coor,y_coor),5,color[i].tolist(),-1)
    cv2.imwrite('first_gray_frame_features.jpg', gray_old_frame)
    
    if video_number == 1:
        # 1st video static points
        
        static_points = np.zeros((2,1,2))
        static_points[0][0] = p0[-1] #left pole
        static_points[1][0] = p0[1]  #right pole
        static_points = static_points.astype(np.float32)

        # 1st video player points
        player_points = np.zeros((3,1,2))
        player_points[0][0] = p0[27]
        player_points[1][0] = p0[12]
        player_points[2][0] = p0[40]
        player_points = player_points.astype(np.float32)
        print static_points
        
    elif video_number == 2:    
        #2nd video static points
        static_points = np.zeros((2,1,2))
        static_points[0][0] = p0[76] # left pole
        static_points[1][0] = p0[22] # flag guy at the back standing on the right
        static_points = static_points.astype(np.float32)

        #2nd video player points
        player_points = np.zeros((4,1,2))
        player_points[0][0] = p0[60] # this side / left side
        player_points[1][0] = p0[27] # other side /right side
        player_points[2][0] = p0[29] # other side / right side
        player_points[3][0] = p0[4] # other side / left side
        player_points = player_points.astype(np.float32)
        
    elif video_number == 3:    
        # 3rd video static points
        static_points = np.zeros((2,1,2))
        static_points[0][0] = p0[30] #nearer pole
        static_points[1][0] = p0[74] #guy sitting across, his shoe
        static_points = static_points.astype(np.float32)

        # 3rd video player points
        player_points = np.zeros((3,1,2))
        player_points[0][0] = p0[-12] # guy bending over
        player_points[1][0] = p0[9] # guy serving
        player_points[2][0] = p0[31] # guy serving
        player_points = player_points.astype(np.float32)

    elif video_number == 5:
        # 5th video static points
        
        static_points = np.zeros((2,1,2))
        static_points[0][0] = p0[-8]  #corner of the podium of the empire
        static_points[1][0] = p0[12]  #bottom left coorner of the net
        static_points = static_points.astype(np.float32)

        player_points = np.zeros((6,1,2))
        player_points[0][0] = p0[35] # other side/ right guy      needs work
        player_points[1][0] = p0[24] # other side/ left guy       good
        player_points[2][0] = p0[33] # other side / left guy      good
        player_points[3][0] = p0[42] # this side / further away   needs work
        player_points[4][0] = p0[19] # this side /closer guy      good
        player_points[5][0] = p0[21] # this side / closer guy     good
        player_points = player_points.astype(np.float32)
        
    elif video_number == 6:
        # 6th video static points
        
        static_points = np.zeros((2,1,2))
        static_points[0][0] = p0[-8] #umpire pole
        static_points[1][0] = p0[28] #top left corner of five sign, but might move down due to ball
        static_points = static_points.astype(np.float32)
        
    elif video_number == 7:
        # 7th video static points
        
        static_points = np.zeros((3,1,2))
        static_points[0][0] = p0[-3] # umpire
        static_points[1][0] = p0[13] # five sign top right corner
        static_points[2][0] = p0[10] # pole nearer to the camera
        static_points = static_points.astype(np.float32)

        player_points = np.zeros((1,1,2))
        player_points[0][0] = p0[21] # right side, bending
        player_points = player_points.astype(np.float32)
    
    for i in range(1,frame_count):
        _,img = cap.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # use this line when you want to see all the movement of all the features
        #static_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, p0, None, **static_lk_params)

        # use this line when you want to see the movement of specific points
        
        new_static_points, st, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, static_points, None, **static_lk_params)
        new_player_points, st_2, err = cv2.calcOpticalFlowPyrLK(gray_old_frame, gray_img, player_points, None, **player_lk_params)
        
        # use this section when looking at ALL feature points 
        """
        good_new = static_points[st==1]
        good_old = p0[st==1]
        """
        # use this section when looking for specific points
        
        good_new = new_static_points[st==1]
        good_old = static_points[st==1]
        
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(img,(a,b),5,color[i].tolist(),-1)
        
        good_new_2 = new_player_points[st_2==1]
        good_old_2 = player_points[st_2==1]

        for i,(new,old) in enumerate(zip(good_new_2,good_old_2)):
            a,b = new.ravel()
            c,d = old.ravel()
            cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(img,(a,b),5,color[i].tolist(),-1)
        
        image = cv2.add(img,mask)

        cv2.imshow('frame',image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        gray_old_frame = gray_img.copy()
        # use this line for ALL feature points
        #p0 = good_new.reshape(-1,1,2)

        # use this line for specific points
        static_points = good_new.reshape(-1,1,2)
        player_points = good_new_2.reshape(-1,1,2)

def main():
    file_name = "/Users/benjsoon/Documents/CS4243 - Computer Vision and Pattern Recognition/CS4243_Project_Instructions_and_Data/beachVolleyball/beachVolleyball7.mov"
    video_number = int(raw_input('Enter video number: '))
    read_video_file(file_name, video_number)


if __name__ == "__main__":
    main()