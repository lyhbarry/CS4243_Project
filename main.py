"""
CS4243 Project - AY2016/2017

All separate components will be executed here.

List of all the components:
1. Top-down projection --> (Rodson)
2. Feature detection and tracking --> (Barry, Benjamin)
3. Render display of movements --> (Anyone)
4. Statistics --> (Lang Fan)
5. Outputs concatenation --> (Jun Wei)
"""
import rw
import time

if __name__ == "__main__":
    # # Directory where my (Barry) videos are.
    # # Comment out existing ones and create your own list for testing,
    # # and do not upload your videos to the repo.
    video_list = ['/Users/lyhbarry/Development/beachVolleyball/beachVolleyball1.mov']
    #
    # # Test function:
    # # Time taken to retrieve frames from video
    # # ~ 6s
    # start_time = time.time()
    # rw.read(video_list[0])
    # print time.time() - start_time

    # NOTE: The code below is not meant to be submitted. Just for a feel of the pano stitching.
    # The below code is something I have tried previously that kinda worked.
    # The new code I submitted didn't work as well because I couldn't get the feature points yet.
    # Note that the feature points obtained through the code below is via SURF using the spectator
    #   plane as the flat plane to calculate the homography from.
    from archival_stitcher import Stitcher

    start_time = time.time()

    # video_6 = 'input/beachVolleyball6.mov'
    # frames6, fps6 = rw.read(video_6)
    # stitcher6 = Stitcher()
    # stitcher6.do_main(frames6, fps6, [None, None, None, frames6[0].shape[0] / 2], None, 6)

    rw.output_generator(video_list[0])

    # print "time taken to generate pano files:", time.time() - start_time
    print "time taken to generate output:", time.time() - start_time
