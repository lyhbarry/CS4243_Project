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
import video1driver

if __name__ == "__main__":
    # NOTE: The code below is not meant to be submitted. Just for a feel of the pano stitching.
    # The below code is something I have tried previously that kinda worked.
    # The new code I submitted didn't work as well because I couldn't get the feature points yet.
    # Note that the feature points obtained through the code below is via SURF using the spectator
    # plane as the flat plane to calculate the homography from.
    video1driver()
    rw.concatenate_output()
