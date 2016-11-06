"""
CS4243 Project - AY2016/2017

All seperate components will be executed here.

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
	# Directory where my (Barry) videos are.
	# Comment out existing ones and create your own list for testing,
	# and do not upload your videos to the repo.
    video_list = ['/Users/lyhbarry/Development/beachVolleyball/beachVolleyball1.mov']

    # Test function:
    # Time taken to retrieve frames from video
    # ~ 6s
    start_time = time.time()
    rw.read(video_list[0])
    print time.time() - start_time