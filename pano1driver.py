import rw
import feature_tracker as ft
import file_handler as fh
import stitcher as st

# This code currently
if __name__ == "__main__":
    video_number = 1
    file_name = 'input/beachVolleyball' + str(video_number) + '.mov'
    path = str(video_number) + '_points.txt'

    points = fh.read_from_file(path)

    frames, fps = rw.read(file_name)
    print "len(frames):", len(frames)

    try:
        statics = ft.read_video_file(frames, video_number, path)
    except:
        statics = fh.read_from_file(path)

    for i in range(len(statics)):
        # manually removed an erronous static point
        if i == 612:
            statics[i] = statics[i][:-1]
            break

    # need to get frames again 'cos previously drawn on it
    frames, fps = rw.read(file_name)

    st.create_full_court_stitch(frames, statics, '1_working/pano/')
