import cv2
import cv2.cv as cv
import numpy as np

from Stitcher import Stitcher
from Tracking import tracking
from matplotlib import pyplot as plt
import lucas_canade_tracking

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']

config_scale = True

if config_scale:
    image_down_scale_factor = 4
    H_left_mid = np.array([[4.27846244e-01, -2.25290426e-01, 3.97710942e+02],
                           [1.88683929e-02, 9.48302837e-01, 1.40909737e+01],
                           [-1.22572919e-03, 2.10230845e-05, 1.00000000e+00]])
    H_mid_right = np.array([[-1.23516364e+00, -1.41395849e-01, 1.62674397e+03],
                            [-8.41283372e-02, -1.16214461e+00, 1.35519101e+02],
                            [-1.60078790e-03, -5.02481792e-05, 1.00000000e+00]])
    crop_image_rect = {'min_x': 200, 'max_x': 2300, 'min_y': 100, 'max_y': 350}
else:
    image_down_scale_factor = 1
    H_left_mid = np.array([[4.48586889e-01, -2.05408064e-01, 1.58586590e+03],
                           [3.11830929e-02, 9.58631698e-01, 5.31001193e+01],
                           [-3.02387986e-04, 1.19548345e-05, 1.00000000e+00]])
    H_mid_right = np.array([[-1.20474129e+00, -1.40161277e-01, 6.45227999e+03],
                            [-8.11346378e-02, -1.12980266e+00, 5.25837708e+02],
                            [-3.88404089e-04, -1.04585070e-05, 1.00000000e+00]])
    crop_image_rect = {'min_x': 800, 'max_x': 9200, 'min_y': 400, 'max_y': 1400}


def crop_img(img):
    """
    Crop the black area after warping images together.
    :param img: The image to be cropped
    :return: The cropped image.
    """
    # TODO: Detect the black area and crop smartly.
    return img[crop_image_rect['min_y']:crop_image_rect['max_y'], crop_image_rect['min_x']: crop_image_rect['max_x']]


def main():

    stitcher = Stitcher()
    if config_scale:
        background = cv2.imread('background_scaled.jpg')
    else:
        background = cv2.imread('background.jpg')

    background_ext = cv2.BackgroundSubtractorMOG2()
    background_ext.apply(background)

    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_width = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT))



    # SET UP tracker
    LucasCanadeTracker = lucas_canade_tracking.LucasCanadeTracking()

    # fourcc = cv2.cv.CV_FOURCC(*'X264')
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    # video_out = cv2.VideoWriter('output.avi',fourcc, 20.0, (8400,1000))
    video_out = cv2.VideoWriter('output_lucas_canade_tracking.mp4',fourcc, 24.0, (2100,250), True)
    print "video_out.isOpened():", video_out.isOpened()

    for fr in range(frame_count):
        status_left, frame_left = cap_left.read()
        status_mid, frame_mid = cap_mid.read()
        status_right, frame_right = cap_right.read()

        scaled_size = (frame_width / image_down_scale_factor, frame_height / image_down_scale_factor)
        frame_left = cv2.resize(frame_left, scaled_size)
        frame_mid = cv2.resize(frame_mid, scaled_size)
        frame_right = cv2.resize(frame_right, scaled_size)

        if status_left and status_mid and status_right:
            warped_left_mid = stitcher.stitch(frame_mid, frame_left, H_left_mid)
            warped_left_mid_right = stitcher.stitch(warped_left_mid, frame_right, H_mid_right)
            warped_left_mid_right_cropped = crop_img(warped_left_mid_right)
            background = background_ext.apply(warped_left_mid_right_cropped)
            background = tracking(background)
            # video_out.write(warped_left_mid_right_cropped)
            print "tracked_frame:", fr

            # plt.imshow(np.dstack((warped_left_mid_right_cropped[:,:,2], warped_left_mid_right_cropped[:,:,1], warped_left_mid_right_cropped[:,:,0])))
            # plt.show()

            # raise ValueError

            if(fr == 0):
                # LucasCanadeTracker.setOldGray(background)
                LucasCanadeTracker.setOldGray(cv2.cvtColor(warped_left_mid_right_cropped, cv2.COLOR_BGR2GRAY))
                LucasCanadeTracker.setMask(warped_left_mid_right_cropped)
                # p0 = [[[1158,123]],[[1128,95]], [[1138,71]], [[1218,65]],[[1204,60]], [[1219,51]], [[1200,47]]] # if use background for tracking
                p0 = [[[1158,123]],[[1128,95]], [[1139,74]], [[1214,68]],[[1204,62]], [[1217,53]], [[1199,47]]] # if use converted warped_left_mid_right_cropped
                
                # for i in range(0, len(p0)):
                #     print (p0[i][0][0],p0[i][0][1])
                #     cv2.circle(warped_left_mid_right_cropped,(p0[i][0][0],p0[i][0][1]),2,(0,0,255),-1)
                # # cv2.imshow("warped_left_mid_right_cropped_with_drawn_key_points", warped_left_mid_right_cropped)
                # cv2.imwrite("warped_left_mid_right_cropped_with_drawn_key_points.jpg", warped_left_mid_right_cropped)
                # # cv2.waitKey(0)
                # raise ValueError("purpose stop")
                LucasCanadeTracker.setp0(p0)
            else:
                # tracked_frame = LucasCanadeTracker.trackingOneFrame(background,warped_left_mid_right_cropped)
                tracked_frame = LucasCanadeTracker.trackingOneFrame(cv2.cvtColor(warped_left_mid_right_cropped, cv2.COLOR_BGR2GRAY),warped_left_mid_right_cropped)
                # print "tracked_frame:", fr
                # cv2.imshow('tracked_frame', tracked_frame)
                # cv2.waitKey(30)
                # video_out.write(125 * np.ones((8400,1000,3)))
                # video_out.write(warped_left_mid_right_cropped)
                video_out.write(tracked_frame)
            

    # cv2.waitKey(0)
    video_out.release()
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()
    


if __name__ == '__main__':
    main()
