import cv2
import cv2.cv as cv
import numpy as np

from Stitcher import Stitcher
from Tracker import Tracker
from Transformer import Transformer
from matplotlib import pyplot as plt
import meanShift
import savePoint
import csv
import sys

sys.setrecursionlimit(1000000)

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

# def testDictionary(test_dict):
#     test_dict["{r}_{c}".format(r = 2, c = 2)] = 1
#     return

def main():
    ### test ###
    # test_dict = {}
    # print not "{r}_{c}".format(r = 1, c = 2) in test_dict
    # test_dict["{r}_{c}".format(r = 1, c = 2)] = 1
    # print "test_dict before pass in:", test_dict
    # testDictionary(test_dict)
    # print "test_dict after pass in:", test_dict
    # temp = np.ones ((10,10))
    # temp_cutted = np.copy(temp[3:6, 3:6])
    # for i in range(0, temp_cutted.shape[0]):
    #     for j in range(0, temp_cutted.shape[1]):
    #         temp_cutted[i][j] = 100
    # print temp
    # # temp = "{r}_{c}".format(r = 1, c = 2)
    # # chars = temp.split("_")
    # # r = int(chars[0])
    # # c = int(chars[1])
    # # print chars
    # # print r, " ", c
    # raise ValueError("temp stop")
    stitcher = Stitcher()
    # if config_scale:
    #     background = cv2.imread('background_scaled.jpg')
    # else:
    #     background = cv2.imread('background.jpg')
    # background_ext = cv2.BackgroundSubtractorMOG2()
    # background_ext.apply(background)

    transformer = Transformer(config_scale)

    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_width = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_count_mid = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT))
    frame_count_left = int(cap_left.get(cv.CV_CAP_PROP_FRAME_COUNT))
    frame_count_right = int(cap_right.get(cv.CV_CAP_PROP_FRAME_COUNT))
    frame_count = np.min([frame_count_left, frame_count_mid, frame_count_right])

    player_name = 'REFEREE'
    # clear previous file
    player_filename = "player_{id}_points.csv".format(id = player_name)
    with open(player_filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',)

    # point = [123, 1156]
    # tracker = Tracker(config_scale, point)

    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    video_out = cv2.VideoWriter('out_put_mean_shift_RFEREE.mp4',fourcc, 24.0, (2100,250), True)

    mean_shift_tracker = meanShift.meanShiftTracker()
    fr = 0
    # for fr in range(frame_count):
    while(fr < frame_count):
        print(fr)
        
        status_left, frame_left = cap_left.read()
        status_mid, frame_mid = cap_mid.read()
        status_right, frame_right = cap_right.read()

        if status_left and status_mid and status_right:
            scaled_size = (frame_width / image_down_scale_factor, frame_height / image_down_scale_factor)
            frame_left = cv2.resize(frame_left, scaled_size)
            frame_mid = cv2.resize(frame_mid, scaled_size)
            frame_right = cv2.resize(frame_right, scaled_size)
            # frame_mid = cv2.convertScaleAbs(frame_mid, alpha=0.92)

            warped_left_mid = stitcher.stitch(frame_mid, frame_left, H_left_mid)
            warped_left_mid_right = stitcher.stitch(warped_left_mid, frame_right, H_mid_right)
            warped_left_mid_right_cropped = crop_img(warped_left_mid_right)

            # plt.imshow(np.dstack((warped_left_mid_right_cropped[:,:,2], warped_left_mid_right_cropped[:,:,1], warped_left_mid_right_cropped[:,:,0])))
            # plt.show()
            # break;
            # background = background_ext.apply(warped_left_mid_right_cropped)

            # if(fr == 2000):
                # plt.imshow(np.dstack((warped_left_mid_right_cropped[:,:,2], warped_left_mid_right_cropped[:,:,1], warped_left_mid_right_cropped[:,:,0])))
                # plt.show()
                # break

            if fr == 0:
                mean_shift_tracker.initFromFirstFrame(warped_left_mid_right_cropped)
                # set fr to 800 after initialize to speed up test
                # fr = 2000
                # mean_shift_tracker.setTrack_window((984,76,5,11)) # set frame number for fr 800
                # mean_shift_tracker.setTrack_window((940,81,5,11)) # set frame number for fr 1200
                # mean_shift_tracker.setTrack_window((927,64,5,11)) # set frame number for fr 1500
                # mean_shift_tracker.setTrack_window((702,60,5,11)) # set frame number for fr 7100
                # mean_shift_tracker.setTrack_window((995,59,5,11)) # set frame number for fr 2000 for REFEREE
                # cap_left.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
                # cap_mid.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
                # cap_right.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
                fr += 1
            else:
                mean_shift_frame, mean_shift_point = mean_shift_tracker.trackOneFrame(warped_left_mid_right_cropped)
                savePoint.saveOnePlayerPoint(player_name, mean_shift_point, fr)
                cv2.imshow('football', mean_shift_frame)
                cv2.waitKey(1)
                video_out.write(mean_shift_frame)
                fr += 1
            
        else:
            fr += 1
            # point = tracker.tracking(background)
            
            # for pt in points:
            # global prev
            # if len(prev) == 0:
            #     prev = points[4]
            # pt = minPoint(points)
            # cv2.circle(warped_left_mid_right_cropped, (point[1], point[0]), 3, (0, 0, 255), -1)
            # cv2.imshow('Objects', warped_left_mid_right_cropped)
            # cv2.waitKey(1)

            # background = transformer.transform(points)
            # plt.imshow(warped_left_mid_right_cropped)
            # plt.show()
            # cv2.imshow('Objects', background)
            # cv2.waitKey(30)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()


if __name__ == '__main__':
    main()
