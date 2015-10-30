import cv2
import numpy as np
import cv2.cv as cv

from Stitcher import Stitcher
from ObjectsExt import ObjectsExt
from BackgroundExt import BackgroundExt

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']

config_scale = False

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
    H_left_mid = np.array([[4.17965460e-01, -2.08590564e-01, 1.58840805e+03],
                           [1.60253386e-02, 9.58337855e-01, 5.44518571e+01],
                           [-3.16345544e-04, 1.24986859e-05, 1.00000000e+00]])
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
    bg = cv2.imread('final_back_ground_2.jpg')
    ext = BackgroundExt()
    # ext.background_img = bg
    # ext.number_of_images = 715


    # raise ValueError

    stitcher = Stitcher()
    if config_scale:
        background = cv2.imread('background_scaled.jpg')
    else:
        background = cv2.imread('background.jpg')
    background_ext = cv2.BackgroundSubtractorMOG2()
    background_ext.apply(background)

    background_extractor = BackgroundExt()

    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_width = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT))

    print "frame_count:", frame_count


    # for fr in np.arange(0, frame_count,1):
    for fr in np.arange(3840, 4300,1): # for background ext, get rid of the goal keeper
        cap_left.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
        cap_mid.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
        cap_right.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
        print "count:", fr 
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
            # background = background_ext.apply(warped_left_mid_right_cropped)
            # background_extractor.add_image(warped_left_mid_right_cropped)
            ext.add_image(warped_left_mid_right_cropped)

            # cv2.imshow('temp background', cv2.convertScaleAbs(ext.background_img))
            # cv2.imshow('warped_left_mid_right_cropped',warped_left_mid_right_cropped)
            # cv2.waitKey(3)
            # if(fr % 1000 == 0):
                # cv2.imshow('temp background', cv2.convertScaleAbs(background_extractor.background_img))
                # cv2.imwrite("final_back_ground_full_fr.jpg", cv2.convertScaleAbs(background_extractor.background_img))
            # # cv2.imshow('Objects', background)
            # cv2.waitKey(3)
            

    # cv2.imwrite("final_back_ground_full_fr.jpg", cv2.convertScaleAbs(background_extractor.background_img))
    cv2.imwrite("final_back_ground_3.jpg", cv2.convertScaleAbs(ext.background_img))
    cv2.imshow("after mannually pick frame for average background:", cv2.convertScaleAbs(ext.background_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()


if __name__ == '__main__':
    main()
