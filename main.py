import numpy as np

import cv2
import cv2.cv as cv

from BackgroundExt import BackgroundExt
from ObjectsExt import ObjectsExt

from Stitcher import Stitcher

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']

image_down_scale_factor = 4

H_left_mid = np.array([[4.27846244e-01, -2.25290426e-01, 3.97710942e+02],
                       [1.88683929e-02, 9.48302837e-01, 1.40909737e+01],
                       [-1.22572919e-03, 2.10230845e-05, 1.00000000e+00]])
H_mid_right = np.array([[-1.23516364e+00, -1.41395849e-01, 1.62674397e+03],
                        [-8.41283372e-02, -1.16214461e+00, 1.35519101e+02],
                        [-1.60078790e-03, -5.02481792e-05, 1.00000000e+00]])

crop_image_rect = {'min_x': 200, 'max_x': 2300, 'min_y': 100, 'max_y': 350}


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
    background = cv2.imread('background.jpg')
    objects_ext = ObjectsExt(background)

    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_width = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT))

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
            objects_img = objects_ext.extract_objects(warped_left_mid_right_cropped)
            cv2.imshow('Objects', objects_img)
            cv2.waitKey(30)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()


if __name__ == '__main__':
    main()
