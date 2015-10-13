import numpy as np

import cv2
import cv2.cv as cv

from Stitcher import Stitcher

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']

image_down_scale_factor = 4

H_left_mid = np.array([[4.27846244e-01, -2.25290426e-01, 3.97710942e+02],
                       [1.88683929e-02, 9.48302837e-01, 1.40909737e+01],
                       [-1.22572919e-03, 2.10230845e-05, 1.00000000e+00]])
H_mid_right = np.array([[-1.26138793e+00, -1.43106109e-01, 1.66053396e+03],
                        [-9.31383114e-02, -1.16542313e+00, 1.46342165e+02],
                        [-1.60414665e-03, -4.87045990e-05, 1.00000000e+00]])


def main():
    stitcher = Stitcher()

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
            cv2.imshow('Warped all', warped_left_mid_right)
            cv2.waitKey(30)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()


if __name__ == '__main__':
    main()
