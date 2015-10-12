import numpy as np

import cv2
import cv2.cv as cv


class BackgroundExt(object):
    def extract_background(self, video_capture):
        frame_width = int(video_capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        frame_count = int(video_capture.get(cv.CV_CAP_PROP_FRAME_COUNT))

        _, img = video_capture.read()
        result_img = np.float32(img)

        for fc in range(frame_count):
            _, frame = video_capture.read()
            result_img = fc / (fc + 1.) * result_img + 1. / (fc + 1.) * frame

        normImg = cv2.convertScaleAbs(result_img)
        return normImg
