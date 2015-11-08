from cv2 import cv
from functools import partial

import cv2
import numpy as np


class Tracker(object):
    mask_scaled = ((2, 234), (2094, 225), (1273, 40), (698, 40))
    mask = ((26, 949), (8398, 893), (5177, 139), (2881, 153))

    def __init__(self, is_scaled):
        if is_scaled:
            self.mask_points = self.mask_scaled
        else:
            self.mask_points = self.mask

    def tracking(self, img):
        lower_left, lower_right, upper_right, upper_left = self.mask_points
        height, width = img.shape[:2]

        # Filter out the pixels outside of the field.
        for i in range(height):
            for j in range(width):
                if i < upper_right[1] or i > lower_right[1]:
                    img[i, j] = 0
                else:
                    x_left_boundary = upper_left[0] - float(i - upper_left[1]) / (lower_left[1] - upper_left[1]) * \
                                                      (upper_left[0] - lower_left[0])
                    x_right_boundary = upper_right[0] + float(i - upper_right[1]) / (lower_right[1] - upper_right[1]) * \
                                                        (lower_right[0] - upper_right[0])
                    if j < x_left_boundary or j > x_right_boundary:
                        img[i, j] = 0

        kernel = np.ones((3, 3), np.uint8)
        img_thresholded = cv2.erode(img, kernel, iterations=1)
        img_thresholded = cv2.dilate(img_thresholded, kernel, iterations=1)

        img_thresholded = cv2.dilate(img_thresholded, kernel, iterations=1)
        img_thresholded = cv2.erode(img_thresholded, kernel, iterations=1)

        contours, _ = cv2.findContours(img_thresholded, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)

        centers = map(partial(np.mean, axis=0), contours)
        feet_points = map(partial(np.amax, axis=0), contours)

        img_tracking_points = np.zeros((height, width), np.uint8)

        for center, feet_point in zip(centers, feet_points):
            img_tracking_points[center[0][1], feet_point[0][0]] = 255

        return img_tracking_points
