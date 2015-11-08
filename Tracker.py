from cv2 import cv
from functools import partial

import cv2
import numpy as np


class Tracker(object):
    area_threshold = 15
    height_threshold = 5
    width_threshold = 3

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

        # Remove noise and shadows
        _, img_thresholded = cv2.threshold(img, 200, 255, cv.CV_THRESH_BINARY)

        contours, _ = cv2.findContours(img_thresholded, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
        img_thresholded = np.zeros((height, width, 3), np.uint8)
        filtered_contours = [];
        for contour in contours:
            if np.abs(np.amin(contour, axis=0)[0][1] - upper_left[1]) <= 1. / 3 * (lower_left[1] - upper_left[1]) :
                area_threshold = self.area_threshold / 2.5
            else:
                area_threshold = self.area_threshold
            if cv2.contourArea(contour) >= area_threshold and \
                    np.amax(contour, axis=0)[0][0] - np.amin(contour, axis=0)[0][0] >= self.width_threshold and \
                    np.amax(contour, axis=0)[0][1] - np.amin(contour, axis=0)[0][1] >= self.height_threshold:
                filtered_contours.append(contour)
        cv2.drawContours(img_thresholded, filtered_contours, -1, (255, 255, 255), -1)

        print(len(filtered_contours))
        centers = map(partial(np.amax, axis=0), filtered_contours)
        feet_points = map(partial(np.mean, axis=0), filtered_contours)

        tracking_points = np.array([(center[0][1], feet_point[0][0]) for center, feet_point in zip(centers, feet_points)], np.uint)

        return tracking_points
