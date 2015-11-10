from cv2 import cv
from functools import partial

import cv2
import numpy as np


class Tracker(object):
    area_threshold = 15
    height_threshold = 5
    width_threshold = 3

    box_delta_y_up = 20
    box_delta_y_down = 5
    box_delta_x_left = 10
    box_delta_x_right = 10

    mask_scaled = ((2, 234), (2094, 225), (1273, 40), (698, 40))
    mask = ((26, 949), (8398, 893), (5177, 139), (2881, 153))

    def __init__(self, is_scaled, init_point):
        self.init_point = init_point
        self.current_point = init_point
        if is_scaled:
            self.mask_points = self.mask_scaled
        else:
            self.mask_points = self.mask

    def tracking(self, img):
        # lower_left, lower_right, upper_right, upper_left = self.mask_points
        # height, width = img.shape[:2]

        # Filter out the pixels outside of the field.
        # for i in range(height):
        #     for j in range(width):
        #         if i < upper_right[1] or i > lower_right[1]:
        #             img[i, j] = 0
        #         else:
        #             x_left_boundary = upper_left[0] - float(i - upper_left[1]) / (lower_left[1] - upper_left[1]) * \
        #                                               (upper_left[0] - lower_left[0])
        #             x_right_boundary = upper_right[0] + float(i - upper_right[1]) / (lower_right[1] - upper_right[1]) * \
        #                                                 (lower_right[0] - upper_right[0])
        #             if j < x_left_boundary or j > x_right_boundary:
        #                 img[i, j] = 0

        # Get a local tracking image
        sub_img = img[self.current_point[0] - self.box_delta_y_up: self.current_point[0] + self.box_delta_y_down,
                  self.current_point[1] - self.box_delta_x_left: self.current_point[1] + self.box_delta_x_right]

        # Remove noise and shadows
        _, img_thresholded = cv2.threshold(sub_img, 200, 255, cv.CV_THRESH_BINARY)

        contours, _ = cv2.findContours(img_thresholded, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
        # img_thresholded = np.zeros((height, width, 3), np.uint8)
        filtered_contours = [];
        for contour in contours:
            # if np.abs(np.amin(contour, axis=0)[0][1] - upper_left[1]) <= 1. / 3 * (lower_left[1] - upper_left[1]):
            #     area_threshold = self.area_threshold / 2.5
            # else:
            #     area_threshold = self.area_threshold
            if cv2.contourArea(contour) >= self.area_threshold and \
                                    np.amax(contour, axis=0)[0][0] - np.amin(contour, axis=0)[0][
                                0] >= self.width_threshold and \
                                    np.amax(contour, axis=0)[0][1] - np.amin(contour, axis=0)[0][
                                1] >= self.height_threshold:
                filtered_contours.append(contour)
        # cv2.drawContours(img_thresholded, filtered_contours, -1, (255, 255, 255), -1)

        # print(len(filtered_contours))

        centers = map(partial(np.amax, axis=0), filtered_contours)
        feet_points = map(partial(np.mean, axis=0), filtered_contours)

        tracking_points = np.array(
            [(center[0][1], feet_point[0][0]) for center, feet_point in zip(centers, feet_points)], np.uint)

        if len(tracking_points) >= 1:
            point = self.__minPoint(tracking_points)
            point[0] = self.current_point[0] + point[0] - self.box_delta_y_up
            point[1] = self.current_point[1] + point[1] - self.box_delta_x_left
            self.current_point = point

        return self.current_point

    def __minPoint(self, points):
        min = self.box_delta_y_up ** 2 + self.box_delta_x_left ** 2
        minPt = []
        for i in range(len(points)):
            dist = (int(points[i][0]) - self.box_delta_y_up) ** 2 + (int(points[i][1]) - self.box_delta_x_left) ** 2
            if dist < min:
                min = dist
                minPt = points[i]

        return minPt
