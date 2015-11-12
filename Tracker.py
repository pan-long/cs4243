from cv2 import cv
from functools import partial

import cv2
import numpy as np


class Tracker(object):
    area_threshold = 10

    box_delta_y_up = 10
    box_delta_y_down = 3
    box_delta_x_left = 5
    box_delta_x_right = 5

    def __init__(self, background, is_scaled, init_point, color):
        self.init_point = init_point
        self.current_point = init_point

        self.velocity = [0, 0]
        if color[0].upper() == 'R':
            self.color = 'R'
        else:
            self.color = 'B'

        self.background = background

    def tracking(self, img):
        # Get a local tracking image
        sub_background = self.background[
                         self.current_point[0] - self.box_delta_y_up: self.current_point[0] + self.box_delta_y_down,
                         self.current_point[1] - self.box_delta_x_left: self.current_point[1] + self.box_delta_x_right]
        sub_img_orig = img[self.current_point[0] - self.box_delta_y_up: self.current_point[0] + self.box_delta_y_down,
                       self.current_point[1] - self.box_delta_x_left: self.current_point[1] + self.box_delta_x_right]

        background_ext = cv2.BackgroundSubtractorMOG2()
        background_ext.apply(sub_background)
        sub_img = background_ext.apply(sub_img_orig)

        # Remove noise and shadows
        _, img_thresholded = cv2.threshold(sub_img, 200, 255, cv.CV_THRESH_BINARY)
        cv2.imshow('sub', sub_img)

        contours, _ = cv2.findContours(img_thresholded, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)
        filtered_contours = [];
        for contour in contours:
            if cv2.contourArea(contour) >= self.area_threshold:
                filtered_contours.append(contour)

        down_right_bounds = np.array(map(partial(np.amax, axis=0), filtered_contours), np.int)
        up_left_bounds = np.array(map(partial(np.amin, axis=0), filtered_contours), np.int)
        centers = np.array(map(partial(np.mean, axis=0), zip(down_right_bounds, up_left_bounds)), np.int)

        tracking_points = []
        for f, c in zip(down_right_bounds, centers):
            tracking_points.append([f[0][1], c[0][0]])

        if len(tracking_points) == 0:
            # tracking_points = np.array(
            #     [(c[0][1], c[0][0]) for c in center], np.int)
            tracking_points.append([self.box_delta_y_up + self.velocity[0], self.box_delta_x_left + self.velocity[1]])
        else:
            tracking_points = np.array(tracking_points, np.int)

        if len(tracking_points) >= 1:
            point = self.__minPoint(tracking_points)
            point[0] = self.current_point[0] + point[0] - self.box_delta_y_up
            point[1] = self.current_point[1] + point[1] - self.box_delta_x_left
            self.velocity[0] = point[0] - self.current_point[0]
            self.velocity[1] = point[1] - self.current_point[1]
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
