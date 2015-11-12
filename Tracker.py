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

    def __init__(self, background, is_scaled, points):
        self.points = points
        self.velocities = [[0, 0] for i in range(len(points))]
        self.background = background

    def tracking(self, img):
        for i in range(len(self.points)):
            self.points[i] = self.trackingPoint(img, i)

        ## TODO: check 2 points mapped to same contour
        return  self.points

    def trackingPoint(self, img, idx):
        current_point = self.points[idx]
        current_velocity = self.velocities[idx]

        # Get a local tracking image
        sub_background = self.background[
                         current_point[0] - self.box_delta_y_up: current_point[0] + self.box_delta_y_down,
                         current_point[1] - self.box_delta_x_left: current_point[1] + self.box_delta_x_right]
        sub_img_orig = img[current_point[0] - self.box_delta_y_up: current_point[0] + self.box_delta_y_down,
                        current_point[1] - self.box_delta_x_left: current_point[1] + self.box_delta_x_right]

        background_ext = cv2.BackgroundSubtractorMOG2()
        background_ext.apply(sub_background)
        sub_img = background_ext.apply(sub_img_orig)

        # Remove noise and shadows
        _, img_thresholded = cv2.threshold(sub_img, 200, 255, cv.CV_THRESH_BINARY)

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
            tracking_points.append([self.box_delta_y_up + self.velocities[idx][0], self.box_delta_x_left + self.velocities[idx][1]])
        else:
            tracking_points = np.array(tracking_points, np.int)

        if len(tracking_points) >= 1:
            point = self.__minPoint(tracking_points)
            point[0] = current_point[0] + point[0] - self.box_delta_y_up
            point[1] = current_point[1] + point[1] - self.box_delta_x_left
            self.velocities[idx][0] = point[0] - current_point[0]
            self.velocities[idx][1] = point[1] - current_point[1]

        return point

    def __minPoint(self, points):
        min = self.box_delta_y_up ** 2 + self.box_delta_x_left ** 2
        minPt = []
        for i in range(len(points)):
            dist = (int(points[i][0]) - self.box_delta_y_up) ** 2 + (int(points[i][1]) - self.box_delta_x_left) ** 2
            if dist < min:
                min = dist
                minPt = points[i]

        return minPt
