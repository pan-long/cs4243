from cv2 import cv
from functools import partial

import cv2
import numpy as np


class Tracker(object):
    area_threshold = 10
    height_threshold = 5
    width_threshold = 3
    color_threshold = 40
    distance_threshold = 3

    box_delta_y_up = 10
    box_delta_y_down = 3
    box_delta_x_left = 5
    box_delta_x_right = 5

    mask_scaled = ((2, 234), (2094, 225), (1273, 40), (698, 40))
    mask = ((26, 949), (8398, 893), (5177, 139), (2881, 153))

    def __init__(self, background, is_scaled, init_point, color):
        self.init_point = init_point
        self.current_point = init_point
        self.center = [self.box_delta_y_up, self.box_delta_x_left]

        self.velocity = [0, 0]
        if color[0].upper() == 'R':
            self.color = 'R'
        else:
            self.color = 'B'

        if is_scaled:
            self.mask_points = self.mask_scaled
        else:
            self.mask_points = self.mask

        self.background = background

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
                # elif cv2.pointPolygonTest(contour, (self.box_delta_y_up + self.velocity[0] - self.current_point[0] + self.center[0],
                #                                     self.box_delta_x_left + self.velocity[1] - self.current_point[1] + self.center[1]), False):
                #     filtered_contours.append(contour)
        # cv2.drawContours(img_thresholded, filtered_contours, -1, (255, 255, 255), -1)

        # print(len(filtered_contours))

        feet = np.array(map(partial(np.amax, axis=0), filtered_contours), np.int)
        center = np.array(map(partial(np.mean, axis=0), filtered_contours), np.int)

        # print(center)
        for c in center:
            if np.hypot(c[0][0] - self.center[0], c[0][1] - self.center[1]) < self.distance_threshold ** 2:
                break
        else:
            center = np.array(map(partial(np.mean, axis=0), contours), np.int)
            for c, contour in zip(center, contours):
                # print(np.hypot(c[0][0] - self.center[0], c[0][1] - self.center[1]))
                if np.hypot(c[0][0] - self.center[0], c[0][1] - self.center[1]) < self.distance_threshold ** 2:
                    filtered_contours.append(contour)

            feet = np.array(map(partial(np.amax, axis=0), filtered_contours), np.int)
            center = np.array(map(partial(np.mean, axis=0), filtered_contours), np.int)

        tracking_points = []
        for f, c in zip(feet, center):
            if c[0][1] >= self.box_delta_x_left + self.box_delta_x_right:
                c[0][1] = self.box_delta_x_left + self.box_delta_x_right - 1
            if f[0][1] >= self.box_delta_y_up + self.box_delta_y_down:
                f[0][1] = self.box_delta_y_up + self.box_delta_y_down - 1

            i = 0
            for j in range(self.box_delta_y_up):
                if c[0][0] - j >= 0:
                    pixel = sub_img_orig[c[0][0] - j, c[0][1]]
                    pixel_hsv =  cv2.cvtColor(np.array([[pixel]]), cv2.COLOR_BGR2HSV)

                    if pixel_hsv[0][0][0] < 20 or \
                            (pixel_hsv[0][0][0] > 110 and pixel_hsv[0][0][0] < 130):
                        i = j
                        break
            # print i

            if self.color == 'R':
                expected_color = sub_img_orig[c[0][0] - i, c[0][1], 2]
                compared_color = sub_img_orig[c[0][0] - i, c[0][1], 0]
            else:
                expected_color = sub_img_orig[c[0][0] - i, c[0][1], 0]
                compared_color = sub_img_orig[c[0][0] - i, c[0][1], 2]

            if expected_color < compared_color and len(center) == 1:  # we are blocked by someone else
                print 'blocked', expected_color, compared_color
                tracking_points.append(
                    [self.box_delta_y_up + self.velocity[0], self.box_delta_x_left + self.velocity[1]])
                # self.box_delta_y_up = self.box_delta_y_up_large
            elif expected_color > compared_color:
                tracking_points.append([f[0][1], c[0][0]])
                # self.box_delta_y_up = self.box_delta_y_up_small
        if len(tracking_points) == 0:
            # tracking_points = np.array(
            #     [(c[0][1], c[0][0]) for c in center], np.int)
            tracking_points.append([self.box_delta_y_up + self.velocity[0], self.box_delta_x_left + self.velocity[1]])
        else:
            tracking_points = np.array(tracking_points, np.int)

        # tracking_points = np.array(
        #     [(f[0][1], c[0][0]) for f, c in zip(feet, center)], np.int)

        # print(tracking_points)
        if len(tracking_points) >= 1:
            point = self.__minPoint(tracking_points)
            point[0] = self.current_point[0] + point[0] - self.box_delta_y_up
            point[1] = self.current_point[1] + point[1] - self.box_delta_x_left
            self.velocity[0] = point[0] - self.current_point[0]
            self.velocity[1] = point[1] - self.current_point[1]
            self.current_point = point
            for con, cen in zip(contours, center):
                if cv2.pointPolygonTest(con, tuple(point), False):
                    self.center[0] = cen[0][0]
                    self.center[1] = cen[0][1]
                    break

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
