import cv2
from cv2 import cv
import numpy as np
from functools import partial
from matplotlib import pyplot as plt

mask_scaled = ((2, 234), (2094, 225), (1273, 40), (698, 40))

def tracking(img):
    lower_left, lower_right, upper_right, upper_left = mask_scaled
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
    img_thresholded = cv2.erode(img, kernel, iterations = 1)
    img_thresholded = cv2.dilate(img_thresholded, kernel, iterations = 1)

    img_thresholded = cv2.dilate(img_thresholded, kernel, iterations = 1)
    img_thresholded = cv2.erode(img_thresholded, kernel, iterations = 1)

    img_orig = np.array(img_thresholded)

    contours, _ = cv2.findContours(img_thresholded, cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE)

    centers = map(partial(np.mean, axis=0), contours)

    for pt in centers:
        img_orig[pt[0][1], pt[0][0]] = 100

    return img_orig


################# For testing ######################
def main():
	img = cv2.imread('football_field.png')

	tracking(img)

if __name__ == '__main__':
	main()