import cv2
import numpy as np

def tracking(img):
	kernel = np.ones((3, 3), np.uint8)
	img_thresholded = cv2.erode(img, kernel, iterations = 1)
	img_thresholded = cv2.dilate(img_thresholded, kernel, iterations = 1)

	img_thresholded = cv2.dilate(img_thresholded, kernel, iterations = 1)
	img_thresholded = cv2.erode(img_thresholded, kernel, iterations = 1)

	return img_thresholded


################# For testing ######################
def main():
	img = cv2.imread('football_field.png')

	tracking(img)

if __name__ == '__main__':
	main()