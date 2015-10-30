import cv2
import numpy as np

lowH  = 75
highH = 130

lowS  = 0
highS = 255

lowV  = 0
highV = 255

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