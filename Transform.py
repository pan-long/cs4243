import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
	image = cv2.imread('football_field.png')

	pts = np.float32([[2881, 153], [5177, 139], [8398, 893], [26, 949]])
	warped = transform(image, pts)

	cv2.imwrite('output/transformed.png', warped)

	plt.figure()
	plt.imshow(image)
	plt.hold(True)
	plt.scatter([2881, 5177, 26, 8398], [153, 139, 949, 893], color='red')
	plt.show()


def transform(image, pts):
	(tl, tr, br, bl) = pts

	# width of new image
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	# height of new image
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# dimension of new image
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped

if __name__ == '__main__':
	main()