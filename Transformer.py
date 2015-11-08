import cv2
import numpy as np
import cv2.cv as cv
from functools import partial
from matplotlib import pyplot as plt


football_field = cv2.imread('football_field.png')
maxHeight, maxWidth = football_field.shape[:2]
mask = np.float32([[2881, 153], [5177, 139], [8398, 893], [26, 949]])
mask_scaled = np.float32([[698, 40], [1273, 40], [2094, 225], [2, 234]])


class Transformer():
	def __init__(self, is_scaled):
		if is_scaled:
			self.mask_points = mask_scaled
		else:
		    self.mask_points = mask

	def warpPerspective(self, M, point):
		x = int((M[0][0] * point[1] + M[0][1] * point[0] + M[0][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))
		y = int((M[1][0] * point[1] + M[1][1] * point[0] + M[1][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))

		return (x, y)

	def transform(self, points):
		field = np.array(football_field)
		(tl, tr, br, bl) = self.mask_points

		# width of new image
		# widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		# widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		# maxWidth = max(int(widthA), int(widthB))

		# height of new image
		# heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		# heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		# maxHeight = max(int(heightA), int(heightB))

		# dimension of new image
		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

		M = cv2.getPerspectiveTransform(self.mask_points, dst)

		for i in range(len(points)):
			warped = self.warpPerspective(M, points[i])
			cv2.circle(field, warped, 5, (0, 0, 255), 3)

		return field