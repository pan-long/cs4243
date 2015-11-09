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
		self.marker_map = {}
		self.prev_points = []

		if is_scaled:
			self.mask_points = mask_scaled
		else:
		    self.mask_points = mask

	def warpPerspective(self, M, point):
		x = int((M[0][0] * point[1] + M[0][1] * point[0] + M[0][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))
		y = int((M[1][0] * point[1] + M[1][1] * point[0] + M[1][2]) / (M[2][0] * point[1] + M[2][1] * point[0] + M[2][2]))

		return (x, y)

	def initMarker(self, points):
		# for mannually mark players
		# for i in range(len(points)):
		# 	print points[i]

		self.marker_map = {'A1': (1825, 1448), 'A2': (2463, 1283), \
							'B1': (1852, 1198), 'B2': (1606, 987), 'B0': (687, 983), 'B4': (2488, 952)}

	def minPoint(self, point, points):
		min = maxHeight ** 2 + maxWidth ** 2

		for i in range(len(points)):
			dist = (points[i][0] - point[0]) ** 2 + (points[i][1] - point[1]) ** 2
			if dist < min:
				min = dist
				minPt = points[i]

		return minPt

	def updateMarker(self, points):
		for m, p in self.marker_map.iteritems():
			pt = self.minPoint(p, points)
			self.marker_map[m] = pt

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

		warped = []
		for i in range(len(points)):
			warped.append(self.warpPerspective(M, points[i]))
			# print points[i], warped[i]

		# manually mark players' location in the first frame
		if len(self.marker_map) == 0:
			self.initMarker(warped)
		else:
			self.updateMarker(warped)

		for m, p in self.marker_map.iteritems():
			font = cv2.FONT_HERSHEY_SIMPLEX
			if m[0] == "A":
				cv2.putText(field, m, p, font, 1, (255, 0, 0), 2, cv2.CV_AA)
			else:
				cv2.putText(field, m, p, font, 1, (0, 0, 255), 2, cv2.CV_AA)

		return field