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
		# positions before warp
		self.marker_map = { 'C0': (71, 1153), \
							'R0': (80, 761), 'R1': (80, 1033), 'R2': (95, 1127), 'R3': (54, 1156), 'R4': (65, 1185), 'R5': (61, 1204), 'R6': (56, 1217), 'R7': (69, 1213), 'R8': (67, 1253), 'R9': (75, 1281), 'R10': (92, 1347), \
							'B0': (71, 1409), 'B1': (72, 2016), 'B2': (47, 1051), 'B3': (58, 1117), 'B4': (74, 1139), 'B5': (123, 1156), 'B6': (61, 1177), 'B7': (48, 1198), 'R8': (102, 1353)}

	def minPoint(self, point, points):
		min = maxHeight ** 2 + maxWidth ** 2
		minPt = (0, 0)
		for i in range(len(points)):
			dist = (int(points[i][0]) - point[0]) ** 2 + (int(points[i][1]) - point[1]) ** 2
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

		# dimension of new image
		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

		M = cv2.getPerspectiveTransform(self.mask_points, dst)

		# manually mark players' location in the first frame
		if len(self.marker_map) == 0:
			self.initMarker(points)
		else:
			self.updateMarker(points)

		for m, p in self.marker_map.iteritems():
			font = cv2.FONT_HERSHEY_SIMPLEX
			p = self.warpPerspective(M, p)
			if m[0] == "B":
				cv2.putText(field, m, p, font, 1, (255, 0, 0), 2, cv2.CV_AA)
			else:
				cv2.putText(field, m, p, font, 1, (0, 0, 255), 2, cv2.CV_AA)

		return field