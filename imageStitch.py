import cv2
import cv2.cv as cv
import numpy as np

def drawMatches(img1, kp1, img2, kp2, matches):
	"""
	My own implementation of cv2.drawMatches as OpenCV 2.4.9
	does not have this function available but it's supported in
	OpenCV 3.0.0

	This function takes in two images with their associated 
	keypoints, as well as a list of DMatch data structure (matches) 
	that contains which keypoints matched in which images.

	An image will be produced where a montage is shown with
	the first image followed by the second image beside it.

	Keypoints are delineated with circles, while lines are connected
	between matching keypoints.

	img1,img2 - Grayscale images
	kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
	detection algorithms
	matches - A list of matches of corresponding keypoints through any
	OpenCV keypoint matching algorithm
	"""

	# Create a new output image that concatenates the two images together
	# (a.k.a) a montage
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1,rows2]),cols1+cols2), dtype='uint8')

	# Place the first image to the left
	# out[:rows1,:cols1] = np.dstack([img1, img1, img1])
	out[:rows1,:cols1] = img1

	# Place the next image to the right of it
	out[:rows2,cols1:] = img2

	# For each pair of points we have between both images
	# draw circles, then connect a line between them
	for mat in matches:

		# Get the matching keypoints for each of the images
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx
	
		# x - columns
		# y - rows
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt
	
		# Draw a small circle at both co-ordinates
		# radius 4
		# colour blue
		# thickness = 1
		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
	
		# Draw a line in between the two points
		# thickness = 1
		# colour blue
		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


	# Show the image
	cv2.imshow('Matched Features', out)
	cv2.waitKey(0)
	cv2.destroyWindow('Matched Features')

	# Also return the image if you'd like a copy
	return out

# import cv2.cv.Sticher
def filter_matches(matches, ratio = 0.75):
	# print len(matches)
	filtered_matches = []
	for m in matches:
		# print m[0].distance, m[1].distance
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			filtered_matches.append(m[0])

	return filtered_matches

def startStictching():
	capLeft = cv2.VideoCapture("./videos/football_left.mp4")
	capMid = cv2.VideoCapture("./videos/football_mid.mp4")
	capRight = cv2.VideoCapture("./videos/football_right.mp4")
	

	fr_width_left = (capLeft.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	fr_height_left = (capLeft.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
	fr_per_sec_left = (capLeft.get(cv.CV_CAP_PROP_FPS))
	frame_count_left = (capLeft.get(cv.CV_CAP_PROP_FRAME_COUNT))
	frame_count_mid = (capMid.get(cv.CV_CAP_PROP_FRAME_COUNT))
	frame_count_right = (capRight.get(cv.CV_CAP_PROP_FRAME_COUNT))

	print "fr_per_sec_left:", fr_per_sec_left
	print "frame_count_left:", frame_count_left
	print "frame_count_mid:", frame_count_mid
	print "frame_count_right:", frame_count_right
	print "fr_height_left:", fr_height_left
	print "fr_width_left:", fr_width_left

	count = 0
	while (capLeft.isOpened()):
		print "isOpened", count
		count += 1
		ret, imgLeft = capLeft.read()
		cv2.imwrite("frameLeft.jpg", imgLeft)
		# cv2.imshow('frameLeft',imgLeft)
		# cv2.waitKey(1000)
		
		ret, imgMid = capMid.read()
		cv2.imwrite("frameMid.jpg", imgMid)
		# cv2.imshow('frameMid',imgMid)
		# cv2.waitKey(1000)
		
		ret, imgRight = capRight.read()
		cv2.imwrite("frameRight.jpg", imgRight)
		# cv2.imshow('frameRight',imgRight)
		# cv2.waitKey(1000)

		stictchFrames(imgLeft, imgMid, imgRight)
		break;

	return

def stictchFrames(frameLeft, frameMid, frameRight):
	# TODO: use frameMid's perspective, warp frameLeft and frameRight
	# return combined frame

	frameLeft = cv2.GaussianBlur(cv2.cvtColor(frameLeft,cv2.COLOR_BGR2GRAY), (5,5), 0)
	frameMid = cv2.GaussianBlur(cv2.cvtColor(frameMid,cv2.COLOR_BGR2GRAY), (5,5), 0)

	# Use the SIFT feature detector
	detector = cv2.SIFT()
	
	# Find 
	left_features, left_descs = detector.detectAndCompute(frameLeft, None)
	mid_features, mid_descs = detector.detectAndCompute(frameMid, None)
	right_features, right_descs = detector.detectAndCompute(frameRight, None)

	# Use Flann Matcher
	FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
	flann_params = dict(algorithm = FLANN_INDEX_KDTREE, 
	trees = 5)
	matcher = cv2.FlannBasedMatcher(flann_params, {})

	# Match Left and Mid
	matches_left_mid = matcher.knnMatch(left_descs, trainDescriptors=mid_descs, k=2)
	matches_right_mid = matcher.knnMatch(right_descs, trainDescriptors=mid_descs, k=2)

	print len(matches_left_mid)

	
	
	# min_distance = 100
	# max_distance = 0

	# for i in range(0, len(matches_left_mid)):
	# 	# print matches_left_mid[i] # a list of two match objects
	# 	if(matches_left_mid[i][0].distance < min_distance):
	# 		min_distance = matches_left_mid[i][0].distance
	# 	if(matches_left_mid[i][0].distance > max_distance):
	# 		max_distance = matches_left_mid[i][0].distance

	# print "min_distance:", min_distance
	# print "max_distance:", max_distance

	# matches_subset = []
	# for i in range(0, len(matches_left_mid)):
	# 	if(matches_left_mid[i][0].distance < 3 * min_distance):
	# 		matches_subset.append(matches_left_mid[i])

	# print "matches_subset:", len(matches_subset)

	matches_subset = filter_matches(matches_left_mid)
	matches_subset = sorted(matches_subset, key= lambda match: match.distance)
	print len(matches_subset)
	img3 = drawMatches(frameLeft,left_features,frameMid,mid_features,matches_subset[:10])
	cv2.imwrite("temp.jpg", img3)
	# raise ValueError
	
	kp1 = []
	kp2 = []

	for match in matches_subset:
		# print match.trainIdx
		kp1.append(mid_features[match.trainIdx])
		kp2.append(left_features[match.queryIdx])

	p1 = np.array([k.pt for k in kp1])
	p2 = np.array([k.pt for k in kp2])
	# print p1
	# print p2

	H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

	print H
	H_inv = np.linalg.inv(H)

	warped_frame_left = cv2.warpPerspective(frameLeft, H_inv, (frameLeft.shape[1], frameLeft.shape[0]))
	cv2.imwrite("warped_frame_left.jpg", warped_frame_left)
	cv2.imshow("warped_frame_left", warped_frame_left)
	cv2.waitKey(0)
	return

def main():
	# print cv2.__version__
	startStictching();
	# temp1 = np.matrix([[1,2,3],[1,2,3],[1,2,3]])
	# temp2 = np.matrix([[1,2,3],[1,2,3],[1,2,3]])
	# temp2 = np.array([1,2,3])
	# print temp1 * temp2
if __name__ == "__main__":
	main()