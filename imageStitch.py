import cv2
import cv2.cv as cv
import numpy as np
# import cv2.cv.Sticher

def startStictching():
	capLeft = cv2.VideoCapture("videos/football_left.mp4")
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

	while (capLeft.isOpened()):
		print "isOpened"
		ret, img = capLeft.read()
		cv2.imshow('frame',img)
		cv2.waitKey(1000)

	return

def stictchFrames(frameLeft, frameMid, frameRight):
	# TODO: use frameMid's perspective, warp frameLeft and frameRight
	# return combined frame
	return

def main():
	# print cv2.__version__
	startStictching();

if __name__ == "__main__":
	main()