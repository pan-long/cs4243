import cv2
import cv2.cv as cv
import numpy as np

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']

def main():
    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_count = cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT)
    for fr in range(frame_count):


if __name__ == '__main__':
    main()
