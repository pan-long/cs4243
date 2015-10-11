import cv2
import cv2.cv as cv

from Stitcher import Stitcher

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']


def main():
    stitcher = Stitcher()

    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_count = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT))
    for fr in range(frame_count):
        status_left, frame_left = cap_left.read()
        status_mid, frame_mid = cap_mid.read()
        status_right, frame_right = cap_right.read()

        if status_left and status_mid and status_right:
            warped_left_mid = stitcher.stitch(frame_mid, frame_left)
            warped_left_mid_right = stitcher.stitch(warped_left_mid, frame_right)
            cv2.imshow('Warped all', warped_left_mid_right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()


if __name__ == '__main__':
    main()
