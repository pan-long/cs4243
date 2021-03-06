import cv2
import cv2.cv as cv
import numpy as np

from Stitcher import Stitcher
from Tracker import Tracker
from Transformer import Transformer

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']

config_scale = True

if config_scale:
    image_down_scale_factor = 4
    H_left_mid = np.array([[4.27846244e-01, -2.25290426e-01, 3.97710942e+02],
                           [1.88683929e-02, 9.48302837e-01, 1.40909737e+01],
                           [-1.22572919e-03, 2.10230845e-05, 1.00000000e+00]])
    H_mid_right = np.array([[-1.23516364e+00, -1.41395849e-01, 1.62674397e+03],
                            [-8.41283372e-02, -1.16214461e+00, 1.35519101e+02],
                            [-1.60078790e-03, -5.02481792e-05, 1.00000000e+00]])
    crop_image_rect = {'min_x': 200, 'max_x': 2300, 'min_y': 100, 'max_y': 350}
else:
    image_down_scale_factor = 1
    H_left_mid = np.array([[4.48586889e-01, -2.05408064e-01, 1.58586590e+03],
                           [3.11830929e-02, 9.58631698e-01, 5.31001193e+01],
                           [-3.02387986e-04, 1.19548345e-05, 1.00000000e+00]])
    H_mid_right = np.array([[-1.20474129e+00, -1.40161277e-01, 6.45227999e+03],
                            [-8.11346378e-02, -1.12980266e+00, 5.25837708e+02],
                            [-3.88404089e-04, -1.04585070e-05, 1.00000000e+00]])
    crop_image_rect = {'min_x': 800, 'max_x': 9200, 'min_y': 400, 'max_y': 1400}


def crop_img(img):
    """
    Crop the black area after warping images together.
    :param img: The image to be cropped
    :return: The cropped image.
    """
    # TODO: Detect the black area and crop smartly.
    return img[crop_image_rect['min_y']:crop_image_rect['max_y'], crop_image_rect['min_x']: crop_image_rect['max_x']]


def main():
    stitcher = Stitcher()
    if config_scale:
        background = cv2.imread('images/background_scaled.jpg')
    else:
        background = cv2.imread('images/background.jpg')

    transformer = Transformer(config_scale)

    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_width = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT))

    init_points = {'C0': (71, 1153), \
                   'R0': (80, 761), 'R1': (80, 1033), 'R2': (95, 1127), 'R3': (54, 1156), 'R4': (65, 1185),
                   'R5': (61, 1204), 'R6': (56, 1217), 'R7': (69, 1213), 'R8': (67, 1253), 'R9': (75, 1281),
                   'R10': (92, 1347), \
                   'B0': (71, 1409), 'B1': (72, 1016), 'B2': (47, 1051), 'B3': (58, 1117), 'B4': (74, 1139),
                   'B5': (123, 1156), 'B6': (61, 1177), 'B7': (48, 1198), 'B8': (102, 1353)}

    points = init_points.values()
    tracker = Tracker(background, config_scale, init_points.values())

    # cap_left.set(cv.CV_CAP_PROP_POS_FRAMES, 1400)
    # cap_mid.set(cv.CV_CAP_PROP_POS_FRAMES, 1400)
    # cap_right.set(cv.CV_CAP_PROP_POS_FRAMES, 1400)
    for fr in range(frame_count):
        print(fr)
        status_left, frame_left = cap_left.read()
        status_mid, frame_mid = cap_mid.read()
        status_right, frame_right = cap_right.read()

        scaled_size = (frame_width / image_down_scale_factor, frame_height / image_down_scale_factor)
        frame_left = cv2.resize(frame_left, scaled_size)
        frame_mid = cv2.resize(frame_mid, scaled_size)
        frame_right = cv2.resize(frame_right, scaled_size)

        # Adjust the brightness difference.
        frame_mid = cv2.convertScaleAbs(frame_mid, alpha=0.92)

        if status_left and status_mid and status_right:
            warped_left_mid = stitcher.stitch(frame_mid, frame_left, H_left_mid)
            warped_left_mid_right = stitcher.stitch(warped_left_mid, frame_right, H_mid_right)
            warped_left_mid_right_cropped = crop_img(warped_left_mid_right)

            # plt.imshow(warped_left_mid_right_cropped)
            # plt.show()
            # cv2.waitKey(0)

            points = tracker.tracking(warped_left_mid_right_cropped)
            for i in range(len(points)):
                cv2.circle(warped_left_mid_right_cropped, (points[i][1], points[i][0]), 3, (0, 0, 255), -1)

            height, width = warped_left_mid_right_cropped.shape[:2]
            warped_left_mid_right_cropped = cv2.resize(warped_left_mid_right_cropped, (width / 2, height / 2))
            cv2.imshow('Objects', warped_left_mid_right_cropped)
            cv2.waitKey(1)

            # background = transformer.transform(points)
            # plt.imshow(warped_left_mid_right_cropped)
            # plt.show()
            # cv2.imshow('Objects', background)
            # cv2.waitKey(30)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()


if __name__ == '__main__':
    main()
