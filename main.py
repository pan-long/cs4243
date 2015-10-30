import cv2
import numpy as np
import cv2.cv as cv

from Stitcher import Stitcher
from ObjectsExt import ObjectsExt
from BackgroundExt import BackgroundExt
from matplotlib import pyplot as plt

videos_path = 'videos/'
videos = ['football_left.mp4', 'football_mid.mp4', 'football_right.mp4']

config_scale = False

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
    # H_left_mid is the Homography to transform base(mid) to left
    H_left_mid = np.array([[  4.48586889e-01,  -2.05408064e-01,   1.58586590e+03],
                           [  3.11830929e-02,   9.58631698e-01,   5.31001193e+01],
                           [ -3.02387986e-04,   1.19548345e-05,   1.00000000e+00]])

    # H_mid_right is the Homography to transform base(mid_left) to right
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

def homographyFromMannualPoints(img_left, img_mid,img_right):
    print "img_left.shape:",img_left.shape
    # plt.imshow(np.dstack((img_left[:,:,2], img_left[:,:,1], img_left[:,:,0])))
    # plt.show()
    # plt.clf()
    # plt.imshow(np.dstack((img_mid[:,:,2], img_mid[:,:,1], img_mid[:,:,0])))
    # plt.show()
    # plt.clf()
    # plt.imshow(np.dstack((img_right[:,:,2], img_right[:,:,1], img_right[:,:,0])))
    # plt.show()
    # plt.clf()
    left_mid_src_points = []
    left_mid_dst_points = []
    

    left_mid_src_points.append(cv2.KeyPoint(119,295,10))
    left_mid_dst_points.append(cv2.KeyPoint(1631,350,10))

    left_mid_src_points.append(cv2.KeyPoint(140,186,10))
    left_mid_dst_points.append(cv2.KeyPoint(1676,246,10))

    left_mid_src_points.append(cv2.KeyPoint(155,58,10))
    left_mid_dst_points.append(cv2.KeyPoint(1725,119,10))

    left_mid_src_points.append(cv2.KeyPoint(120,65,10))
    left_mid_dst_points.append(cv2.KeyPoint(1687,124,10))    

    left_mid_src_points.append(cv2.KeyPoint(254,56,10))
    left_mid_dst_points.append(cv2.KeyPoint(1827,124,10))

    left_mid_src_points.append(cv2.KeyPoint(133,405,10))
    left_mid_dst_points.append(cv2.KeyPoint(1619,462,10))
    
    left_mid_src_points.append(cv2.KeyPoint(194,684,10))
    left_mid_dst_points.append(cv2.KeyPoint(1614,753,10))

    left_mid_src_points.append(cv2.KeyPoint(164,916,10))
    left_mid_dst_points.append(cv2.KeyPoint(1531,974,10))

    left_with_points = cv2.drawKeypoints(img_mid,left_mid_src_points,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("mid_with_points", left_with_points)
    cv2.waitKey(0)


    mid_with_points = cv2.drawKeypoints(img_left,left_mid_dst_points,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("left_with_points", mid_with_points)
    cv2.waitKey(0)


    H_left_mid = getHomography(left_mid_src_points, left_mid_dst_points)
    print "H_left_mid:\n",H_left_mid
    return 

def getHomography(keypoints_src, keypoints_dst):

    if(len(keypoints_src) != len(keypoints_dst)):
        raise ValueError("length of keypoints_src does not match length of keypoints_dst")
    src = []
    dst = []
    for i in range(0, len(keypoints_src)):
        print keypoints_src[i].pt[::-1]
        src.append(keypoints_src[i].pt)
        dst.append(keypoints_dst[i].pt)
    # print "src:\n",np.array(src)
    # print "dst:\n",np.array(dst)

    H, status = cv2.findHomography(np.array(src),np.array(dst), cv2.RANSAC, 10.0)

    # for i in range(0, len(src)):
    #     temp = [0,0,1]
    #     temp[0:2] = src[i]
    #     transformed = np.dot(H,np.transpose(temp))
    #     print transformed/transformed[len(transformed) - 1]
    return H

def main():
    bg = cv2.imread('final_back_ground_2.jpg')
    ext = BackgroundExt()
    # ext.background_img = bg
    # ext.number_of_images = 715

    stitcher = Stitcher()
    if config_scale:
        background = cv2.imread('background_scaled.jpg')
    else:
        background = cv2.imread('background.jpg')
    background_ext = cv2.BackgroundSubtractorMOG2()
    background_ext.apply(background)

    background_extractor = BackgroundExt()

    cap_left = cv2.VideoCapture(videos_path + videos[0])
    cap_mid = cv2.VideoCapture(videos_path + videos[1])
    cap_right = cv2.VideoCapture(videos_path + videos[2])

    frame_width = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap_mid.get(cv.CV_CAP_PROP_FRAME_COUNT))

    print "frame_count:", frame_count


    for fr in np.arange(0, frame_count,1):
    # for fr in np.arange(4200, 4300,1): # for background ext, get rid of the goal keeper 4200, 4152
        cap_left.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
        cap_mid.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
        cap_right.set(cv.CV_CAP_PROP_POS_FRAMES, fr)
        print "count:", fr 
        status_left, frame_left = cap_left.read()
        status_mid, frame_mid = cap_mid.read()
        status_right, frame_right = cap_right.read()

        # homographyFromMannualPoints(frame_left, frame_mid, frame_right)
        # raise ValueError("purpose stop for getting Homography")
        
        # cv2.imshow("left before warp:", frame_left)
        # cv2.waitKey(0)

        scaled_size = (frame_width / image_down_scale_factor, frame_height / image_down_scale_factor)
        frame_left = cv2.resize(frame_left, scaled_size)
        frame_mid = cv2.resize(frame_mid, scaled_size)
        frame_right = cv2.resize(frame_right, scaled_size)

        if status_left and status_mid and status_right:
            warped_left_mid = stitcher.stitch(frame_mid, frame_left, H_left_mid)
            # warped_left_mid = stitcher.stitch(frame_mid, frame_left, None)
            warped_left_mid_right = stitcher.stitch(warped_left_mid, frame_right, H_mid_right)
            warped_left_mid_right_cropped = crop_img(warped_left_mid_right)
            # cv2.imshow('warped_left_mid', warped_left_mid)
            # background = background_ext.apply(warped_left_mid_right_cropped)
            # background_extractor.add_image(warped_left_mid_right_cropped)
            # ext.add_image(warped_left_mid_right_cropped)

            # cv2.imshow('temp background', cv2.convertScaleAbs(ext.background_img))
            cv2.imwrite("temp_images/frame_{i}.jpg".format(i = fr), warped_left_mid_right_cropped)
            cv2.imshow('warped_left_mid_right_cropped',warped_left_mid_right_cropped)
            cv2.waitKey(3)
            # if(fr % 1000 == 0):
                # cv2.imshow('temp background', cv2.convertScaleAbs(background_extractor.background_img))
                # cv2.imwrite("final_back_ground_full_fr.jpg", cv2.convertScaleAbs(background_extractor.background_img))
            # # cv2.imshow('Objects', background)
            # cv2.waitKey(3)
            

    # cv2.imwrite("final_back_ground_full_fr.jpg", cv2.convertScaleAbs(background_extractor.background_img))
    # cv2.imwrite("final_back_ground_3.jpg", cv2.convertScaleAbs(ext.background_img))
    cv2.imshow("after mannually pick frame for average background:", cv2.convertScaleAbs(ext.background_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap_left.release()
    cap_mid.release()
    cap_right.release()


if __name__ == '__main__':
    main()
