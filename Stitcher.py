import cv2
import math
import numpy as np
import numpy.linalg as la
from Blending import img_blending

class Stitcher(object):
    """
    Simple implementation of a stitcher that can take 2 images and figure out the homography then stitch
    them together.
    """

    # Use the SIFT feature detector.
    detector = cv2.SIFT()

    # Use Flann Matcher.
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    count_for_best_matches_in_knn = 2

    # Size of the Gaussian kernel used in Gaussian blurring process.
    Gaussian_ksize = (5, 5)

    ransac_reprojection_threshold = 5.0

    def filter_matches(self, matches, ratio=0.75):
        """
        Filter out good matches from a list of matches.

        :param matches: The matches to be filtered.
        :param ratio: The threshold used in filtering, the smaller, the better.
        :return: The filtered matches.
        """
        filtered_matches = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered_matches.append(m[0])

        return filtered_matches

    def find_dimensions(self, image, homography):
        """
        Find the dimension of an image after warping back to base image.

        :param image: The image to be warped.
        :param homography: The homography of this image to another base image.
        :return: The dimension tuple of the image to be warped to base image in the format of
            (min_x, min_y, max_x, max_y).
        """
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)

        (y, x) = image.shape[:2]

        base_p1[:2] = [0, 0]

        base_p2[:2] = [x, 0]
        base_p3[:2] = [0, y]
        base_p4[:2] = [x, y]

        max_x = None
        max_y = None
        min_x = None
        min_y = None

        for pt in [base_p1, base_p2, base_p3, base_p4]:

            hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

            hp_arr = np.array(hp, np.float32)

            normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

            if (max_x == None or normal_pt[0, 0] > max_x):
                max_x = normal_pt[0, 0]

            if (max_y == None or normal_pt[1, 0] > max_y):
                max_y = normal_pt[1, 0]

            if (min_x == None or normal_pt[0, 0] < min_x):
                min_x = normal_pt[0, 0]

            if (min_y == None or normal_pt[1, 0] < min_y):
                min_y = normal_pt[1, 0]

        min_x = min(0, min_x)
        min_y = min(0, min_y)

        return (min_x, min_y, max_x, max_y)

    def find_homography(self, base_img, img_to_match):
        """
        Find the homography from img_to_match to base_img.
        :param base_img: The base image used to find homography.
        :param img_to_match: The target image that is going to be matched with base image.
        :return: The 3x3 homography matrix.
        """
        # Convert the base_img to grayscale
        base_img_grayscale = cv2.GaussianBlur(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), 
                                                self.Gaussian_ksize, 0)

        # Convert the img_to_stitch to grayscale.
        img_to_match_grayscale = cv2.GaussianBlur(cv2.cvtColor(img_to_match, cv2.COLOR_BGR2GRAY),
                                                  self.Gaussian_ksize, 0)

        # Detect and compute the features in 2 images.
        img_to_match_features, img_to_match_descs = self.detector.detectAndCompute(img_to_match_grayscale, None)
        base_img_features, base_img_descs = self.detector.detectAndCompute(base_img_grayscale, None)


        # Match the features in 2 images.
        matches = self.matcher.knnMatch(img_to_match_descs, trainDescriptors=base_img_descs,
                                        k=self.count_for_best_matches_in_knn)

        # Filter out the best matches using the default threshold.
        matches_filtered = self.filter_matches(matches)

        base_img_features_filtered = []
        img_to_match_features_filtered = []

        for match in matches_filtered:
            base_img_features_filtered.append(base_img_features[match.trainIdx])
            img_to_match_features_filtered.append(img_to_match_features[match.queryIdx])

        base_img_features_points = np.array([f.pt for f in base_img_features_filtered])
        img_to_match_features_points = np.array([f.pt for f in img_to_match_features_filtered])

        # Find out the homography matrix from 2 sets of feature points.
        H, status = cv2.findHomography(base_img_features_points, img_to_match_features_points, cv2.RANSAC,
                                       self.ransac_reprojection_threshold)
        return H


    def stitch(self, base_img, img_to_stitch, homography=None):
        """
        Stitch img_to_stitch to base_img.
        :param base_img: The base image to which the img_to_stitch is going to be stitched on.
        :param img_to_stitch: The image to be stitched on base_img.
        :return: The warped image of the base_img and img_to_stitch.

        Note that the black part of the warped image will be chopped after stitching.
        """
        if homography is None:
            H = self.find_homography(base_img, img_to_stitch)
        else:
            H = homography
        H = H / H[2, 2]
        H_inv = la.inv(H)

        (min_x, min_y, max_x, max_y) = self.find_dimensions(img_to_stitch, H_inv)
        max_x = max(max_x, base_img.shape[1])
        max_y = max(max_y, base_img.shape[0])

        move_h = np.matrix(np.identity(3), np.float32)

        if (min_x < 0):
            move_h[0, 2] += -min_x
            max_x += -min_x

        if (min_y < 0):
            move_h[1, 2] += -min_y
            max_y += -min_y

        mod_inv_h = move_h * H_inv

        img_w = int(math.ceil(max_x))
        img_h = int(math.ceil(max_y))

        # Warp the new image given the homography from the old images.
        base_img_warp = cv2.warpPerspective(base_img, move_h, (img_w, img_h), borderMode=cv2.BORDER_TRANSPARENT)

        img_to_stitch_warp = cv2.warpPerspective(img_to_stitch, mod_inv_h, (img_w, img_h),
                                                 borderMode=cv2.BORDER_TRANSPARENT)

        # Put the base image on an enlarged palette.
        enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

        # Create a mask from the warped image for constructing masked composite.
        (ret, data_map) = cv2.threshold(cv2.cvtColor(img_to_stitch_warp, cv2.COLOR_BGR2GRAY),
                                        0, 255, cv2.THRESH_BINARY)

        enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                    mask=np.bitwise_not(data_map),
                                    dtype=cv2.CV_8U)

        # Now add the warped image.
        LS = img_blending(img_to_stitch_warp, enlarged_base_img)
        final_img = LS[0]
        for i in xrange(1,4):
            final_img = cv2.pyrUp(final_img)
            final_img = cv2.add(final_img, LS[i])
        
        # final_img = cv2.add(enlarged_base_img, img_to_stitch_warp,
        #                     dtype=cv2.CV_8U)
        
        return final_img
