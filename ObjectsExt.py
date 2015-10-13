import cv2
import numpy as np


class ObjectsExt(object):
    """
    Extract objects from a image. A background image is required to be compared with the image passed in.
    """
    def __init__(self, background):
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    def extract_objects(self, image, threshold=20):
        """
        Extract objects in image.
        :param image: The image that the objects are from.
        :param threshold: The threshold of the difference in grayscale to extract objects.
        :return: An image that contains only the objects.
        """
        image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        width, height = image.shape[:2]
        result_img = np.zeros((width, height))
        for i in range(height):
            for j in range(width):
                # * 1 to convert from unit8 to int to avoid overflow.
                if abs(image_grayscale[j, i] * 1 - self.background[j, i]) < threshold:
                    result_img[j, i] = 255
                else:
                    result_img[j, i] = image_grayscale[j, i]

        return cv2.convertScaleAbs(result_img)
