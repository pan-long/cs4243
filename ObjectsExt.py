import cv2
import numpy as np


class ObjectsExt(object):
    """
    Extract objects from a image. A background image is required to be compared with the image passed in.
    """
    def __init__(self, background):
        self.background = background

    def extract_objects(self, image, threshold=20):
        """
        Extract objects in image.
        :param image: The image that the objects are from.
        :param threshold: The threshold of the difference in any of BGR to extract objects.
        :return: An image that contains only the objects.
        """
        image = np.array(image, dtype=int)
        width, height = image.shape[:2]
        result_img = np.zeros((width, height, 3))
        for i in range(height):
            for j in range(width):
                if any(d > threshold for d in abs(image[j, i] - self.background[j, i])):
                    result_img[j, i] = image[j, i]
                else:
                    result_img[j, i] = [255, 255, 255]

        return cv2.convertScaleAbs(result_img)
