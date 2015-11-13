import cv2
import numpy as np


class distCal (object):
    """
    Extract the background by averaging all the frames.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the extractor.
        :return: None.
        """
        self.distance = 0
        self.actual_distance = 0

    def getDist(self, pt1, pt2):
        distance = np.sqrt(pt1 ** 2 + pt2 ** 2)
        actual_distance = distance / 1067 * 110

        return actual_distance

