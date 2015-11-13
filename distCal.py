import numpy as np


class distCal (object):
    """
    Calculate players' running distance
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

    def getDist(self,pts):
        shape = pts.shape
        distance = 0
        for i in range (shape):
            distance += np.sqrt(pts[0] ** 2 + pts[1] ** 2)

        actual_distance = distance / 1067 * 110

        return actual_distance

