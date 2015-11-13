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

    def getDist(self,prevD,x1,x2,y1,y2):
        distance = np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)

        actual_distance = distance / 1067 * 110

        return actual_distance + prevD

