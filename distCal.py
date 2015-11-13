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

    def getDist(self,x,y):
        distance = 0
        for i in range (len(x)-1):
            distance += np.sqrt((x[i+1]-x[i]) ** 2 + (y[i+1]-y[i]) ** 2)

        actual_distance = distance / 1067 * 110

        return actual_distance

