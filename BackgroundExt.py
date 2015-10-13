import cv2


class BackgroundExt(object):
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
        self.number_of_images = 0
        self.background_img = None

    def add_image(self, image):
        """
        Add a new image in the image series.
        :param image: The image added
        :return: The background after adding this new image.
        """
        if self.background_img is None:
            self.background_img = image
        else:
            self.background_img = self.number_of_images / (self.number_of_images + 1.) * self.background_img + \
                                  1. / (self.number_of_images + 1.) * image
            self.background_img = cv2.convertScaleAbs(self.background_img)
        self.number_of_images += 1
        return self.background_img
