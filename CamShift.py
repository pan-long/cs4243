import numpy as np
import cv2

class camShiftTracker(object):
    roi = None
    mask = None
    hsv_roi = None
    roi_hist = None

    margin = 20

    def __init__(self, r, c):
        self.row = r
        self.column = c
        self.window_width = 2
        self.window_height = 5

        self.track_window = (c, r, self.window_width, self.window_height)
        # Setup the termination criteria
        self.term_critteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )


    def initFromFirstFrame(self, frame):
        print '====================== MeanShift: init frame ==================================='
        # set up the ROI for tracking
        roi = frame[self.row : self.row+self.window_height, self.column : self.column+self.window_width]
        print 'roi shape: ', roi.shape

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0], mask, [180], [0, 180])
        cv2.normalize(roi_hist,roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.roi = roi
        self.hsv_roi = hsv_roi
        self.mask = mask
        self.roi_hist = roi_hist


    def trackFrame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # compensate_c = self.track_window[0] - 10
        # compensate_r = self.track_window[1] - 10

        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0,180], 1)

        dst_cutted = dst[self.track_window[1] - self.margin : self.track_window[1] + self.track_window[3] + self.margin, \
                   self.track_window[0] - self.margin : self.track_window[0] + self.track_window[2] + self.margin]

        col_offset = self.track_window[0] - self.margin
        row_offset = self.track_window[1] - self.margin

        if len(dst_cutted) > 0:
            # apply meanshift to get the new location
            ret, self.track_window = cv2.CamShift(dst_cutted, self.track_window, self.term_critteria)
            self.track_window = (self.track_window[0] + col_offset, self.track_window[1] + row_offset, self.track_window[2], self.track_window[3])

            print "self.track_window:", self.track_window

        x, y, w, h = self.track_window

        return [(y + h) / 2, (x + w) / 2]