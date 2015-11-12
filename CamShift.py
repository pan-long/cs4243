import numpy as np
import cv2

class camShiftTracker(object):
    roi = None
    mask = None
    hsv_roi = None
    roi_hist = None

    def __init__(self, r, c):
        self.window_width = 2
        self.window_height = 5

        self.track_window = (c, r, self.window_width, self.window_height)
        # Setup the termination criteria
        self.term_critteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )


    def initFromFirstFrame(self, frame):
        print '====================== MeanShift: init frame ==================================='
        # set up the ROI for tracking
        roi = frame[self.r : self.r+self.h, self.c : self.c+self.w]
        print 'roi shape: ', roi.shape

        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi],[0], mask, [180], [0, 180])
        cv2.normalize(roi_hist,roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.roi = roi
        self.hsv_roi = hsv_roi
        self.mask = mask
        self.roi_hist = roi_hist

        # cv2.rectangle(frame, (self.c, self.r), (self.c+self.w, self.r+self.h), (0, 0, 255), 1)
        # cv2.imshow("temp_ret", frame)
        # cv2.waitKey(0)


    def trackFrame(self, frame):
        # cutted_frame = frame[self.track_window[1] - 10:self.track_window[1] + self.track_window[3] + 10, self.track_window[0] - 10 :self.track_window[0] + self.track_window[2] + 10]
        # cv2.imshow("cutted_frame", cutted_frame)
        # cv2.waitKey(0)
        # hsv = cv2.cvtColor(frame[self.track_window[1] - 10:self.track_window[1] + self.track_window[3] + 10, self.track_window[0] - 10 :self.track_window[0] + self.track_window[2] + 10 ], cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # compensate_c = self.track_window[0] - 10
        # compensate_r = self.track_window[1] - 10
        print hsv.shape
        print self.roi_hist.shape

        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0,180], 1)

        dst_cutted = dst[self.track_window[1] - 20 : self.track_window[1] + self.track_window[3] + 20, \
                   self.track_window[0] - 20 : self.track_window[0] + self.track_window[2] + 20]

        col_offset = self.track_window[0] - 20
        row_offset = self.track_window[1] - 20

        # cv2.imshow("calcBackProject", dst)
        # cv2.waitKey(0)

        # cv2.imshow("dst_cutted", dst_cutted)
        # cv2.waitKey(0)

        # apply meanshift to get the new location
        # ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        ret, self.track_window = cv2.CamShift(dst_cutted, self.track_window, self.term_crit)
        self.track_window = (self.track_window[0] + col_offset, self.track_window[1] + row_offset, self.track_window[2], self.track_window[3])


        print "self.track_window:", self.track_window
        # self.track_window[1] += compensate_r
        # self.track_window[0] += compensate_c
        # self.track_window = (self.track_window[0] + compensate_c, self.track_window[1] + compensate_r, self.track_window[2], self.track_window[3])
        print "self.track_window after compensate:", self.track_window

        # Draw it on image
        x, y, w, h = self.track_window
        # cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 1)
        # cv2.imshow('img2', frame)
        # cv2.waitKey(0)

        return [(x + w) / 2, (y + h) / 2]