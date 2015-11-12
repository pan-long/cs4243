import numpy as np
import cv2

class meanShiftTracker(object):
   roi = None
   hsv_roi = None 
   mask = None
   roi_hist = None
   # setup initial location of window
   # r,h,c,w = 123-10,5,1156-4,5  # simply hardcoded the values
   # r,h,c,w = 50,4,1117,2 # temporarily work for Camshift
   r, h, c, w = 50, 5, 1117, 2

   # r,h,c,w = 110, 2, 1153, 2
   track_window = (c, r, w, h)

   # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
   term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1 )
   # term_crit = ( cv2.TERM_CRITERIA_EPS , 50)

   def initFromFirstFrame(self, frame):
      print '====================== MeanShift: init frame ==================================='
      # set up the ROI for tracking
      roi = frame[self.r : self.r+self.h, self.c : self.c+self.w]
      cv2.imshow("roi",roi)
      cv2.waitKey(0)

      hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(hsv_roi, np.array((0.,0.,0.)), np.array((180.,255.,255.)))
      roi_hist = cv2.calcHist([hsv_roi],[0], mask, [180], [0,180])
      cv2.normalize(roi_hist,roi_hist, 0, 255, cv2.NORM_MINMAX)

      self.roi = roi
      self.hsv_roi = hsv_roi
      self.mask = mask
      self.roi_hist = roi_hist

      # cv2.rectangle(frame, (self.c, self.r), (self.c+self.w, self.r+self.h), (0, 0, 255), 1)
      # cv2.imshow("temp_ret", frame)
      # cv2.waitKey(0)


   def trackOneFrame(self, frame):
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

      cv2.imshow("calcBackProject", dst)
      cv2.waitKey(0)

      cv2.imshow("dst_cutted", dst_cutted)
      cv2.waitKey(0)

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
      cv2.rectangle(frame, (x,y), (x+w ,y+h), 255, 1)
      cv2.imshow('img2', frame)
      cv2.waitKey(0)

      # k = cv2.waitKey(60) & 0xff
      # if k == 27:
      #    break
      # else:
      #    cv2.imwrite(chr(k)+".jpg",img2)
      
      return frame

      