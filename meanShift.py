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

   track_window_width_absolute_thresh = 5
   track_window_height_absolute_thresh = 15

   track_window_width_scale_thresh = 1.4
   track_window_height_scale_thresh = 1.4

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

   def setTrack_window(self, track_window):
      self.track_window = track_window

   def adjustTrackWindowIfTooLarge(self, pre_track_window):
      current_window = self.track_window
      if((current_window[2] >= self.track_window_width_scale_thresh * pre_track_window[2] and current_window[2] >= self.track_window_width_absolute_thresh ) \
         or (current_window[3] >= self.track_window_height_scale_thresh * pre_track_window[3] and current_window[3] >= self.track_window_height_absolute_thresh)):
         # adjust to previous track window
         print "@@@@ want to adjust to:", current_window
         print "@@@@ but adjust to previous window:", pre_track_window

         # we need a small adjustment of offset in the movement direction
         offset_c, offset_r = self.findOffsetMovement(pre_track_window,current_window) 

         self.track_window = self.newTrackWindowWithCentreOffsets(pre_track_window, offset_c, offset_r)
         print "@@@@ finally adjusted window:", self.track_window

   def newTrackWindowWithCentreOffsets(self,track_window, offset_c, offset_r):
      c = track_window[0]
      r = track_window[1]
      w = track_window[2]
      h = track_window[3]
      return (c + offset_c,r + offset_r,w,h)

   # return offset_x, offset_y (offset_c, offset_r)
   def findOffsetMovement(self, pre_track_window, current_window):
      offset_scale = 1/5.0
      # pre_centre_x, pre_centre_y = self.findTrackWindowCentre(pre_track_window)
      # cur_centre_x, cur_centre_y = self.findTrackWindowCentre(current_window)
      # dx = cur_centre_x - pre_centre_x
      # dy = cur_centre_y - pre_centre_y
      dx = current_window[0] - pre_track_window[0]
      dy = current_window[1] - pre_track_window[1]
      print "@@@ dx, dy of centres:", dx, ", ", dy

      return int(dx*offset_scale), int(dy*offset_scale)


   # return c,r (x,y) of the centre of the window
   def findTrackWindowCentre(self, track_window):
      c = track_window[0]
      r = track_window[1]
      w = track_window[2]
      h = track_window[3]

      return c+w/2, r+h/2



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
      neighbourhood_size = 1
      # track_window: (col,row,width,height)
      dst_cutted = dst[self.track_window[1] - neighbourhood_size:self.track_window[1] + self.track_window[3] + neighbourhood_size, self.track_window[0] - neighbourhood_size :self.track_window[0] + self.track_window[2] + neighbourhood_size]

      col_offset = self.track_window[0] - neighbourhood_size
      row_offset = self.track_window[1] - neighbourhood_size

      cv2.imshow("calcBackProject", dst)
      cv2.waitKey(0)

      cv2.imshow("dst_cutted", dst_cutted)
      cv2.waitKey(0)

      # apply meanshift to get the new location
      # ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
      pre_track_window = self.track_window
      ret, self.track_window = cv2.CamShift(dst_cutted, self.track_window, self.term_crit)
      self.track_window = (self.track_window[0] + col_offset, self.track_window[1] + row_offset, self.track_window[2], self.track_window[3])
      # self.adjustTrackWindowIfTooLarge(pre_track_window)


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

      