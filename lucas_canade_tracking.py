import numpy as np
import cv2

class LucasCanadeTracking(object):
   # params for ShiTomasi corner detection
   feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

   # Parameters for lucas kanade optical flow
   lk_params = dict( winSize  = (10,10),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

   # Create some random colors
   color = np.random.randint(0,255,(100,3))

   old_gray = None
   p0 = None
   mask = None

   # Take first frame and find corners in it
   # ret, old_frame = cap.read()
   # old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
   # p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

   # Create a mask image for drawing purpposes
   def setMask(self, old_color_frame):
      self.mask = np.zeros_like(old_color_frame)
      # print "self.mask:",self.mask

   def setOldGray(self, old_gray):
      self.old_gray = old_gray

   def setp0(self, initial_feature_points):
      # self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask = None, **self.feature_params)
      self.p0 = np.asarray(initial_feature_points).astype(np.float32)
      print "self.p0:\n",self.p0
      print "self.p0.shape:\n",self.p0.shape
      print "self.p0.dtype:\n",self.p0.dtype
      

   def trackingOneFrame(self,frame, color_frame):
      # print "in trackingOneFrame self.mask:",self.mask
      # print frame.shape
      # print frame.dtype
      # raise ValueError

      # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      frame_gray = frame

      # calculate optical flow
      p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

      # Select good points
      good_new = p1[st==1]
      good_old = self.p0[st==1]

      # draw the tracks
      for i,(new,old) in enumerate(zip(good_new,good_old)):
         a,b = new.ravel()
         c,d = old.ravel()
         # print "(a,b)",(a,b)
         # cv2.line(self.mask, (a,b),(c,d), self.color[i].tolist(), 1)
         # cv2.circle(color_frame,(a,b),1,(self.color[i].tolist()), -1)
         cv2.circle(color_frame,(a,b),1,(0,0,255), -1)
      
      # print "self.mask after drawing:",self.mask
      # print color_frame
      img = cv2.add(color_frame,self.mask)

      # print "img:", img
      # Now update the previous frame and previous points
      self.old_gray = frame_gray.copy()
      self.p0 = good_new.reshape(-1,1,2)

      return img

