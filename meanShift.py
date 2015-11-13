import cv2
import numpy as np


class meanShiftTracker(object):
    roi = None
    hsv_roi = None
    mask = None
    roi_hist = None
    # setup initial location of window
    r, h, c, w = 50, 5, 1117, 2

    track_window = (c, r, w, h)

    track_window_width_absolute_thresh = 5
    track_window_height_absolute_thresh = 15

    track_window_width_scale_thresh = 1.4
    track_window_height_scale_thresh = 1.4

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1)
    # term_crit = ( cv2.TERM_CRITERIA_EPS , 50)

    track_window_cutting_width_thresh = 3
    track_window_cutting_height_thresh = 8

    white_region_thresh = 1
    black_intensity = 0

    dilation_kernel = np.ones((2, 2), np.uint8)

    MAX_ROW_INDEX = 250
    MAX_COL_INDEX = 2100

    manual_tracking = False

    def mouseEventCallback(self, event, x, y, flags, user_data):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.setTrack_window((x - 2, y - 5, 4, 10))
            self.manual_tracking = True

    def initFromFirstFrame(self, frame):
        print '====================== MeanShift: init frame ==================================='
        # set up the ROI for tracking
        roi = frame[self.r: self.r + self.h, self.c: self.c + self.w]
        # cv2.imshow("roi",roi)
        # cv2.waitKey(0)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 0., 0.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        self.roi = roi
        self.hsv_roi = hsv_roi
        self.mask = mask
        self.roi_hist = roi_hist

        cv2.imshow('football', frame)
        cv2.setMouseCallback('football', self.mouseEventCallback)
        cv2.waitKey(0)
        

    def setTrack_window(self, track_window):
        self.track_window = track_window

    def adjustTrackWindowIfTooLarge(self, pre_track_window):
        current_window = self.track_window
        if ((current_window[2] >= self.track_window_width_scale_thresh * pre_track_window[2] and current_window[
            2] >= self.track_window_width_absolute_thresh) \
                    or (
                            current_window[3] >= self.track_window_height_scale_thresh * pre_track_window[3] and
                            current_window[
                                3] >= self.track_window_height_absolute_thresh)):
            # adjust to previous track window
            # we need a small adjustment of offset in the movement direction
            offset_c, offset_r = self.findOffsetMovement(pre_track_window, current_window)

            self.track_window = self.newTrackWindowWithCentreOffsets(pre_track_window, offset_c, offset_r)

    def newTrackWindowWithCentreOffsets(self, track_window, offset_c, offset_r):
        c = track_window[0]
        r = track_window[1]
        w = track_window[2]
        h = track_window[3]
        return (c + offset_c, r + offset_r, w, h)

    # return offset_x, offset_y (offset_c, offset_r)
    def findOffsetMovement(self, pre_track_window, current_window):
        offset_scale = 1 / 5.0
        dx = current_window[0] - pre_track_window[0]
        dy = current_window[1] - pre_track_window[1]

        return int(dx * offset_scale), int(dy * offset_scale)

    # return c,r (x,y) of the centre of the window
    def findTrackWindowCentre(self, track_window):
        c = track_window[0]
        r = track_window[1]
        w = track_window[2]
        h = track_window[3]

        return c + w / 2, r + h / 2

    def searchConnectedNeighbours(self, full_img, cur_r, cur_c, memorization):
        if (not "{r}_{c}".format(r=cur_r, c=cur_c) in memorization):
            memorization["{r}_{c}".format(r=cur_r, c=cur_c)] = 1

        for i in range(cur_r - 1, cur_r + 1 + 1):
            for j in range(cur_c - 1, cur_c + 1 + 1):
                if (not (i == cur_r and j == cur_c)):
                    # Do not consider diagonal:
                    # if(not (i == cur_r and j == cur_c) and \
                    #    not (i == cur_r - 1 and j == cur_c - 1) and \
                    #    not (i == cur_r - 1 and j == cur_c + 1) and \
                    #    not (i == cur_r + 1 and j == cur_c + 1) and \
                    #    not (i == cur_r + 1 and j == cur_c - 1)):
                    # i, j has intensity greater than black and i, j is not checked
                    if (i < self.MAX_ROW_INDEX and j < self.MAX_COL_INDEX and full_img[i][
                        j] > self.white_region_thresh and (not "{r}_{c}".format(r=i, c=j) in memorization)):
                        self.searchConnectedNeighbours(full_img, i, j, memorization)

    # r_start, r_end, c_start, c_end are all inclusive
    def expandSearchAreaConnectedComponentAndSuppress(self, cur_roi_img, full_img, r_start, r_end, c_start, c_end):
        memorization = {}
        for i in range(r_start, r_end + 1):
            for j in range(c_start, c_end + 1):
                if (full_img[i][j] > self.white_region_thresh):
                    self.searchConnectedNeighbours(full_img, i, j, memorization)

        r_min = r_start
        r_max = r_end
        c_min = c_start
        c_max = c_end
        for key, value in memorization.iteritems():
            r_c_chars = key.split("_")
            r = int(r_c_chars[0])
            c = int(r_c_chars[1])
            # update r_min, r_max, c_min, c_max
            if (r < r_min):
                r_min = r
            if (r > r_max):
                r_max = r
            if (c < c_min):
                c_min = c
            if (c > c_max):
                c_max = c
        # cut out the new roi
        new_roi_img = np.copy(full_img[r_min:r_max + 1, c_min:c_max + 1])
        # suppress the ones not in memorization
        for i in range(r_min, r_max + 1):
            for j in range(c_min, c_max + 1):
                if (not "{r}_{c}".format(r=i, c=j) in memorization):
                    # suppress the current pixel in new_roi_img
                    new_roi_img[i - r_min][j - c_min] = self.black_intensity
        return new_roi_img, r_min, r_max, c_min, c_max


    def adjustSearchArea(self, cur_roi_img, full_img, r_start, r_end, c_start, c_end):
        # check roi top
        r_top = r_start
        for i in range(0, cur_roi_img.shape[1]):
            if (cur_roi_img[0][i] > self.white_region_thresh):
                col = c_start + i
                row = r_start
                while (full_img[row][col] > self.white_region_thresh):
                    row -= 1
                row += 1  # back to last white point
                if (row < r_top):
                    r_top = row

        # check roi left
        c_left = c_start
        for i in range(0, cur_roi_img.shape[0]):
            if (cur_roi_img[i][0] > self.white_region_thresh):
                row = r_start + i
                col = c_start
                while (full_img[row][col] > self.white_region_thresh):
                    col -= 1
                col += 1  # back to last white point
                if (col < c_left):
                    c_left = col

        # check roi right
        c_right = c_end
        for i in range(0, cur_roi_img.shape[0]):
            if (cur_roi_img[i][cur_roi_img.shape[1] - 1] > self.white_region_thresh):
                row = r_start + i
                col = c_end
                while (full_img[row][col] > self.white_region_thresh):
                    col += 1
                # if the col is larger than previous checked ones
                col -= 1  # back to last white point
                if (col > c_right):
                    c_right = col

        # check roi bottom
        r_bottom = r_end
        for i in range(0, cur_roi_img.shape[1]):
            if (cur_roi_img[cur_roi_img.shape[0] - 1][i] > self.white_region_thresh):
                col = c_start + i
                row = r_end
                while (full_img[row][col] > self.white_region_thresh):
                    row += 1
                row -= 1  # back to last white point
                if (row > r_bottom):
                    r_bottom = row

        # To deal with the case where there is empty in the middle but like a contour case (hollow shape), pad extra to the 4 sides
        pad_extra = 0  # in pixel
        return full_img[r_top - pad_extra:r_bottom + 1 + pad_extra,
               c_left - pad_extra:c_right + 1 + pad_extra], r_top - pad_extra, r_bottom + pad_extra, c_left - pad_extra, c_right + pad_extra

    def shrinkSearchArea(self, cur_roi_img, full_img, r_start, r_end, c_start, c_end):
        # print "r_start, r_end, c_start, c_end:", (r_start, r_end, c_start, c_end)
        black_intensity = 0
        # shrink top
        r_top = r_start
        tops = []  # row index of the top
        for i in range(0, cur_roi_img.shape[1]):
            if (cur_roi_img[0][i] <= black_intensity):
                # print "$$$ need to shrink top"
                col = c_start + i
                row = r_start
                while (row < r_end and full_img[row][col] <= black_intensity):
                    row += 1
                if (row != r_end):
                    tops.append(row)
            else:
                tops.append(r_start)
        if (len(tops) > 0):
            r_top = np.min(tops)

        # shrink left
        c_left = c_start
        lefts = []
        for i in range(0, cur_roi_img.shape[0]):
            if (cur_roi_img[i][0] <= black_intensity):
                # print "$$$ need to shrink left"
                row = r_start + i
                col = c_start
                while (col < c_end and full_img[row][col] <= black_intensity):
                    col += 1
                if (col != c_end):
                    lefts.append(col)
            else:
                lefts.append(c_start)
        if (len(lefts) > 0):
            c_left = np.min(lefts)

        # shrink right
        c_right = c_end
        rights = []
        for i in range(0, cur_roi_img.shape[0]):
            if (cur_roi_img[i][cur_roi_img.shape[1] - 1] <= black_intensity):
                # print "$$$ need to shrink right"
                row = r_start + i
                col = c_end
                while (col > c_start and full_img[row][col] <= black_intensity):
                    col -= 1
                if (col != c_start):
                    rights.append(col)
            else:
                rights.append(c_end)
        if (len(rights) > 0):
            c_right = np.max(rights)

        # shrink bottom
        r_bottom = r_end
        bottoms = []  # row index of the top
        for i in range(0, cur_roi_img.shape[1]):
            if (cur_roi_img[cur_roi_img.shape[0] - 1][i] <= black_intensity):
                # print "$$$ need to shrink bottom"
                col = c_start + i
                row = r_end
                while (row > r_start and full_img[row][col] <= black_intensity):
                    row -= 1
                if (row != r_start):
                    bottoms.append(row)
            else:
                bottoms.append(r_end)
        if (len(bottoms) > 0):
            r_bottom = np.max(bottoms)

        return full_img[r_top:r_bottom + 1, c_left:c_right + 1], r_top, r_bottom, c_left, c_right

    def trackOneFrame(self, frame):        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # dst = cv2.dilate(dst, self.dilation_kernel, iterations=1)
        # cv2.imshow("calcBackProject", dst)
        # cv2.waitKey(0)

        neighbourhood_size = 0
        # track_window: (col,row,width,height)
        dst_cutted = dst[self.track_window[1] - neighbourhood_size:self.track_window[1] + self.track_window[
            3] + neighbourhood_size, self.track_window[0] - neighbourhood_size:self.track_window[0] + self.track_window[
            2] + neighbourhood_size]
        # cv2.imshow("dst_cutted before expansion", dst_cutted)
        # cv2.waitKey(0)



        # Should be doing expansion first then shrink!!!
        # dst_cutted, r_top, r_bottom, c_left, c_right = self.adjustSearchArea(dst_cutted, dst, self.track_window[1] - neighbourhood_size, self.track_window[1] + self.track_window[3] + neighbourhood_size - 1, self.track_window[0] - neighbourhood_size, self.track_window[0] + self.track_window[2] + neighbourhood_size - 1)
        dst_cutted, r_top, r_bottom, c_left, c_right = self.expandSearchAreaConnectedComponentAndSuppress(dst_cutted,
                                                                                                          dst,
                                                                                                          self.track_window[
                                                                                                              1] - neighbourhood_size,
                                                                                                          self.track_window[
                                                                                                              1] +
                                                                                                          self.track_window[
                                                                                                              3] + neighbourhood_size - 1,
                                                                                                          self.track_window[
                                                                                                              0] - neighbourhood_size,
                                                                                                          self.track_window[
                                                                                                              0] +
                                                                                                          self.track_window[
                                                                                                              2] + neighbourhood_size - 1)
        # print "dst_cutted.shape after expansion:", dst_cutted.shape
        # cv2.imshow("dst_cutted after expansion", dst_cutted)
        # cv2.waitKey(0)

        orig_dst_cutted = dst_cutted
        dst_cutted, r_top, r_bottom, c_left, c_right = self.shrinkSearchArea(dst_cutted, dst, r_top, r_bottom, c_left,
                                                                             c_right)
        if not len(dst_cutted):
            dst_cutted = orig_dst_cutted
        # print "dst_cutted.shape after shrink:", dst_cutted.shape
        # cv2.imshow("dst_cutted after shrink", dst_cutted)
        # cv2.waitKey(0)

        col_offset = c_left
        row_offset = r_top

        """
        Apply CamShift
        """  
        # apply meanshift to get the new location
        pre_track_window = self.track_window
        # print "before Camshift:", self.track_window
        ret, self.track_window = cv2.CamShift(dst_cutted, (0, 0, c_right - c_left + 1, r_bottom - r_top + 1),
                                              self.term_crit)

        self.track_window = list(self.track_window)
        if self.track_window[2] == 0:
            self.track_window[2] = 4
        if self.track_window[3] == 0:
            self.track_window[3] = 10
        self.track_window = tuple(self.track_window)

        # set offset
        self.track_window = (self.track_window[0] + col_offset, self.track_window[1] + row_offset, self.track_window[2],
                             self.track_window[3])

        dst_cutted_after_cam_shift = dst[self.track_window[1]:self.track_window[1] + self.track_window[3],
                                     self.track_window[0]:self.track_window[0] + self.track_window[2]]
        # cv2.imshow("dst_cutted_after_cam_shift", dst_cutted_after_cam_shift)
        # cv2.waitKey(0)

        # Draw it on image
        x, y, w, h = self.track_window
        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 1)
        # cv2.imshow('img2', frame)
        # cv2.waitKey(0)

        return frame, [y+h, x + w/2] # return point's as [r, c]
