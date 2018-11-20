import cv2
import numpy as np

class Tracker():
    def __init__(self):
        #trackwindow = x, y, w, h
        self.prevwindow, self.trackwindow, self.roi, self.hsv_roi, self.mask, self.roi_hist = [], [], [], [], [], []
        self.hsv, self.dst, self.state = 0, 0, 0
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.warning = False

    def settings(self, box, frame):
        # box[] = [(x, y, w, h), (x, y, w, h), ....] : tensorflow
        # trackwindow = [(x, y, w, h), (x, y, w, h), ....] : opencv
        self.prevwindow, self.trackwindow, self.roi, self.hsv_roi, self.mask, self.roi_hist = [], [], [], [], [], []
        for i in box:
            i0, i2 = int(i[0] * 1280), int(i[2] * 1280)
            i1, i3 = int(i[1] * 720), int(i[3] * 720)
            self.trackwindow.append((i0, i1, i2, i3))

        self.prevwindow = self.trackwindow

        for i in self.trackwindow:
            index = self.trackwindow.index(i)
            self.roi.append(frame[i[0]: i[0] + i[3], i[1]: i[1] + i[2]])
            self.hsv_roi.append(cv2.cvtColor(self.roi[index], cv2.COLOR_BGR2HSV))
            self.mask.append(cv2.inRange(self.hsv_roi[index], np.array((0., 60., 32.)), np.array((180., 255., 255.))))
            self.roi_hist.append(cv2.calcHist([self.hsv_roi[index]], [0], self.mask[index], [180], [0, 180]))
            cv2.normalize(self.roi_hist[index], self.roi_hist[index], 0, 255, cv2.NORM_MINMAX)
            # Setup the termination criteria, either 10 iteration or move by at least 1 pt

    def update(self, frame, ret):
        for i in self.trackwindow:
            index = self.trackwindow.index(i)
            self.hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.dst = cv2.calcBackProject([self.hsv], [0], self.roi_hist[index], [0, 180], 1)
            # apply meanshift to get the new location
            ret, i = cv2.meanShift(self.dst, i, self.term_crit)
            # Draw it on image
            x, y, w, h = i
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            if (i[1] - self.prevwindow[index][1]) >= 30:
                self.warning = True
            print(i[1], self.prevwindow[index][1], self.warning)
            self.trackwindow[index] = i
            self.warning = False

        self.prevwindow = self.trackwindow

    def updateInit(self):
        self.hsv, self.dst, self.state = 0, 0, 0
