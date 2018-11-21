import cv2

class multitracker():
    def __init__(self):
        self.multitrackers = cv2.MultiTracker_create()
        self.bboxes = []
        self.prev = []
        self.isFirst = 0
        self.over = False
        self.id = 0

    def settings(self, coord, frame):
        self.multitrackers = cv2.MultiTracker_create()
        self.prev = []
        for i in coord:
            i0, i2 = int(i[0] * 1280), int(i[2] * 1280)
            i1, i3 = int(i[1] * 800), int(i[3] * 800)
            #trackers.add(tracker, frame, box)
            #roi = frame[x:x + h, y:y + w]
            #roi = frame[i0: i0 + i3, i1: i1 + i2]
            window = (i0, i1, i2, i3)
            self.prev.append(window)
            csrt = cv2.TrackerCSRT_create()
            self.multitrackers.add(csrt, frame, window)
            print("init =", self.prev)

    def updatebox(self, frame):
        '''
            for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        :return:
        '''
        #(success, boxes) = trackers.update(frame)
        self.id = 0
        (success, boxes) = self.multitrackers.update(frame)
        print(type(boxes))
        print(len(boxes))
        for box in boxes:
            #(x0, y0, w0, h0) = [int(k) for k in self.prev[self.id]]
            (x, y, w, h) = [int(v) for v in box]
            #self.prev[self.id] = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            #if y - y0 >= 20:
            #    self.over = True
            #self.id += 1
        #self.prev = boxes
        self.prev = boxes.tolist()
        
        print(self.prev)
        #print("prev = ", self.prev)