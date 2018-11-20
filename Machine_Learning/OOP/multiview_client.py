from multiprocessing import Process
from multiprocessing.connection import Listener,Client

import threading, time, os, cv2
import numpy as np
import sys

def showMultiImage(dst, src, h, w, d, col, row):
    # 3 color
    if d == 3:
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w] = src[0:h, 0:w]
    # 1 color
    elif d == 1:
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 0] = src[0:h, 0:w]
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 1] = src[0:h, 0:w]
        dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 2] = src[0:h, 0:w]

def create_image(h, w, d):
    image = np.zeros((h, w, d), np.uint8)
    color = tuple(reversed((0, 0, 0)))
    image[:] = color
    return image

def create_image_multiple(h, w, d, hcout, wcount):
    image = np.zeros((h * hcout, w * wcount, d), np.uint8)
    color = tuple(reversed((0, 0, 0)))
    image[:] = color
    return image



def oneshot():
    address = ('localhost', 8001)  # family is deduced to be 'AF_INET'
    listener = Listener(address)
    # listener2 = Listener(address, authkey=b'abcd')
    conn = listener.accept()
    # conn2 = listener2.accept()
    print('connection accepted from', listener.last_accepted)
    # cap2 = cv2.VideoCapture(0)  # 카메라 생성
    # cap =  cv2.VideoCapture(1)q
    # ret1 = cap2.set(3, 640)
    # ret = cap.set(3, 640)
    while (True):
        try:
            msg = conn.recv()
        except EOFError:
            break
        # ret1, frame1 = cap.read()
        # ret2, frame2 = cap.read()
        frame1 = msg
        frame2 = msg
        # if not ret1:
        #     continue
        # 이미지 높이
        height1 = frame1.shape[0]
        height2 = frame2.shape[0]
        # 이미지 넓이
        width1 = frame1.shape[1]
        width2 = frame2.shape[1]
        # 이미지 색상 크기
        depth1 = frame1.shape[2]
        depth2 = frame2.shape[2]

        dstvideo = create_image_multiple(height1, width1, depth1, 1, 2)
        showMultiImage(dstvideo, frame1, height1, width1, depth1, 0, 0)
        # 오른쪽 위에 표시(0,1)
        showMultiImage(dstvideo, frame2, height2, width2, depth2, 0, 1)

        cv2.imshow('Object detector', dstvideo)
        if cv2.waitKey(1) == ord('q'):
            listener.close()
            break
    cv2.destroyAllWindows()