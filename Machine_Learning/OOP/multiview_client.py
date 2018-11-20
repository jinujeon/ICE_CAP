from multiprocessing import Process
from multiprocessing.connection import Listener,Client

import threading, time, os, cv2
import numpy as np
import sys
import socket
import io
import struct
import time
import pickle
import zlib

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
    cam_info = (['1st_Floor', 0], ['2nd_Floor', 1], ['3rd_Floor', 2])
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('220.67.124.240', 8485))
    # connection = client_socket.makefile('wb')
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    # 소켓 통신 통해서 보낼것 -> frame, id, loc
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    while (True):
        ret1, frame1 = cap.read()
        ret2, frame2 = cap.read()
        ret3, frame3 = cv2.imencode('.jpg', frame1, encode_param)
        data = pickle.dumps(frame3, 0)
        size = len(data)
        client_socket.sendall(struct.pack(">L", size) + data)

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
            break
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    oneshot()