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

def q(a,b,c):
    cam = Cam(a,b,c)
    video = cv2.VideoCapture(cam.id)
    ret = video.set(3, 640)
    ret = video.set(4, 360)
    address = ('localhost', 8001)
    conn = Client(address)
    print(type(conn))
    while True:
        ret, cam.frame = video.read()
        try:
            conn.send(cam.frame)
        except EOFError:
            pass
        except ConnectionAbortedError:
            pass
        except ConnectionRefusedError:
            pass
        cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            conn.close()
            break

def qq(a,b,c):
    cam = Cam(a,b,c)
    video = cv2.VideoCapture(cam.id)
    ret = video.set(3, 640)
    ret = video.set(4, 480)
    # address = ('localhost', 8000)
    # conn = Client(address, authkey=b'abcd')
    while True:
        ret, cam.frame = video.read()
        # conn.send(cam.frame)
        cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

def qqq(a,b,c):
    cam = Cam(a,b,c)
    video = cv2.VideoCapture(cam.id)
    ret = video.set(3, 640)
    ret = video.set(4, 360)
    while True:
        ret, cam.frame = video.read()
        cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

class Cam(threading.Thread):
    def __init__(self,location,id,virtual):
        threading.Thread.__init__(self, name='Cam({})'.format(id))
        self.url = "http://127.0.0.1:8000/home/change_stat"
        self.virtual = virtual
        self.id = id
        self.count_warn = 0
        self.frame = None
        self.count_trash = 0
        self.is_warning = False
        self.is_trash = False
        self.trash_timer = 0
        self.e_list = []
        self.c_list = []
        self.data = {'cam_id': id, 'cam_status': 'safe', 'cam_location': location, 'trash': False,'trusion': False}
        self.time = time.time()
        self.MODEL_NAME = 'inference_graph'
        # self.CWD_PATH = os.getcwd()
        # self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
        # self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'training', 'labelmap.pbtxt')
        # self.NUM_CLASSES = 4
        # self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        # self. categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
        #                                                             use_display_name=True)
        # self.category_index = label_map_util.create_category_index(self.categories)

if __name__ == "__main__":
    cam_info = (['1st_Floor', 0], ['2nd_Floor', 1], ['3rd_Floor', 2])
    procs = []
    proc = Process(target=oneshot)
    procs.append(proc)
    proc.start()
    for cam_list in range(3):
        video = cv2.VideoCapture(cam_list)
        if video.isOpened() == True:  # 카메라 생성 확인
            if cam_list == 2:
                virtual = True
                proc = Process(target=qqq, args=(cam_info[cam_list][0], cam_info[cam_list][1], virtual))
                procs.append(proc)
                proc.start()
            else:
                virtual = False
                if cam_list == 0:
                    proc = Process(target=q, args=(cam_info[cam_list][0], cam_info[cam_list][1], virtual))
                    procs.append(proc)
                    proc.start()
                else:
                    proc = Process(target=qq, args=(cam_info[cam_list][0], cam_info[cam_list][1], virtual))
                    procs.append(proc)
                    proc.start()
        # proc = Process(target=oneshot)
        # procs.append(proc)
        # proc.start()

    for proc in procs:
        proc.join()
    # oneshot()