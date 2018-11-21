from multiprocessing import Process
import logging
import socket
import threading, time, os, cv2
import numpy as np
import sys
import io
import struct
import time
import pickle
import zlib
import time

a = 1

def capture(frame,index):

    # png로 압축 영상 저장
    name = '/img_{}.png'.format(int(time.time())-(a*index))
    #index += 1
    cv2.imwrite('C:/Users/ice/Documents/GitHub/ICE_CAP/Django_channels/mysite/notifier/statics' + name, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 5])
    if index == 8000:
        a += 1

    print(index)

def cam_stream1(location,id,virtual, client_socket,encode_param):
    send_info(location, id, virtual, client_socket)
    video = cv2.VideoCapture(id)
    ret = video.set(3, 800)
    ret = video.set(4, 600)
    index = 0
    while True:
        ret, frame = video.read()
        capture(frame, index)
        index += 1
        ret, frame_encode = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame_encode, 0)
        size = len(data)

        try:
            client_socket.sendall(struct.pack(">L", size) + data)
        except ConnectionResetError:
            logging.error('ConnectionResetError')
            break
        except ConnectionAbortedError:
            logging.error('ConnectionAbortedError')
            break
        cv2.imshow('Object detector(cam_{})'.format(id), frame)
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            print("{}: Stream send closed".format(os.getpid()))
            client_socket.close()
            break

def cam_stream2(location,id,virtual, client_socket,encode_param):
    send_info(location, id, virtual, client_socket)
    video = cv2.VideoCapture(id)
    ret = video.set(3, 640)
    ret = video.set(4, 360)
    while True:
        ret, frame = video.read()
        ret, frame_encode = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame_encode, 0)
        size = len(data)
        try:
            client_socket.sendall(struct.pack(">L", size) + data)
        except ConnectionResetError:
            logging.error('ConnectionResetError')
            break
        except ConnectionAbortedError:
            logging.error('ConnectionAbortedError')
            break
        cv2.imshow('Object detector(cam_{})'.format(id), frame)
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            print("{}: Stream send closed".format(os.getpid()))
            client_socket.close()
            break

def cam_stream3(location,id,virtual, client_socket,encode_param):
    send_info(location, id, virtual, client_socket)
    video = cv2.VideoCapture(id)
    ret = video.set(3, 640)
    ret = video.set(4, 360)
    while True:
        ret, frame = video.read()
        ret, frame_encode = cv2.imencode('.jpg', frame, encode_param)
        data = pickle.dumps(frame_encode, 0)
        size = len(data)
        try:
            client_socket.sendall(struct.pack(">L", size) + data)
        except ConnectionResetError:
            logging.error('ConnectionResetError')
            break
        except ConnectionAbortedError:
            logging.error('ConnectionAbortedError')
            break
        cv2.imshow('Object detector(cam_{})'.format(id), frame)
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            print("{}: Stream send closed".format(os.getpid()))
            client_socket.close()
            break

def send_info(location,id,virtual,sock):
    req = "location:"+ location+ "\r\ncam_id:" + str(id) + "\r\nvirtual:" + str(virtual)
    sock.send(req.encode('UTF-8'))  # send message to server
    return print("Send Complete")

def connect_server(ADDRESS,PORT):
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('220.67.124.193', 8485))
            print('Connect Succes')
            return client_socket
        except TimeoutError:
            logging.error('TimeoutError')
            pass

def main():
    cam_info = (['1st_Floor', 0], ['2nd_Floor', 1], ['3rd_Floor', 2])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ADDRESS = '220.67.124.193'
    PORT = 8485
    client_socket = connect_server(ADDRESS,PORT)
    procs = []
    # proc = Process(target=oneshot)
    # procs.append(proc)
    # proc.start()
    for cam_list in range(3):
        video = cv2.VideoCapture(cam_list)
        if video.isOpened():  # 카메라 생성 확인
            if cam_list == 2:
                virtual = True
                send_info(cam_info[cam_list][0], cam_info[cam_list][1], virtual, client_socket)
                proc = Process(target=cam_stream3, args=(cam_info[cam_list][1], client_socket, encode_param))
                procs.append(proc)
                proc.start()
            else:
                virtual = False
                if cam_list == 0:
                    proc = Process(target=cam_stream1, args=(cam_info[cam_list][0],cam_info[cam_list][1],virtual, client_socket, encode_param))
                    procs.append(proc)
                    proc.start()
                else:
                    send_info(cam_info[cam_list][0], cam_info[cam_list][1], virtual, client_socket)
                    proc = Process(target=cam_stream2, args=(cam_info[cam_list][1], client_socket, encode_param))
                    procs.append(proc)
                    proc.start()

    for proc in procs:
        proc.join()

if __name__ == "__main__":
    main()

# class Cam(threading.Thread):
#     def __init__(self,location,id,virtual):
#         threading.Thread.__init__(self, name='Cam({})'.format(id))
#         self.url = "http://127.0.0.1:8000/home/change_stat"
#         self.virtual = virtual
#         self.id = id
#         self.count_warn = 0
#         self.frame = None
#         self.count_trash = 0
#         self.is_warning = False
#         self.is_trash = False
#         self.trash_timer = 0
#         self.e_list = []
#         self.c_list = []
#         self.data = {'cam_id': id, 'cam_status': 'safe', 'cam_location': location, 'trash': False,'trusion': False}
#         self.time = time.time()
#         self.MODEL_NAME = 'inference_graph'
#         # self.CWD_PATH = os.getcwd()
#         # self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
#         # self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'training', 'labelmap.pbtxt')
#         # self.NUM_CLASSES = 4
#         # self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
#         # self. categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
#         #                                                             use_display_name=True)
#         # self.category_index = label_map_util.create_category_index(self.categories)
#
# def oneshot():
#     cap =  cv2.VideoCapture(0)
#     ret = cap.set(3, 640)
#     while (True):
#         ret1, frame1 = cap.read()
#         ret2, frame2 = cap.read()
#         # 이미지 높이
#         height1 = frame1.shape[0]
#         height2 = frame2.shape[0]
#         # 이미지 넓이
#         width1 = frame1.shape[1]
#         width2 = frame2.shape[1]
#         # 이미지 색상 크기
#         depth1 = frame1.shape[2]
#         depth2 = frame2.shape[2]
#
#         dstvideo = create_image_multiple(height1, width1, depth1, 1, 2)
#         showMultiImage(dstvideo, frame1, height1, width1, depth1, 0, 0)
#         # 오른쪽 위에 표시(0,1)
#         showMultiImage(dstvideo, frame2, height2, width2, depth2, 0, 1)
#
#         cv2.imshow('Object detector', dstvideo)
#         if cv2.waitKey(1) == ord('q'):
#             break
#     cv2.destroyAllWindows()
#
# def showMultiImage(dst, src, h, w, d, col, row):
#     # 3 color
#     if d == 3:
#         dst[(col * h):(col * h) + h, (row * w):(row * w) + w] = src[0:h, 0:w]
#     # 1 color
#     elif d == 1:
#         dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 0] = src[0:h, 0:w]
#         dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 1] = src[0:h, 0:w]
#         dst[(col * h):(col * h) + h, (row * w):(row * w) + w, 2] = src[0:h, 0:w]
#
# def create_image(h, w, d):
#     image = np.zeros((h, w, d), np.uint8)
#     color = tuple(reversed((0, 0, 0)))
#     image[:] = color
#     return image
#
# def create_image_multiple(h, w, d, hcout, wcount):
#     image = np.zeros((h * hcout, w * wcount, d), np.uint8)
#     color = tuple(reversed((0, 0, 0)))
#     image[:] = color
#     return image