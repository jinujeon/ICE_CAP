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

# a = 1

# def capture(frame,index):
#
#     # png로 압축 영상 저장
#     name = '/img_{}.png'.format(index)
#     cv2.imwrite('C:/Users/ice/Documents/GitHub/ICE_CAP/Django_channels/mysite/notifier/statics' + name, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 6])
#     # print(index)


# def cam_stream(location,id_list,virtual, client_socket,encode_param):
#     video_list = []
#     ret_list = []
#     frame_list = []
#     for id in id_list:
#         send_info(location, id, virtual, client_socket)
#         video_list.append(cv2.VideoCapture(id))
#     print("videos: ",video_list)
#
#     #fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     # 파일에 저장하기 위해 VideoWriter 객체를 생성
#     #out = cv2.VideoWriter('C:/Users/ice/Documents/GitHub/ICE_CAP/Django_channels/mysite/notifier/statics/output.avi', fourcc, 30.0, (640, 480))
#
#     #ret = video.set(3, 640)
#     #ret = video.set(4, 480)
#     index = 0
#
#     while True:
#         for video in video_list:
#             ret, frame = video.read()
#             ret_list.append(ret)
#             frame_list.append(frame)
#         print("ret_test: ",ret_list,"frame_list: ", frame_list)
#         #out.write(frame)
#         #capture(frame, index)
#         # index += 1
#         # if index == 5:
#         #     index = 0
#         if ret_list[0] :
#             ret0, frame_encode0 = cv2.imencode('.jpg', frame_list[0], encode_param)
#             data0 = pickle.dumps(frame_encode0, 0)
#             size = len(data0)
#
#             try:
#                 client_socket.sendall(struct.pack(">L", size) + data0)
#             except ConnectionResetError:
#                 logging.error('ConnectionResetError')
#                 break
#             except ConnectionAbortedError:
#                 logging.error('ConnectionAbortedError')
#                 break
#             cv2.imshow('Object detector(cam_{})'.format(0), frame_list[0])
#         if ret_list[1] :
#             ret1, frame_encode1 = cv2.imencode('.jpg', frame_list[1], encode_param)
#             data1 = pickle.dumps(frame_encode1, 0)
#             size = len(data1)
#
#             try:
#                 client_socket.sendall(struct.pack(">L", size) + data1)
#             except ConnectionResetError:
#                 logging.error('ConnectionResetError')
#                 break
#             except ConnectionAbortedError:
#                 logging.error('ConnectionAbortedError')
#                 break
#             cv2.imshow('Object detector(cam_{})'.format(1), frame_list[1])
#         # Press 'q' to quit
#         if cv2.waitKey(1) == ord('q'):
#             print("Process[{}]: Socket closed".format(os.getpid()))
#             client_socket.close()
#             break



def cam_stream1(location,id,virtual, client_socket,encode_param):
    send_info(location, id, virtual, client_socket)
    video = cv2.VideoCapture(id)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # 파일에 저장하기 위해 VideoWriter 객체를 생성
    out = cv2.VideoWriter('C:/Users/ice/Documents/GitHub/ICE_CAP/Django_channels/mysite/notifier/statics/output.avi', fourcc, 30.0, (640, 480))

    ret = video.set(3, 640)
    ret = video.set(4, 480)
    index = 0
    while True:
        ret, frame = video.read()
        out.write(frame)
        #capture(frame, index)
        index += 1
        if index == 5:
            index = 0
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
            print("Process[{}]: Socket closed".format(os.getpid()))
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
            print("Process[{}]: Socket closed".format(os.getpid()))
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
            print("Process[{}]: Socket closed".format(os.getpid()))
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
            client_socket.connect((ADDRESS, PORT))
            print('Connect Succes')
            return client_socket
        except TimeoutError:
            logging.error('TimeoutError')
            pass

def main():
    cam_info = (['1st_Floor', 0], ['2nd_Floor', 1], ['3rd_Floor', 2])
    id_list = [0,1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ADDRESS = 'localhost'
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

