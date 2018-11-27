import numpy as np
import cv2
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
def cam_stream(id_list,virtual, client_socket,encode_param):
    video_capture_0 = cv2.VideoCapture(0)
    video_capture_1 = cv2.VideoCapture(1)
    video_list = []
    ret_list = []
    frame_list = []
    # for id in id_list:
    #     send_info(len(id_list),id, virtual, client_socket)
    while True:
        # Capture frame-by-frame
        ret0, frame0 = video_capture_0.read()
        ret1, frame1 = video_capture_1.read()

        if (ret0):
            # Display the resulting frame
            ret0, frame_encode0 = cv2.imencode('.jpg', frame0, encode_param)
            data0 = pickle.dumps(frame_encode0, 0)
            size = len(data0)

            try:
                client_socket.sendall(struct.pack(">L", size) + data0)
                print("Send comp1")
            except ConnectionResetError:
                logging.error('ConnectionResetError')
                break
            except ConnectionAbortedError:
                logging.error('ConnectionAbortedError')
                break
            cv2.imshow('Cam 0', frame0)

        if (ret1):
            # Display the resulting frame
            ret1, frame_encode1 = cv2.imencode('.jpg', frame1, encode_param)
            data1 = pickle.dumps(frame_encode1, 0)
            size = len(data1)

            try:
                client_socket.sendall(struct.pack(">L", size) + data1)
                print("Send comp2")
            except ConnectionResetError:
                logging.error('ConnectionResetError')
                break
            except ConnectionAbortedError:
                logging.error('ConnectionAbortedError')
                break
            cv2.imshow('Cam 1', frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def send_info(len,id,virtual,sock):
    req = "cam_count"+ str(len) + "\r\ncam_id:" + str(id) + "\r\nvirtual:" + str(virtual)
    sock.send(req.encode('UTF-8'))  # send message to server
    return print("Send Complete")

def initailize_server(ADDRESS,PORT,id_list, virtual):
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((ADDRESS, PORT))
            cam_count = str(len(id_list))
            client_socket.send(cam_count.encode('UTF-8'))
            data = client_socket.recv(1024)
            print(data)
            print('Connect Succes')
            return client_socket
        except TimeoutError:
            logging.error('TimeoutError')
            pass

def main():
    #cam_info = (['1st_Floor', 0], ['2nd_Floor', 1], ['3rd_Floor', 2])
    id_list = [0,1] #설치되어 있는 카메라 아이디
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ADDRESS = 'localhost'
    PORT = 8485
    client_socket = initailize_server(ADDRESS,PORT,id_list,False)

    cam_stream(id_list, False, client_socket, encode_param)

if __name__ == "__main__":
    main()
# When everything is done, release the capture
# video_capture_0.release()
# video_capture_1.release()
# cv2.destroyAllWindows()