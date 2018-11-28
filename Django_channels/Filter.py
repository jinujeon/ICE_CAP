import numpy as np
import logging
import socket
import threading, time, os, cv2
import struct
import pickle

# class Frame_filter:
#     def __init__(self):
#
#

class Frame_sender:
    def __init__(self,id_list,virtual,client_socket,encode_param):
        self.video_list = []
        self.frame_list = []
        self.ret_list = []
        self.virtual = virtual
        ##입력받은 카메라 수 만큼 video_capture object를 생성##
        for id in id_list:
            exec('self.video_capture_{} =  cv2.VideoCapture({})'.format(id,id))
            exec('self.video_list.append(self.video_capture_{})'.format(id)) #exec로 생성한 videocapture객체에 접근하기 위해 리스트에 저장
        self.cam_count = len(id_list)
        self.socket = client_socket
        self.encode_param = encode_param
    def run(self):
        while True:
            # Capture frame-by-frame
            id = 0
            for video in self.video_list:
                exec('self.ret{}, self.frame{} = video.read()'.format(id,id))
                exec('self.ret_list.append(self.ret{})'.format(id))
                exec('self.frame_list.append(self.frame{})'.format(id))
                id += 1
            cv2.waitKey(100) #0.1sec delay -> 10 frames per second
            for idx in range(len(self.ret_list)):
                if self.ret_list[idx]:
                    ret0, frame_encode0 = cv2.imencode('.jpg', self.frame_list[idx], self.encode_param)
                    data0 = pickle.dumps(frame_encode0, 0)
                    size = len(data0)
                    print("##SIZE{}##: ".format(idx), size)
                    print("")
                    try:
                        if ((int(time.time() * 10) % self.cam_count) == idx): #0.1초에 한 번씩 전송하는 카메라 프레임을 변경 ex) 0.1초에는 cam1의 frame, 0.2초에는 cam2의 frame..........
                            self.socket.send(str(idx).encode('UTF-8'))
                            data = self.socket.recv(1024)
                            print("Echo cam num:", data)
                            self.socket.sendall(struct.pack(">L", size) + data0)
                            print("Send comp{}".format(idx))
                    except ConnectionResetError:
                        logging.error('ConnectionResetError')
                        break
                    except ConnectionAbortedError:
                        logging.error('ConnectionAbortedError')
                        break
                    cv2.imshow('Cam {}'.format(idx), self.frame_list[idx])
                    #exec("""cv2.imshow('Cam {}', self.frame_list[{}])""".format(idx,idx))
            self.frame_list = []
            self.ret_list = []


def send_info(len,id,virtual,sock):
    req = "cam_count"+ str(len) + "\r\ncam_id:" + str(id) + "\r\nvirtual:" + str(virtual)
    sock.send(req.encode('UTF-8'))  # send message to server
    return print("Send Complete")

def initailize_server(ADDRESS,PORT,id_list,):
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((ADDRESS, PORT))
            cam_count = str(len(id_list))
            client_socket.send(cam_count.encode('UTF-8'))
            #data = client_socket.recv(1024)
            #print(data)
            print('Connect Succes')
            return client_socket
        except TimeoutError:
            logging.error('TimeoutError')
            pass

def main():
    id_list = [0,1] #설치되어 있는 카메라 아이디
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ADDRESS = 'localhost'
    PORT = 8485
    client_socket = initailize_server(ADDRESS,PORT,id_list)


    fs1 = Frame_sender(id_list, False,client_socket, encode_param)
    fs1.run()
if __name__ == "__main__":
    main()
