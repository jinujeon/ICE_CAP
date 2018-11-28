
import logging
import socket
import time, cv2
import struct
import pickle

class Frame_scheduler:
    def __init__(self,detection_result,cam_count):
        self.detection_result = detection_result
        self.cam_stat_dict = dict([(id, None) for id in range(cam_count)])
        self.cam_count = cam_count

    def __setitem__(self, index, item):
        self._check_key(index)
        self.cam_stat_dict[index] = item

    def __getitem__(self, index):
        self._check_key(index)
        return self.cam_stat_dict[index]

    def _check_key(self,index):
        if not isinstance(index, int):
            raise TypeError('not Integer type')
        if index not in range(self.cam_count):
            raise IndexError('index out of range')

    def get_priority(self):
        pass
    def set_cam_frame_order(self,cam_id):
        self.get_priority()


class Frame_sender:
    def __init__(self,id_list,virtual,client_socket,encode_param):
        self.video_list = []
        self.frame_list = []
        self.ret_list = []
        self.virtual_fence = virtual
        ##입력받은 카메라 수 만큼 video_capture object를 생성##
        for id in id_list:
            exec('self.video_capture_{} =  cv2.VideoCapture({})'.format(id,id))
            exec('self.video_list.append(self.video_capture_{})'.format(id)) #exec로 생성한 videocapture객체에 접근하기 위해 리스트에 저장
        self.cam_count = len(id_list)
        self.socket = client_socket
        self.encode_param = encode_param
        self.detect_info = ''

    def send_frame(self,idx):
        if self.ret_list[idx]:
            ret0, frame_encode0 = cv2.imencode('.jpg', self.frame_list[idx], self.encode_param)
            data0 = pickle.dumps(frame_encode0, 0)
            size = len(data0)
            print("##SIZE{}##: ".format(idx), size)
            try:
                if ((int(time.time() ) % self.cam_count) == idx):
                    self.socket.send(str(idx).encode('UTF-8'))
                    data = self.socket.recv(1024)
                    print("Echo cam num:", data)
                    self.socket.sendall(struct.pack(">L", size) + data0)
                    print("Send comp{}".format(idx))
            except ConnectionResetError:
                logging.error('ConnectionResetError')
            except ConnectionAbortedError:
                logging.error('ConnectionAbortedError')
            cv2.imshow('Cam {}'.format(idx), self.frame_list[idx])

    def run(self):
        while True:
            # Capture frame-by-frame
            id = 0
            for video in self.video_list:
                exec('self.ret{}, self.frame{} = video.read()'.format(id,id))
                exec('self.ret_list.append(self.ret{})'.format(id))
                exec('self.frame_list.append(self.frame{})'.format(id))
                id += 1
            cv2.waitKey(100)
            for idx in range(len(self.ret_list)):
                self.send_frame(idx)
            self.frame_list = []
            self.ret_list = []
            self.recv_detect_info()

    def recv_detect_info(self):
        data = self.socket.recv(1024)
        self.detect_info = data.decode('UTF-8')



def send_info(id,virtual,sock):
    req = "cam_id:" + str(id) + "\r\nintrusion:" + str(virtual)
    sock.send(req.encode('UTF-8'))  # send message to server
    return print("Send Complete")

def initialize_server(ADDRESS,PORT,id_list,):
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((ADDRESS, PORT))
            cam_count = str(len(id_list))
            client_socket.send(cam_count.encode('UTF-8'))
            data = client_socket.recv(1024)
            print(data)
            # send_info(0,False,client_socket)
            for id in id_list:
                send_info(id,False,client_socket)
            print('Connect Succes')
            return client_socket
        except TimeoutError:
            logging.error('TimeoutError')


def main():
    id_list = [0,1] #설치되어 있는 카메라 id
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ADDRESS = '220.67.124.193'
    PORT = 8485
    client_socket = initialize_server(ADDRESS,PORT,id_list)
    fs1 = Frame_sender(id_list, False,client_socket, encode_param)
    fs1.run()

if __name__ == "__main__":
    main()
