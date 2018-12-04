
import logging
import socket
import time, cv2
import struct
import pickle

class VideoCamera(object):
    def __init__(self, idx):
        self.time = time.time()     #동영상 촬영시간 측정
        self.sizecontrol = 0        #동영상 용량 조절
        self.clock = time.gmtime(time.time()) #동영상 이름 -> 현재시간
        self.name = str(self.clock.tm_year) +'.'+ str(self.clock.tm_mon) +'.'+ str(self.clock.tm_mday) +'.'+ str(self.clock.tm_hour + 9) +'.'+ str(self.clock.tm_min) +'.'+ str(self.clock.tm_sec)
        self.videooutput = 0        #동영상 녹화 변수 정의
        # 카메라에 접근하기 위해 VideoCapture 객체를 생성
        self.video = cv2.VideoCapture(idx)
        # exec('self.video{} = cv2.VideoCapture(idx)'.format(idx, idx))
        (self.grabbed, self.frame) = self.video.read()

    def __del__(self):
        # 현재까지의 녹화를 멈춘다.
        self.video.release()
        self.videooutput.release()
        cv2.destroyAllWindows()

    def write(self):
        # 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 파일에 저장하기 위해 VideoWriter 객체를 생성
        self.videooutput = cv2.VideoWriter('output'+self.name+'.avi', fourcc, 10, (640, 480))

    def sizecon(self):
        # 동영상 용량 조정
        self.sizecontrol += 1
        if self.sizecontrol == 8000:
            self.sizecontrol = 0

    def storeframe(self, frame):
        # 동영상 프레임 실제 저장
        self.videooutput.write(frame)

class Frame_scheduler:
    def __init__(self,detection_result,id_list):
        self.detection_result = detection_result
        self.cam_count = len(id_list)
        self.cam_stat_dict = dict([(id, None) for id in range(self.cam_count)])

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

    def get_priority(self,id):
        priority = None
        if self.cam_stat_dict[id] == 'fallen':
            priority = 2
        if self.cam_stat_dict[id] == 'trash':
            priority = 1
        if self.cam_stat_dict[id] == 'intrusion':
            priority = 3
        return priority

    def set_cam_frame_order(self):
        for id in range(self.cam_count):
            self.get_priority(id)


class Frame_sender:
    def __init__(self,id_list,virtual,address,port,encode_param):
        self.id_list = id_list
        self.video_list = []
        self.frame_list = []
        self.ret_list = []
        self.virtual_fence = virtual
        ##입력받은 카메라 수 만큼 video_capture object를 생성##
        for id in id_list:
            exec('self.video_capture_{} =  store{}.frame'.format(id,id))
            exec('self.video_list.append(self.video_capture_{})'.format(id)) #exec로 생성한 videocapture객체에 접근하기 위해 리스트에 저장
        self.cam_count = len(id_list)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((address, port))
        self.encode_param = encode_param
        self.detect_info = ''

    def initialize_server(self):
        while True:
            try:
                self.socket.send(str(self.cam_count).encode('UTF-8')) #설치된 카메라 수 전송
                data = self.socket.recv(1024)
                print(data)
                # send_info(0,False,client_socket)
                for id in self.id_list:
                    self.send_info(id, False, self.socket)
                print('Connect Succes')
                return self.socket
            except TimeoutError:
                logging.error('TimeoutError')

    def send_info(self,id, virtual, sock):
        req = "cam_id:" + str(id) + "\r\nintrusion:" + str(virtual)
        sock.send(req.encode('UTF-8'))  # send message to server
        return print("Send Complete")

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
        self.initialize_server()
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
            #self.recv_detect_info()

    def recv_detect_info(self):
        data = self.socket.recv(1024)
        self.detect_info = data.decode('UTF-8')



def main():
    id_list = [0,1] #설치되어 있는 카메라 id
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    ADDRESS = 'localhost'
    PORT = 8485
    for i in range(len(id_list)):
        exec('store{} = VideoCamera({})'.format(id_list[i], id_list[i]))    # 영상 저장을 위한 객체 생성
        exec('store{}.write()'.format(id_list[i]))  # 영상저장함수실행
    # client_socket = initialize_server(ADDRESS,PORT,id_list)
    fs1 = Frame_sender(id_list, False,ADDRESS,PORT, encode_param)
    fs1.run()

if __name__ == "__main__":
    main()
