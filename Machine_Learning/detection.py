import socket,logging
import sys
import json
import cv2
import pickle
import numpy as np
import struct
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import math
import urllib.request
import time, threading
from utils import label_map_util
from utils import visualization_utilsM as vis_util

class Frame_recv():
    def __init__(self,HOST,PORT):
        self.host = HOST
        self.port = PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.addr = None
        self.is_cam = False
        self.is_conn = True
        self.cam_no = None
        self.data = b""
        self.frame = None
        self.index = 0
        self.payload_size = struct.calcsize(">L")
        self.cam_list = ['cam0','cam1','cam2','cam3','cam4','cam5','cam6','cam7','cam8','cam9']

    def __repr__(self):
        return 'Frame_recv'

    def conn_init(self):
        print('Socket bind complete')
        self.sock.bind((self.host,self.port))
        self.sock.listen(10)
        print('Socket now listening')
        self.conn, self.addr = self.sock.accept()
        self.is_cam = False

    def decode_data(self):
        data = self.conn.recv(1024)
        decoded = data.decode('utf-8')
        return decoded

    def make_cam(self): # 무한 루프문에서 돌게 하기
        decoded = self.decode_data()
        try:
            cam_num = int(decoded)  # Number of cameras
        except:
            self.is_cam = False
        else:  # Create objects for each camera
            self.conn.send("OK".encode("UTF-8"))
            print("Number of CCTV: {}".format(cam_num))
            for num in range(cam_num):
                decoded = self.decode_data()
                exec('{} =  Cam()'.format(self.cam_list[num]))
                exec("{}.parse_data(decoded)".format(self.cam_list[num]))
                exec("self.cam_list[num] = {}".format(self.cam_list[num]))
            self.cam_list = self.cam_list[:cam_num]
            self.is_cam = True

    def recv_frame(self):
        try:
            decoded = self.decode_data()
            self.conn.send(decoded.encode('utf-8'))
            self.index = int(decoded)
            # print("Cam({})'s Frame will be received".format(self.index))
            # Receive Frame SIZE from CCTV using TCP
            while len(self.data) < self.payload_size:
                self.data += self.conn.recv(4096)
        except ConnectionResetError:
            print("ConnectionResetError")
            self.conn.close()
            self.is_conn = False
        else:
            packed_msg_size = self.data[:self.payload_size]
            self.data = self.data[self.payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            # print("msg_size: {}".format(msg_size))
            while len(self.data) < msg_size:
                self.data += self.conn.recv(4096)
            frame_data = self.data[:msg_size]
            self.data = self.data[msg_size:]
            # Decode encoded Frame
            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            self.frame = frame

    def run(self):
        if not frecv.is_cam:
            frecv.make_cam()
        frecv.recv_frame()
        frecv.cam_list[self.index].frame = frecv.frame
        return self.index

class Cam(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='Cam({})'.format(id))
        self.id = None
        self.frame = None
        self.data = {'cam_id': self.id, 'trash': False, 'intrusion': False, 'fallen': False,'restricted':False,'fence':False,'wall':False}
        self.url = "http://220.67.124.197:8000/home/change_stat"
        self.actrec = actRecognition()
        self.e_list = []
        self.fxy_list = []
        self.txy_list = []

    def __repr__(self):
        return "CCTV_{}".format(self.id)

    def parse_data(self,data):
        """
        :param data: Received data from streaming process, decoded.
        :return: print id, restricted info
        """
        print("Parsing the CAM INFO...")
        self.id = int(data[data.find('cam_id') + 7:data.find('restricted') - 2])
        self.data['cam_id'] = self.id
        print("Cam_id: {}".format(self.id))
        if (data[data.find('restricted') + 11:data.find('wall') - 2]) == 'True':
            self.data['restricted'] = True
        else:
            self.data['restricted'] = False
        if (data[data.find('wall') + 5:]) == 'True':
            self.data['wall'] = True
        else:
            self.data['wall'] = False
        print("restricted: {}".format(self.data['restricted']))
        print(("COMPLETE"))

class actRecognition():
    def __init__(self):
        self.intr_warning = False  # 접근 제한 구역 침입 감지
        self.trash_warning = False  # 쓰레기 투기 감지
        self.fence_warning = False  # 월담 감지
        self.fence_prev = []  # 월담 구역에서 사람 경계 박스의 이전 위치 저장 list
        self.t_prev = []  # 쓰레기 투기 감지 시 쓰레기와 사람 경계 박스의 이전 위치 저장 list
        self.trFirst = True  # 쓰레기 감지 초기화 구분
        self.peopleNum = 0  # 월담 감지에서의 사람 수
        self.pID = 0  # 쓰레기 감지에서의 쓰레기를 들고 있는 사람 좌표가 저장된 인덱스
        self.trDistance = 0  # 쓰레기와 사람 상자 중점 간 거리
        self.midTr = []  # 쓰레기의 중점 좌표
        self.midP = []  # 쓰레기를 들고 있는 사람의 중점 좌표
        self.trashMulti = cv2.MultiTracker_create()  # 쓰레기 감지 추적기
        self.colist1 = [10, 350, 630, 350]  # 가상 펜스 직선을 그릴 좌표 - 위 직선
        self.colist2 = [10, 400, 630, 400]  # 가상 펜스 직선을 그릴 좌표 - 아래 직선
        self.trash_time = 0
        self.trashnum = 0
        self.fence_tracker = cv2.MultiTracker_create() # 월담 감지 추적기
        self.fallen_time = 0
        self.fence_people = []

    def fence_settings(self, cam):
        self.fence_prev, self.fence_people = [], []
        self.fence_tracker = cv2.MultiTracker_create()
        self.peopleNum = len(cam.fxy_list)
        for i in cam.fxy_list:
            x, w = abs(int(i[0] * 640)), abs(int(i[2] * 640))
            y, h = abs(int(i[1] * 480)), abs(int(i[3] * 480))
            cv2.rectangle(cam.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            window = (x, y, w, h)
            try:
                csrt = cv2.TrackerCSRT_create()
                self.fence_tracker.add(csrt, cam.frame, window)
            except cv2.error:
                pass
            if (y + h) < self.colist1[1] and y < self.colist1[1]:  # 담 너머의 사람일 때 마지막 원소 값 = 0
                self.fence_prev.append([x, y, w, h, 0, False])
            # 담 앞의 사람이고 상체만 인식되었을 때 마지막 원소 값 = 1
            elif self.colist2[1] < (y + h) < self.colist1[1] and y < self.colist1[1]:
                self.fence_prev.append([x, y, w, h, 1, False])
            # 담 앞의 사람이고 상하체가 모두 인식되었을 때 마지막 원소 값 = 2
            else:
                self.fence_prev.append([x, y, w, h, 2, False])
        print("init = ", self.fence_prev)

    def fence_updates(self, cam):
        (success, boxes) = self.fence_tracker.update(cam.frame)
        print("update success:", success)
        temp = []
        for box in boxes:
            [x, y, w, h] = [abs(int(v)) for v in box]
            cv2.rectangle(cam.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if (y + h) < self.colist1[1] and y < self.colist1[1]:  # 담 너머의 사람일 때 마지막 원소 값 = 0
                temp.append([x, y, w, h, 0, False])
            # 담 앞의 사람이고 상체만 인식되었을 때 마지막 원소 값 = 1
            elif self.colist2[1] < (y + h) < self.colist1[1] and y < self.colist1[1]:
                temp.append([x, y, w, h, 1, False])
            # 담 앞의 사람이고 상하체가 모두 인식되었을 때 마지막 원소 값 = 2
            else:
                temp.append([x, y, w, h, 2, False])
        if len(self.fence_people) != 0 and len(self.fence_people) <= len(temp):
            for index in self.fence_people:
                temp[index][5] = True
        for index, i in enumerate(temp):
            if index < len(self.fence_prev):
                print("y + h", y + h)
                # 담 뒤에 있던 사람이 앞으로 간 것으로 위치가 바뀐 경우 월담 감지
                if self.fence_prev[index][4] == 0 and self.fence_prev[index][4] != i[4] and not self.fence_prev[index][
                    5]:
                    self.fence_people.append(index)
                    self.fence_warning = True
                    print("y + h", y + h)
                    print("뒤 -> 앞 월담 감지")
                    i[5] = True
                    i[4] = 2
                # 담 앞에 있던 사람이 뒤로 간 것으로 위치가 바뀐 경우 월담 감지
                elif self.fence_prev[index][4] != i[4] and self.fence_prev[index][4] == 1 and not \
                self.fence_prev[index][5]:
                    self.fence_people.append(index)
                    self.fence_warning = True
                    print("앞 -> 뒤 월담 감지 (상체 부분 인식)")
                    i[5] = True
                    i[4] = 0
                elif self.fence_prev[index][4] == 2 and self.fence_prev[index][4] != i[4] and not \
                self.fence_prev[index][5]:
                    self.fence_people.append(index)
                    print("y + h", y + h)
                    self.fence_warning = True
                    print("앞 -> 뒤 월담 감지 (전신 인식)")
                    i[5] = True
                    i[4] = 0
        self.fence_prev = temp
        print("now = ", self.fence_prev)
        print("fence people = ", self.fence_people)

    def trSettings(self, cam):
        '''
        쓰레기 투기 감지용 객체 추적기를 초기화하여 쓰레기와 사람 좌표를 모두 넣어 세팅
        :param cam:
        :return:
        '''
        print("쓰레기 초기화")
        self.t_prev = []
        self.trashMulti = cv2.MultiTracker_create()
        window = ()
        i0, i1, i2, i3, self.trDistance = 0, 0, 0, 0, 0
        self.midTr, self.midP = [], []
        list = cam.txy_list + cam.fxy_list
        print("list =", list)
        self.trashnum = len(cam.txy_list)
        for index, i in enumerate(list):
            i0, i2 = abs(int(i[0] * 640)), abs(int(i[2] * 640))
            i1, i3 = abs(int(i[1] * 480)), abs(int(i[3] * 480))
            window = (i0, i1, i2, i3)
            self.t_prev.append(window)
            try:
                csrt = cv2.TrackerCSRT_create()
                self.trashMulti.add(csrt, cam.frame, window)  # 객체 추적기에 쓰레기 좌표 추가
                cv2.rectangle(cam.frame, (i0, i1), (i0 + i2, i1 + i3), (255, 0, 0), 2)
            except cv2.error:
                pass
            # 사람과 쓰레기 객체 경계 박스가 겹칠 때
            if index >= self.trashnum and ((i0 <= self.t_prev[0][0] <= (i0 + i2)) or
                                           (i0 <= (self.t_prev[0][2] + self.t_prev[0][0]) <= (i0 + i2))):
                if len(self.t_prev) > 1:
                    self.pID = len(self.t_prev) - 1  # 쓰레기를 들고 있는 사람
                ''''
                self.midTr = [(self.t_prev[0][0] * 2 + self.t_prev[0][2]) / 2,
                              (self.t_prev[0][1] * 2 + self.t_prev[0][3]) / 2]
                self.midP = [(i0 * 2 + i2) / 2, (i1 * 2 + i3) / 2]
                # 쓰레기와 사람 경계 박스의 중점 간 거리를 계산
                self.trDistance = math.sqrt(
                    math.pow(self.midTr[0] - self.midP[0], 2) + math.pow(self.midTr[1] - self.midP[1], 2))
                print("처음 거리:", self.trDistance)
                '''

    def trUpdates(self, cam):
        '''
        trSettings에서 세팅된 객체 추적기를 업데이트 하여 사람과 쓰레기의 중점 좌표 간 거리를 계산,
        그 거리가 200 픽셀 이상일때 쓰레기 투기로 감지.
        :param cam:
        :return:
        '''
        (success, boxes) = self.trashMulti.update(cam.frame)  # 객체 추적기 안의 박스들을 업데이트
        temp = []
        for box in boxes:
            [x, y, w, h] = [abs(int(v)) for v in box]
            cv2.rectangle(cam.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            temp.append([x, y, w, h])
        if len(temp) == 0:
            print("추적기에 아무것도 없음")
        else:
            print(temp)
            if self.pID < len(temp):
                print("중점 계산")
                self.midTr = [(temp[0][0] * 2 + temp[0][2]) / 2, (temp[0][1] * 2 + temp[0][3]) / 2]
                self.midP = [(temp[self.pID][0] * 2 + temp[self.pID][2]) / 2,
                             (temp[self.pID][1] * 2 + temp[self.pID][3]) / 2]
                p1 = (int((temp[0][0] * 2 + temp[0][2]) / 2), int((temp[0][1] * 2 + temp[0][3]) / 2))
                p2 = (int((temp[self.pID][0] * 2 + temp[self.pID][2]) / 2),
                      int((temp[self.pID][1] * 2 + temp[self.pID][3]) / 2))
                # 쓰레기를 들고 있던 사람 경계 박스의 중점과 쓰레기 경계 박스 중점 간 거리를 계산
                newDistance = math.sqrt(
                    math.pow(self.midTr[0] - self.midP[0], 2) + math.pow(self.midTr[1] - self.midP[1], 2))
                cv2.line(cam.frame, p1, p2, (255, 255, 255), 4)
                print("newDistance = ", newDistance)
                # 두 중점 간 거리가 처음보다 200 픽셀 이상 차이 나면
            if newDistance >= 200:
                self.trash_warning = True
                print("쓰레기 투기 - 거리 멀어짐")
            self.t_prev = temp

    def intr_compute(self, x, y):
        '''
        가상 펜스 경계 직선과 한 꼭짓점 간의 위치 파악 후 한 꼭짓점에 대한 boolean return
        :param x: 사람 객체 상자 한 꼭짓점의 x 좌표
        :param y: 사람 객체 상자 한 꼭짓점의 y 좌표
        :return:
        '''
        warning = False
        gr = (self.colist[3] - self.colist[1]) / (self.colist[2] - self.colist[0])  # 기울기
        yinter = -1 * gr * self.colist[0] + self.colist[1]  # y절편
        yout = gr * x + yinter
        if y < yout:  # 꼭지점이 경계선 위에 있을 때
            warning = True
        return warning

    def Trash_detect(self, cam):
        '''
        쓰레기 투기 감지를 진행
        :param cam:
        :return:
        '''
        if ((('trash' in cam.e_list) or ('metal' in cam.e_list) or (
                'bottle' in cam.e_list)) and self.trFirst):  # 쓰레기가 처음 감지되었을 때
            self.trash_time = 0
            self.trSettings(cam)
            self.trFirst = False
        elif (('trash' in cam.e_list) or ('metal' in cam.e_list) or ('bottle' in cam.e_list)) and (
                'person' in cam.e_list) and not self.trFirst:  # 쓰레기와 사람이 모두 감지되고 있을 때
            self.trash_time = 0
            self.trUpdates(cam)
        elif ('trash' in cam.e_list) or ('metal' in cam.e_list) or ('bottle' in cam.e_list):
            if self.trash_time == 0:
                self.trash_time = time.time()
            if time.time() - self.trash_time >= 10:
                print("투기된 쓰레기가 감지되었습니다. - 10초 이상")
                self.trash_warning = True
        else:  # 사람만 감지될 때
            if self.trash_time == 0:
                self.trash_time = time.time()
            if time.time() - self.trash_time >= 10:
                self.trash_time = 0
                self.trFirst = True

        # 상황 판단 DB 업데이트
        if self.trash_warning:
            print("쓰레기 무단 투기가 감지 되었습니다.")
            if cam.data['trash'] == False:
                cam.data['trash'] = True
                cam.actrec.send_post(cam)
            self.trash_warning = False
        else:
            if cam.data['trash'] == True:
                cam.data['trash'] = False
                cam.actrec.send_post(cam)

    def intr_check(self, cam):
        '''
        가상 펜스 경계를 넘었는지 확인하는 함수
        intr_compute 를 통해 각 꼭짓점의 침입 여부 계산
        :param cam:
        :return:
        '''
        f_stat = True
        for i, b in enumerate(cam.fxy_list):  # 객체 상자 좌표 리스트에서
            stat = []
            for m in range(0, len(cam.fxy_list[i]), 2):
                # 네 개의 꼭짓점 하나씩을 가상펜스 직선과 비교하여 그 위에 위치해있으면 True, 아니면 False
                stat.append(self.intr_compute(cam.fxy_list[i][m], cam.fxy_list[i][m + 1]))
            for k in stat:
                # 네 개의 꼭짓점이 모두 가상 펜스 위에 위치해있으면 최종적으로 f_stat에 True가 저장
                f_stat = f_stat and k
            self.intr_warning = f_stat
        if self.intr_warning:
            print("접근 제한 구역 침입이 감지되었습니다")
            if cam.data['intrusion'] == False:
                cam.data['intrusion'] = True
                cam.actrec.send_post(cam)

            self.intr_warning = False
        else:
            if cam.data['intrusion'] == True:
                print("해당 구역은 안전합니다(가상 펜스)")
                cam.data['intrusion'] = False
                cam.actrec.send_post(cam)

    def fallen_check(self,cam):
        '''
        쓰러진 사람 인식 시 DB 업데이트
        :return:
        '''
        if 'fallen' in cam.e_list:
            if self.fallen_time == 0:
                self.fallen_time = time.time()
            if time.time() - self.fallen_time >= 10:
                if cam.data['fallen'] == False:
                    cam.data['fallen'] = True
                    cam.actrec.send_post(cam)
                    print("쓰러진 사람 발견")
        else:
            self.fallen_time = 0
            if cam.data['fallen'] == True:
                cam.data['fallen'] = False
                cam.actrec.send_post(cam)
                print("쓰러진 사람 없음")

    def fence_detect(self, cam):
        '''
        월담 감지 진행
        :param cam:
        :return:
        '''
        # 사람 수가 0이 아니고 사람 수가 변경되었을 때
        if len(cam.fxy_list) != 0 and self.peopleNum != len(cam.fxy_list):
            self.fence_settings(cam)  # 객체 추적기 초기화
        elif len(cam.fxy_list) != 0:  # 사람 수의 변경이 없고 사람이 있을 때
            self.fence_updates(cam)  # 객체 추적기 update
        else:  # 사람이 없을 때
            self.peopleNum = 0

        if self.fence_warning:
            if cam.data['fence'] == False:
                print("월담을 감지했습니다")
                cam.data['fence'] = True
                cam.actrec.send_post(cam)
            self.fence_warning = False
        else:
            if cam.data['fence'] == True:
                print("해당 구역은 안전합니다(월담)")
                cam.data['fence'] = False
                cam.actrec.send_post(cam)

    def send_post(self,cam):
        params = json.dumps(cam.data).encode("utf-8")
        req = urllib.request.Request(cam.url, data=params,
                                     headers={'content-type': 'application/json'})
        response = urllib.request.urlopen(req)
        print(cam.id)

    def draw_line(self,cam):
        if cam.data['wall']:
            cv2.line(cam.frame, (self.colist1[0], self.colist1[1]), (self.colist1[2],self.colist1[3]),(255,0,0),2)
            cv2.line(cam.frame, (self.colist2[0], self.colist2[1]), (self.colist2[2],self.colist2[3]),(255,0,0),2)

    def run(self,cam):
        if cam.data["restricted"]:
            cam.actrec.intr_check(cam)
        if cam.data['wall']:
            cam.actrec.fence_detect(cam)
        cam.actrec.fallen_check(cam)
        cam.actrec.Trash_detect(cam)

class Obj_detection():
    def __init__(self,sess,detection_boxes, detection_scores, detection_classes, num_detections,image_tensor,frame_expanded,cam):
        self.sess = sess
        self.detection_boxes = detection_boxes
        self.detection_scores = detection_scores
        self.detection_classes = detection_classes
        self.num_detections = num_detections
        self.image_tensor = image_tensor
        self.frame_expanded = frame_expanded
        self.cam = cam

    def run(self):
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: self.frame_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            self.cam.frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            self.cam,
            use_normalized_coordinates=True,
            min_score_thresh=0.60)

if __name__ == '__main__':
    # 1.Start Initialize
    HOST = '192.168.0.66'
    # HOST= 'localhost'
    PORT = 8485
    # Ready to start machine learning
    # Load to memory
    # MODEL_NAME = 'inference_graph'
    MODEL_NAME = 'inference_graph'
    CWD_PATH = "C:/tensorflow1/models/research/object_detection"
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
    NUM_CLASSES = 5
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Load complete
    # Start connection
    frecv = Frame_recv(HOST,PORT)
    frecv.conn_init()
    # Connection complete
    # Until connection is closed
    while frecv.is_conn:
        # 2. Start receiving frame
        frecv.run()
        # Frame receive complete
        # 3.Ready to start Obj detection
        frame_expanded = np.expand_dims(frecv.cam_list[frecv.index].frame, axis=0)
        obj_dt = Obj_detection(sess, detection_boxes, detection_scores, detection_classes, num_detections, image_tensor,
                               frame_expanded,frecv.cam_list[frecv.index])
        # Start Obj detection
        obj_dt.run()
        # Obj detection complete. Frame is stored at object's attribute

        # 4.Start Act detection
        frecv.cam_list[frecv.index].actrec.run(frecv.cam_list[frecv.index])
        # for index in frecv.cam_list:
        #     if index.id == frecv.cam_list[i].id:
        #         index = frecv.cam_list[i]
        # End Act detection

        # Show for debugging
        # If restricted area, draw alert line
        frecv.cam_list[frecv.index].actrec.draw_line(frecv.cam_list[frecv.index])
        cv2.imshow('Object detector', frecv.cam_list[frecv.index].frame)
        # print("CAM_ID: {}".format(frecv.cam_list[frecv.index].id))

            # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            frecv.conn.close()
            break