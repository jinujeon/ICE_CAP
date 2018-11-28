import socket,logging
import sys
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

class Cam(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='Cam({})'.format(id))
        self.url = "http://127.0.0.1:8000/home/change_stat"
        # self.colist = []
        self.frame = None
        self.id = None
        self.actrec = actRecognition()
        # self.count_trash = 0
        # self.count_fallen = 0
        # self.count_tracking= 0
        # self.is_trash = False
        # self.is_fallen = False
        # self.time = time.time()
        # self.trash_timer = 0
        self.e_list = []
        self.fxy_list = []
        self.txy_list = []
        self.data = {'cam_id': 0, 'cam_status': 'safe', 'trash': False,'instrusion': False,'fallen': False}
        # self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        # self. categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
        #                                                             use_display_name=True)
        # self.category_index = label_map_util.create_category_index(self.categories)

    def __repr__(self):
        return "CCTV_{}".format(self.id)

    def parse_data(self,data):
        """
        :param data: Received data from streaming process, decoded.
        :return: print id, instrusion info
        """
        print("Parsing the CAM INFO...")
        self.id = int(data[data.find('cam_id') + 7:data.find('instrusion') - 2])
        self.data['cam_id'] = self.id
        print("Cam_id: {}".format(self.id))
        if (data[data.find('instrusion:') + 11:]) == 'True':
            self.data['instrusion'] = True
        else:
            self.data['instrusion'] = False
        print("instrusion: {}".format(self.data['instrusion']))
        print(("COMPLETE"))

class actRecognition():
    def __init__(self):
        self.intr_warning = False  # 월담 감지
        self.trash_warning = False  # 쓰레기 투기 감지
        self.fence_warning = False  # 접근 제한 구역 침입 감지
        self.intr_prev = []  # 월담 구역에서 사람 경계 박스의 이전 위치 저장 list
        self.t_prev = []  # 쓰레기 투기 감지 시 쓰레기와 사람 경계 박스의 이전 위치 저장 list
        self.IntrFirst = True  # 월담 감지 초기화 구분
        self.trFirst = True  # 쓰레기 감지 초기화 구분
        self.peopleNum = 0  # 월담 감지에서의 사람 수
        self.pID = 0  # 쓰레기 감지에서의 쓰레기를 들고 있는 사람 좌표가 저장된 인덱스
        self.trDistance = 0  # 쓰레기와 사람 상자 중점 간 거리
        self.midTr = []  # 쓰레기의 중점 좌표
        self.midP = []  # 쓰레기를 들고 있는 사람의 중점 좌표
        self.intrMulti = cv2.MultiTracker_create()  # 월담 감지 추적기
        self.trashMulti = cv2.MultiTracker_create()  # 쓰레기 감지 추적기
        self.colist = [10, 330, 630, 330]  # 가상 펜스 직선을 그릴 좌표
        self.trash_time = 0

    def IntrSettings(self, cam):
        '''
        월담 감지에서의 객체 추적기를 초기화하여 사람 좌표를 추가해 세팅함
        :param cam:
        :return:
        '''
        self.intr_prev = []
        self.id = 0
        self.peopleNum = len(cam.fxy_list)
        for i in cam.fxy_list:
            i0, i2 = abs(int(i[0] * 640)), abs(int(i[2] * 640))
            i1, i3 = abs(int(i[1] * 360)), abs(int(i[3] * 360))
            window = (i0, i1, i2, i3)
            print(window)
            self.intr_prev.append(window)
            try:
                csrt = cv2.TrackerCSRT_create()
                self.intrMulti.add(csrt, cam.frame, window)
            except cv2.error:
                pass

    def IntrUpdates(self, cam):
        '''
        IntrSettings에서 세팅된 객체 추적기를 업데이트하여 같은 사람에 대해 상자 좌표의 수직을 비교,
        30 픽셀 이상 위로 움직였을 때 월담을 감지하여 DB 업데이트
        :param cam:
        :return:
        '''
        self.id = 0
        try:
            (success, boxes) = self.intrMulti.update(cam.frame)
            temp = []
            for box in boxes:
                [x, y, w, h] = [abs(int(v)) for v in box]
                temp.append([x, y, w, h])
                if self.id < len(self.intr_prev):
                    print("차이", self.intr_prev[self.id][1] - y)
                    if self.intr_prev[self.id][1] - y > 30:  # y축 변화가 위로 수직일 때만
                        print("******침입 감지 ! ! ! ! !*******")
                        self.intr_warning = True
                self.id += 1
                cv2.rectangle(cam.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.intr_prev = temp
            print("prev= ", self.intr_prev)
        except cv2.error:
            pass

    def trSettings(self, cam):
        '''
        쓰레기 투기 감지용 객체 추적기를 초기화하여 쓰레기와 사람 좌표를 모두 넣어 세팅
        :param cam:
        :return:
        '''
        self.t_prev = []
        self.trashMulti = cv2.MultiTracker_create()
        csrt = cv2.TrackerCSRT_create()
        window = ()
        i0, i1, i2, i3, self.trDistance = 0, 0, 0, 0, 0
        self.midTr, self.midP = [], []
        for i in cam.txy_list:  # 쓰레기 경계 박스 좌표를 tensorflow 기준에서 opencv 기준으로
            i0, i2 = abs(int(i[0] * 640)), abs(int(i[2] * 640))
            i1, i3 = abs(int(i[1] * 360)), abs(int(i[3] * 360))
            window = (i0, i1, i2, i3)
            self.t_prev.append(window)
            try:
                self.trashMulti.add(csrt, cam.frame, window)  # 객체 추적기에 쓰레기 좌표 추가
            except cv2.error:
                pass
        for i in cam.fxy_list:  # 사람 경계 박스 좌표를 tensorflow 기준에서 opencv 기준으로
            i0, i2 = abs(int(i[0] * 640)), abs(int(i[2] * 640))
            i1, i3 = abs(int(i[1] * 360)), abs(int(i[3] * 360))
            window = (i0, i1, i2, i3)
            self.t_prev.append(window)
            # 사람 경계 박스를 계산 중 쓰레기 경계 박스의 가로와 사람 경계 박스의 가로가 겹치면
            if (i0 <= self.t_prev[0][0] <= (i0 + i2)) or (i0 <= (self.t_prev[0][2] + self.t_prev[0][0]) <= (i0 + i2)):
                self.pID = len(self.t_prev) - 1  # 쓰레기를 들고 있는 사람
                self.midTr = [(self.t_prev[0][0] + self.t_prev[0][2]) / 2, (self.t_prev[0][1] + self.t_prev[0][3]) / 2]
                self.midP = [(i0 + i2) / 2, (i1 + i3) / 2]
                # 쓰레기와 사람 경계 박스의 중점 간 거리를 계산
                self.trDistance = math.sqrt(
                    math.pow(self.midTr[0] - self.midP[0], 2) + math.pow(self.midTr[1] - self.midP[1], 2))
            try:
                self.trashMulti.add(csrt, cam.frame, window)  # 객체 추적기에 사람 좌표 추가
            except cv2.error:
                pass

    def trUpdates(self, cam):
        '''
        trSettings에서 세팅된 객체 추적기를 업데이트 하여 사람과 쓰레기의 중점 좌표 간 거리를 계산,
        그 거리가 200 픽셀 이상일때 쓰레기 투기로 감지.
        :param cam:
        :return:
        '''
        try:
            (success, boxes) = self.trashMulti.update(cam.frame)  # 객체 추적기 안의 박스들을 업데이트
            temp = []
            for box in boxes:
                [x, y, w, h] = [abs(int(v)) for v in box]
                temp.append([x, y, w, h])
                cv2.rectangle(cam.frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            self.midTr = [(temp[0][0] + temp[0][2]) / 2, (temp[0][1] + temp[0][3]) / 2]
            self.midP = [(temp[self.pID][0] + temp[self.pID][2]) / 2, (temp[self.pID][1] + temp[self.pID][3]) / 2]
            # 쓰레기를 들고 있던 사람 경계 박스의 중점과 쓰레기 경계 박스 중점 간 거리를 계산
            newDistance = math.sqrt(
                math.pow(self.midTr[0] - self.midP[0], 2) + math.pow(self.midTr[1] - self.midP[1], 2))
            # 두 중점 간 거리가 처음보다 50 픽셀 이상 차이 나면
            if newDistance - self.trDistance >= 50:
                self.trash_warning = True
            self.t_prev = temp
        except cv2.error:
            pass

    def fence_compute(self, x, y):
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
            self.trSettings(cam)
            self.trFirst = False
        elif (('trash' in cam.e_list) or ('metal' in cam.e_list) or ('bottle' in cam.e_list)) and (
                'person' in cam.e_list):  # 쓰레기와 사람이 모두 감지되고 있을 때
            self.trUpdates(cam)
        elif ('trash' in cam.e_list) or ('metal' in cam.e_list) or ('bottle' in cam.e_list):
            if self.trash_time == 0:
                self.trash_time = time.time()
            if time.time() - self.trash_time >= 10:
                print("투기된 쓰레기가 감지되었습니다.")
                self.trash_warning = True
        else:  # 사람만 감지될 때
            self.trash_time = 0
            self.trFirst = True
        # 상황 판단 DB 업데이트
        if self.trash_warning:
            print("쓰레기 무단 투기가 감지 되었습니다.")
            if cam.data['trash'] == False:
                cam.data['cam_status'] = 'warning'
                cam.data['trash'] = True
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))
            self.trash_warning = False
        else:
            print("쓰레기가 없습니다.")
            if cam.data['trash'] == True:
                cam.data['cam_status'] = 'safe'
                cam.data['trash'] = False
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))

    def fence_check(self, cam):
        '''
        가상 펜스 경계를 넘었는지 확인하는 함수
        fence_compute 를 통해 각 꼭짓점의 침입 여부 계산
        :param cam:
        :return:
        '''
        f_stat = True
        for i, b in enumerate(cam.fxy_list):  # 객체 상자 좌표 리스트에서
            stat = []
            for m in range(0, len(cam.fxy_list[i]), 2):
                # 네 개의 꼭짓점 하나씩을 가상펜스 직선과 비교하여 그 위에 위치해있으면 True, 아니면 False
                stat.append(self.fence_compute(cam.fxy_list[i][m], cam.fxy_list[i][m + 1]))
            for k in stat:
                # 네 개의 꼭짓점이 모두 가상 펜스 위에 위치해있으면 최종적으로 f_stat에 True가 저장
                f_stat = f_stat and k
            self.fence_warning = f_stat
        if self.fence_warning:
            print("접근 제한 구역 침입이 감지되었습니다")
            if cam.data['instrusion'] == False:
                cam.data['cam_status'] = 'warning'
                cam.data['instrusion'] = True
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))
            self.fence_warning = False
        else:
            print("해당 구역은 안전합니다(가상 펜스)")
            if cam.data['instrusion'] == True:
                cam.data['cam_status'] = 'safe'
                cam.data['instrusion'] = False
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))

    def fallen_check(self,cam):
        '''
        쓰러진 사람 인식 시 바로 DB 업데이트
        :return:
        '''
        if 'fallen' in cam.e_list:
            if cam.data['fallen'] == False:
                cam.data['cam_status'] = 'warning'
                cam.data['fallen'] = True
                print("쓰러진 사람 발견")
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))
        else:
            if cam.data['fallen'] == True:
                cam.data['cam_status'] = 'safe'
                cam.data['fallen'] = False
                print("쓰러진 사람 없음")
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))

    def Intrusion_detect(self, cam):
        '''
        월담 감지 진행
        :param cam:
        :return:
        '''
        # 사람 수가 0이 아니고 사람 수가 변경되었을 때
        if len(cam.fxy_list) != 0 and self.peopleNum != len(cam.fxy_list):
            if not self.IntrFirst:  # 사람이 처음, 혹은 인식되지 않다가 다시 인식되었다면
                self.intrMulti = cv2.MultiTracker_create()
            else:
                self.IntrFirst = False
            self.IntrSettings(cam)  # 객체 추적기 초기화
        elif len(cam.fxy_list) != 0:  # 사람 수의 변경이 없고 사람이 있을 때
            self.IntrUpdates(cam)  # 객체 추적기 update
        else:  # 사람이 없을 때
            self.IntrFirst = True  # 첫 인식 변수 초기화

        if self.intr_warning:
            print("월담을 감지했습니다")
            if cam.data['instrusion'] == False:
                cam.data['cam_status'] = 'warning'
                cam.data['instrusion'] = True
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))
            self.intr_warning = False
        else:
            print("해당 구역은 안전합니다(월담)")
            if cam.data['instrusion'] == True:
                cam.data['cam_status'] = 'safe'
                cam.data['instrusion'] = False
                params = json.dumps(cam.data).encode("utf-8")
                req = urllib.request.Request(cam.url, data=params,
                                             headers={'content-type': 'application/json'})
                response = urllib.request.urlopen(req)
                print(response.read().decode('utf8'))

def decode_data(conn):
    data = conn.recv(1024).decode('utf-8')
    return data

# 1.Start Initialize
HOST='192.168.0.66'
PORT=8485
cam = Cam()
# Load to memory
MODEL_NAME = 'inference_graph'
CWD_PATH = "C:/models/research/object_detection"
# CWD_PATH = 'C:/tensorflow1/models/research/object_detection'
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')
NUM_CLASSES = 5
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,use_display_name=True)
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

# Socket conection
print('Socket created')
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST,PORT))
print('Socket bind complete')
# logging.INFO('Socket bind complete')
sock.listen(10)
print('Socket now listening')
# logging.INFO('Socket now listening')
conn,addr=sock.accept()

# Receive camera information
# Availabe Cameras
cam_list = ['cam1','cam2','cam3','cam4','cam5','cam6','cam7','cam8','cam9']

while True:
    decoded = decode_data(conn)
    # data = conn.recv(1024)
    # decoded = data.decode('utf-8')
    try:
        cam_num = int(decoded) # Number of cameras
    except:
        continue
    else: # Create objects for each camera
        conn.send("OK".encode("UTF-8"))
        print("Number of CCTV: {}".format(cam_num))
        for num in range(cam_num):
            decoded = decode_data(conn)
            # data = conn.recv(1024)
            # decoded = data.decode('utf-8')
            exec('{} =  Cam()'.format(cam_list[num]))
            exec("{}.parse_data(decoded)".format(cam_list[num]))
            exec("cam_list[num] = {}".format(cam_list[num]))
        break
# Initialize done


# 2.Ready to start receiving data
# data = conn.recv(1024)
# decoded = data.decode('utf-8')
cam_list = cam_list[:cam_num]
data = b""
payload_size = struct.calcsize(">L")
decoded = decode_data(conn)
conn.send(decoded.encode('utf-8'))
for index in cam_list:
    if index.id == decoded:
        cam = index
# Done

# Start receiving frame
while True:
    try:
        # Receive Frame SIZE from CCTV using TCP
        while len(data) < payload_size:
            #print("Recv: {}".format(len(data)))
            data += conn.recv(4096)
    except ConnectionResetError:
        print("ConnectionResetError")
        conn.close()
        break

    # Set SIZE of data
    # Receive Frame
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    #print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # Decode encoded Frame
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cam.frame = frame
    # Finish receiving frame

    # 3.Ready to start Obj detection
    frame_expanded = np.expand_dims(cam.frame, axis=0)
    # Start Obj detection
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    vis_util.visualize_boxes_and_labels_on_image_array(
        cam.frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        cam,
        use_normalized_coordinates=True,
        min_score_thresh=0.60)
    cam.time = time.time()
    # End Obj detection

    # 4.Start Act detection
    if cam.data["instrusion"]:
        cam.actrec.Intrusion_detect(cam)
        cam.actrec.fence_check(cam)
    else:
        cam.actrec.fallen_check(cam)
        cam.actrec.Trash_detect(cam)
    for index in cam_list:
        if index.id == cam.id:
            index = cam
    # End Act detection

    # Show for debugging
    if cam.data['instrusion']:
        cv2.line(cam.frame, (cam.actrec.colist[0], cam.actrec.colist[1]),(cam.actrec.colist[2], cam.actrec.colist[3]),(255, 0, 0), 2)
    cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        conn.close()
        break