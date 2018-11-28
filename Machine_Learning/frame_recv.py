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
import urllib.request
import time, threading
from utils import label_map_util
from utils import visualization_utilsM as vis_util

class Cam(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='Cam({})'.format(id))
        # self.url = "http://220.67.124.197:8000/home/change_stat"
        self.url = "http://127.0.0.1:8000/home/change_stat"
        self.colist = []
        self.frame = None
        self.id = None
        self.expired = False
        self.count_trash = 0
        self.count_fallen = 0
        self.count_tracking= 0
        self.is_trash = False
        self.is_fallen = False
        self.time = time.time()
        self.trash_timer = 0
        self.e_list = []
        self.fxy_list = []
        self.data = {'cam_id': 0, 'cam_status': 'safe', 'trash': False,'instrusion': False,'fallen': False}
        self.MODEL_NAME = 'inference_graph'
        self.CWD_PATH = "C:/models/research/object_detection"
        # self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
        # self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'training', 'labelmap.pbtxt')
        # self.NUM_CLASSES = 5
        # self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        # self. categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
        #                                                             use_display_name=True)
        # self.category_index = label_map_util.create_category_index(self.categories)

    def __repr__(self):
        return "CCTV_{}".format(self.id)

    def start_timer(self, second):
        def handler():
            self.expired = True

        self.timer = threading.Timer(second, handler)
        self.timer.start()

    def stop_timer(self):
        self.expired = False
        if self.timer.is_alive():   # Timer thread alive?
            self.timer.cancel()

    def parse_data(self,data):
        """
        :param data: Received data from streaming process, decoded.
        :return: print id, instrusion info
        """
        print("Parsing the CAM INFO...")
        self.id = int(data[data.find('cam_id') + 7:data.find('virtual') - 2])
        self.data['cam_id'] = self.id
        print("Cam_id: {}".format(self.id))
        if (data[data.find('virtual:') + 8:]) == 'True':
            self.data['instrusion'] = True
        else:
            self.data['instrusion'] = False
        print("instrusion: {}".format(self.data['instrusion']))
        print(("COMPLETE"))


def capture(frame,index):
    # png로 압축 영상 저장
    name = '/img_{}.png'.format(index)
    cv2.imwrite('C:/Users/Jun-Young/Desktop/Jun/I/ICE_CAP/Django_channels/mysite/notifier/statics' + name, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 5])
    # print(index)

# Initialize
HOST='192.168.0.66'
PORT=8485

# Load to memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(cam.PATH_TO_CKPT, 'rb') as fid:
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
cam_list = ['cam1','cam2','cam3','cam4','cam5','cam6','cam7','cam8','cam9']
x=0
while True:
    data = conn.recv(1024)
    decoded = data.decode('utf-8')
    try:
        cam_num = int(decoded)
    except:
        continue
    else:
        conn.send("OK".encode("UTF-8"))
        print("Number of CCTV: {}".format(cam_num))
        for num in range(cam_num):
            data = conn.recv(1024)
            decoded = data.decode('utf-8')
            exec('{} =  Cam()'.format(cam_list[num]))
            exec("{}.parse_data(decoded)".format(cam_list[num]))

data = b""
payload_size = struct.calcsize(">L")
index = 0
while True:
    try:
        while len(data) < payload_size:
            #print("Recv: {}".format(len(data)))
            data += conn.recv(4096)
    except ConnectionResetError:
        print("ConnectionResetError")
        conn.close()
        break

    #print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    #print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    capture(frame, index)
    index += 1
    if index == 5:
        index = 0
    cam.frame = frame
    #capture(frame,index)
    #index += 1
    frame_expanded = np.expand_dims(cam.frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    # cam.time = time.time()
    if int(time.time() - cam.time) >= 1:
        (boxes, scores, classes, num) = sess.run(
           [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        vis_util.visualize_boxes_and_labels_on_image_array(
            cam.frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            cam.category_index,
            cam,
            use_normalized_coordinates=True,
            min_score_thresh=0.60)
        cam.time = time.time()
        cv2.line(cam.frame, (cam.colist[0], cam.colist[1]),
                 (cam.colist[2], cam.colist[3]),
                 (255, 0, 0), 2)
        cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

    # cv2.imshow('Object detector({})'.format(cam.id), cam.frame)
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        cam.socket.close()
        break