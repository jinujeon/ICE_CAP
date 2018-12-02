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
# import tensorflow as tf
import sys
import math
import urllib.request
import time, threading
# from utils import label_map_util
# from utils import visualization_utilsM as vis_util

class Cam(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, name='Cam({})'.format(id))
        self.id = None
        self.frame = None
        self.data = {'cam_id': self.id, 'cam_status': 'safe', 'trash': False, 'intrusion': False, 'fallen': False}
        self.url = "http://127.0.0.1:8000/home/change_stat"
        # self.actrec = actRecognition()
        self.e_list = []
        self.fxy_list = []
        self.txy_list = []

    def __repr__(self):
        return "CCTV_{}".format(self.id)

    def parse_data(self,data):
        """
        :param data: Received data from streaming process, decoded.
        :return: print id, intrusion info
        """
        print("Parsing the CAM INFO...")
        self.id = int(data[data.find('cam_id') + 7:data.find('intrusion') - 2])
        self.data['cam_id'] = self.id
        print("Cam_id: {}".format(self.id))
        if (data[data.find('intrusion:') + 10:]) == 'True':
            self.data['intrusion'] = True
        else:
            self.data['intrusion'] = False
        print("intrusion: {}".format(self.data['intrusion']))
        print(("COMPLETE"))



def decode_data(conn):
    data = conn.recv(1024)
    decoded = data.decode('utf-8')
    return decoded

# 1.Start Initialize
HOST='localhost'
PORT=8485

# Socket conection
print('Socket created')
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST,PORT))
print('Socket bind complete')
sock.listen(10)
print('Socket now listening')
conn,addr=sock.accept()

# Receive camera information
# Availabe Cameras
cam_list = ['cam0','cam1','cam2','cam3','cam4','cam5','cam6','cam7','cam8','cam9']

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
# Done
# Start receiving frame
while True:
    i = 0
    try:
        decoded = decode_data(conn)
        conn.send(decoded.encode('utf-8'))
        for index in range(len(cam_list)):
            if cam_list[index].id == decoded:
                i = index
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
    cam_list[i].frame = frame
    # Finish receiving frame

    # Show for debugging
    if cam_list[i].data['intrusion']:
        cv2.line(cam_list[i].frame, (cam_list[i].actrec.colist[0], cam_list[i].actrec.colist[1]),(cam_list[i].actrec.colist[2], cam_list[i].actrec.colist[3]),(255, 0, 0), 2)

    cv2.imshow('Object detector', cam_list[i].frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        conn.close()
        break