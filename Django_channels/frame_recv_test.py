
import socket
import sys
import cv2
import pickle
import numpy as np
import struct
import zlib



def capture(frame,index):

    # png로 압축 없이 영상 저장
    name = '/img_{}.png'.format(index)
    #index += 1
    cv2.imwrite('C:/Users/ice/Documents/GitHub/ICE_CAP/Django_channels/mysite/notifier/statics' + name, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(index)



HOST='192.168.0.66'
PORT=8485

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()
while True:
    data = conn.recv(1024)
    decoded = data.decode('utf-8')
    if decoded:
        cam_info = decoded
        break
    else:
        continue
print("#########Cam_id#########: ",cam_info)
location = cam_info[cam_info.find('location')+9:cam_info.find('\r\n')]
cam_id = cam_info[cam_info.find('cam_id')+7 :cam_info.find('virtual') -2]
fence = cam_info[cam_info.find('virtual:')+8:cam_info.find('False')  + 5]
print("#######Cam_id:", cam_id, "location:", location, "fence:", fence)
index = 0
data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    capture(frame,index)
    index += 1
    cv2.imshow('ImageWindow',frame)
    cv2.waitKey(1)