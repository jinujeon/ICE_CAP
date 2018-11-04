#!/opt/local/bin/python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

CAM_ID = 0



#검정색 이미지를 생성
#h : 높이
#w : 넓이
#d : 깊이 (1 : gray, 3: bgr)
def create_image(h, w, d):
    image = np.zeros((h, w,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image



#검정색 이미지를 생성 단 배율로 더 크게
#hcout : 높이 배수(2: 세로로 2배)
#wcount : 넓이 배수 (2: 가로로 2배)
def create_image_multiple(h, w, d, hcout, wcount):
    image = np.zeros((h*hcout, w*wcount,  d), np.uint8)
    color = tuple(reversed((0,0,0)))
    image[:] = color
    return image



#통이미지 하나에 원하는 위치로 복사(표시) 
#dst : create_image_multiple 함수에서 만든 통 이미지
#src : 복사할 이미지
#h : 높이
#w : 넓이
#d : 깊이
#col : 행 위치(0부터 시작)
#row : 열 위치(0부터 시작)

def showMultiImage(dst, src, h, w, d, col, row):
    dst[(col*h):(col*h)+h, (row*w):(row*w)+w] = src[0:h, 0:w]


##### 코드 시작 ####
cap2 = cv2.VideoCapture(CAM_ID) #카메라 생성
cap = cv2.VideoCapture(1)
ret = cap.set(3,640)
ret1 = cap2.set(3,640)
ret = cap.set(4,480)
ret1 = cap2.set(4,480)

if cap.isOpened() == False: #카메라 생성 확인
    print ('Can\'t open the CAM(%d)' % (CAM_ID))
    exit()
elif cap2.isOpened() == False: #카메라 생성 확인
    print ('Can\'t open the CAM(%d)' % (CAM_ID))
    exit()

#윈도우 생성 및 사이즈 변경
cv2.namedWindow('multiView')

while(True):
    #카메라에서 이미지 얻기
    ret, frame = cap.read()
    ret1, frame1 = cap2.read()

    # 이미지 높이
    height = frame.shape[0]

    # 이미지 넓이
    width = frame.shape[1]

    # 이미지 색상 크기
    depth = frame.shape[2]

    # 화면에 표시할 이미지 만들기 ( 1 x 2 )
    dstimage = create_image_multiple(height, width, depth, 1, 2)

    # 원하는 위치에 복사
    #왼쪽 위에 표시(0,0)
    showMultiImage(dstimage, frame, height, width, depth, 0, 0)
    #오른쪽 위에 표시(0,1)
    showMultiImage(dstimage, frame1, height, width, depth, 0, 1)


    # 화면 표시
    cv2.imshow('multiView',dstimage)

    #1ms 동안 키입력 대기 ESC키 눌리면 종료
    if cv2.waitKey(1) == 27:
        break;

#윈도우 종료
cap.release()
cv2.destroyWindow('multiView')
