# -*- coding: utf-8 -*-
__author__ = 'Seran'

import cv2

def im_trim (img): #함수로 만든다
    x = 10; y = 10 #자르고 싶은 지점의 x좌표와 y좌표 지정
    w = 100; h = 100 #x로부터 width, y로부터 height를 지정
    img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
    cv2.imwrite("./frame%d.jpg" % count, img_trim) #org_trim.jpg 라는 이름으로 저장
    print("saved")
    return img_trim #필요에 따라 결과물을 리턴

# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
# vidcap = cv2.VideoCapture('/pedestrians.avi')

video_src = 'pedestrians.avi'

vidcap = cv2.VideoCapture(video_src)

count = 0

while (vidcap.isOpened()):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = vidcap.read()



    # 캡쳐된 이미지를 저장하는 함수
    trim_image = im_trim(image)  # trim_image 변수에 결과물을 넣는다

    print('Saved frame%d.jpg' % count)
    count += 1

