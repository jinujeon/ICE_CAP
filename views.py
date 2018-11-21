from django.shortcuts import render
import cv2, time
import numpy as np
from django.http import StreamingHttpResponse, request
import threading
from django.views.decorators.gzip import gzip_page
import gzip
from time import sleep
from . import image_frame_store

class VideoCamera(object):
    def __init__(self):
        self.k = True
        self.index = 0
        # self.video = cv2.VideoCapture(0)
        # (self.grabbed, self.frame) = self.video.read()
        # threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        # image = self.frame
        try :
            name = '/img_{}.png'.format(cam.index)
            image = cv2.imread('C:/Users/ruldy/Roaming/' + name)
            ret, jpeg = cv2.imencode('.jpg', image)
        except:
            if cam.index == 0:
                cam.index = 4
            else: cam.index -= 1
            name = '/img_{}.png'.format(cam.index)
            image = cv2.imread('C:/Users/ruldy/Roaming/' + name)
            ret, jpeg = cv2.imencode('.jpg', image)
        else:
            cam.index += 1

        if cam.index % 5 == 0:
            cam.index = 0
        return jpeg.tobytes()

    # def update(self):
    #     while True:
    #         (self.grabbed, self.frame) = self.video.read()

cam = VideoCamera() #서버 실행시 최초 1회만 실행

def gen(camera):
    # cam = VideoCamera()
    # if cam.k == False:
    #     cam.__init__()
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



#@gzip.gzip_page
def livefe(request):
    try:
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except ConnectionAbortedError as e:
        print(e)


def homeview(request):
    return render(request,'blog/alert.html')

def request_page(request):
    # cam.__del__()
    # cam.k = False
    cam.index = 0
    return render(request, 'blog/indextest.html')



