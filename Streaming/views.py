from django.shortcuts import render
import cv2, time
import numpy as np
from django.http import StreamingHttpResponse, request, HttpResponse
import threading
from django.views.decorators.gzip import gzip_page
import gzip
from time import sleep
import sys
sys.path.insert(0, 'path/to/C:/Users/ruldy/Roaming')

# import store_frame

class VideoCamera(object):
    def __init__(self):
        self.k = True
        self.index = 0
        self.slep = 0
        # self.video = cv2.VideoCapture(0)
        # (self.grabbed, self.frame) = self.video.read()
        # threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self, cm_id):
        try:
            name = '/img_{}.png'.format(cam.index)
            image = cv2.imread('C:/Users/ruldy/Roaming/'+str(cm_id) + name)
            ret, jpeg = cv2.imencode('.jpg', image)
        except:
            if cam.index == 0:
                cam.index = 4
            else:
                cam.index -= 1
            name = '/img_{}.png'.format(cam.index)
            image = cv2.imread('C:/Users/ruldy/Roaming/'+str(cm_id) + name)
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
        cam.slep += 1
        if cam.slep % 10 == 0:
            frame = cam.get_frame(2)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else :
            pass

def gen2(camera):
    # cam = VideoCamera()
    # if cam.k == False:
    #     cam.__init__()
    while True:
        cam.slep += 1
        if cam.slep % 10 == 0:
            frame = cam.get_frame(1)
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else :
            pass



#@gzip.gzip_page
def livefe(request):
    try:
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except ConnectionAbortedError as e:
        print(e)

#@gzip.gzip_page
def livefe2(request):
    try:
        return StreamingHttpResponse(gen2(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except ConnectionAbortedError as e:
        print(e)


def homeview(request):
    return render(request,'blog/index.html')

def request_page(request):
    # cam.__del__()
    # cam.k = False
    cam.index = 0
    return render(request, 'blog/indextest.html')

