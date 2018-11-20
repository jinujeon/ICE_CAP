from django.shortcuts import render
import cv2
import numpy as np
from django.http import StreamingHttpResponse, request
import threading
from django.views.decorators.gzip import gzip_page
import gzip

class VideoCamera(object):
    def __init__(self):
        self.k = True
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

cam = VideoCamera() #서버 실행시 최초 1회만 실행

def gen(camera):
    # cam = VideoCamera()
    if cam.k == False:
        cam.__init__()
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


#@gzip.gzip_page
def livefe(request):
    try:
        return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    except ConnectionAbortedError as e:
        print(e)

def homeview(request):
    return render(request,'blog/index.html')

def request_page(request):
    cam.__del__()
    cam.k = False
    return render(request, 'blog/indextest.html')



