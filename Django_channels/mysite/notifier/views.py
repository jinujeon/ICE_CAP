from django.views.generic import TemplateView
from django.http import HttpResponse
from .models import Camera, Profile
from django.views.decorators.csrf import csrf_exempt
import urllib.parse as urlparse
import json
from django.shortcuts import render
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import cv2
import numpy as np
from django.http import StreamingHttpResponse
import threading
from django.views.decorators.gzip import gzip_page
import gzip


class HomeView(TemplateView):
   template_name = "home/home.html"




# def homeview(request):
#     # profile = Profile.objects.all()
#     return render(request, 'home/home.html')


# def alert(request):
#     cams = Camera.objects.all()
#     context = {'cams': cams}
#     return render(request, 'home/alert.html', context)
class AlertView(TemplateView):
    def get(self,request,*args, **kwargs):
        cams = Camera.objects.all()
        context = {'cams': cams}
        return render(request, 'home/alert.html', context)


@csrf_exempt
def change_stat(request):
    global op
    cams = Camera.objects.all()
    if request.method == "POST":
        decoded_data = request.read().decode('utf-8')
        # print(decoded_data)
        dict_data = json.loads(decoded_data)
        print("#####Dict######:", dict_data)
        cam_id = dict_data['cam_id']
        cam_status = dict_data['cam_status']
        cam_location = dict_data['cam_location']
        cam_fallen = dict_data['fallen']
        cam_trash = dict_data['trash']
        cam_instrusion = dict_data['instrusion']
        print("----values=====:", cam_id, cam_status, cam_location, cam_trash, cam_instrusion, cam_fallen)
        for cam in cams:
            if (cam.cam_id == cam_id):
                cam.cam_status = cam_status
                cam.cam_location = cam_location
                cam.fallen = cam_instrusion
                cam.trash = cam_trash
                cam.instrusion = cam_instrusion
            cam.save()

    return HttpResponse("OK")

class VideoCamera(object):
    def __init__(self):
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


cam = VideoCamera()


def gen(camera):
    while True:
        frame = cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# @gzip.gzip_page
def livefe(request):
    try:
        return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        pass
# def update_profile(request, user_id):
#    user = User.objects.get(pk=user_id)
#    user.save()
