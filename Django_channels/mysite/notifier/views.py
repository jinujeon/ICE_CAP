from django.views.generic import TemplateView
from django.http import HttpResponse
from.models import Camera
from django.views.decorators.csrf import csrf_exempt
import urllib.parse as urlparse
import json
from django.shortcuts import render

class HomeView(TemplateView):
    template_name = "home/home.html"

def alert(request):
	cams = Camera.objects.all()
	context = {'cams': cams}
	return render(request, 'home/alert.html',context)

@csrf_exempt
def change_stat(request):
	cams = Camera.objects.all()
	if request.method == "POST":
		decoded_data = request.read().decode('utf-8')	
		#print(decoded_data)
		dict_data = json.loads(decoded_data)
		print(dict_data)
		cam_id = dict_data['cam_id']
		cam_status = dict_data['cam_status']
		cam_location = dict_data['cam_location']
		print(cam_id, cam_status, cam_location)
		for cam in cams:
			if(cam.cam_id == cam_id):
				cam.cam_status = cam_status
				cam.cam_location = cam_location
			cam.save()
	return HttpResponse("OK")
