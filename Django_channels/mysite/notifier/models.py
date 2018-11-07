from django.db import models

# Create your models here.

class Camera(models.Model):
	cam_id = models.IntegerField(default=0)
	cam_status = models.CharField(max_length=20)
	cam_location = models.CharField(max_length=50)


class Trash(models.Model):
	trash = models.BooleanField(default=False)
