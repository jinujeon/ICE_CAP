
from .models import Camera
from django.db.models.signals import post_save
from django.dispatch import receiver
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from. import send

@receiver(post_save, sender=Camera)
def announce_cam_stat(sender, instance, created, **kwargs):
    if created:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "cameras", {"type": "cam_message",
                       "event": "created",
                       "cam_id": instance.cam_id,
			"cam_location": instance.cam_location,
                        "trash": instance.trash,
                        "instrusion": instance.instrusion,
                        "fallen": instance.fallen
                        })
    elif instance.cam_status == 'warning':

        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "cameras", {"type": "cam_message",
                       "event": "warning",
                       "cam_id": instance.cam_id,
			"cam_location": instance.cam_location,
                        "trash": instance.trash,
                        "instrusion": instance.instrusion,
                        "fallen": instance.fallen})
