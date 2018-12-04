import numpy as np
import cv2

video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)
video_capture_0.set(cv2.CAP_PROP_FPS, 0.1)
video_capture_1.set(cv2.CAP_PROP_FPS, 0.1)
# exec('self.width = int(self.video_capture_{}.get(cv2.CAP_PROP_FRAME_WIDTH))'
#                  '\nself.height = int(self.video_capture_{}.get(cv2.CAP_PROP_FRAME_HEIGHT))'
#                  '\nprint(self.width, self.height)'.format(id,id))
while True:
    # Capture frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if (ret0):
        # Display the resulting frame
        cv2.imshow('Cam 0', frame0)

    if (ret1):
        # Display the resulting frame
        cv2.imshow('Cam 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()