import cv2
video = cv2.VideoCapture(0)
ret = video.set(3, 640)
ret = video.set(4, 360)
while True:
        ret, frame = video.read()
        cv2.imshow('Object detector()', frame)
        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break