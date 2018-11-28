import cv2, time

class VideoCamera(object):
    def __init__(self):
        self.time = 0
        self.clock = time.gmtime(time.time()) #동영상 이름 -> 현재시간
        self.now = str(self.clock.tm_year) +'.'+ str(self.clock.tm_mon) +'.'+ str(self.clock.tm_mday) +'.'+ str(self.clock.tm_hour + 9) +'.'+ str(self.clock.tm_min) +'.'+ str(self.clock.tm_sec)
        # 카메라에 접근하기 위해 VideoCapture 객체를 생성
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        self.out = 0

    def __del__(self):
        self.video.release()
        self.out.release()
        cv2.destroyAllWindows()

    def write(self):
        # 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 파일에 저장하기 위해 VideoWriter 객체를 생성
        self.out = cv2.VideoWriter('output'+self.now+'.avi', fourcc, 15, (640, 480))


    def getframe(self):
        self.time += 1
        (self.grabbed, self.frame) = self.video.read()
        # self.ret, self.frame_encode = cv2.imencode('.png', self.frame, [(cv2.IMWRITE_PNG_COMPRESSION), 0])
        # self.frame = cv2.imdecode(self.frame_encode, 1)


    def storeframe(self):
        self.out.write(self.frame)

store = VideoCamera() # 영상 저장을 위한 객체 생성
store.write() # 영상저장함수실행

while True:

    # 카메라로부터 이미지를 가져옴
    store.getframe()
    # 캡쳐하는데 문제가 있으면 루프 중단
    if store.grabbed == False:
        break;

    # 이미지를 파일에 저장, VideoWriter 객체에 연속적으로 저장하면 동영상이 됨.
    if store.time % 4 == 0:
        store.storeframe()

    print(store.time) # 저장시간

    if store.time == 8000:
        store.__del__() # 현재까지 영상 저장
        store = VideoCamera()  # 영상 저장을 위한 객체 재생성
        store.write() # 영상저장함수실행
        pass

    # 화면에 이미지를 출력, 연속적으로 화면에 출력하면 동영상이 됨.
    cv2.imshow('frame', store.frame)

    # ESC 키 누르면 루프 중단
    if cv2.waitKey(1) & 0xFF == 27:
        break;
