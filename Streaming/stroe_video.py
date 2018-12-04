import cv2, time

class VideoCamera(object):
    def __init__(self, idx):
        self.time = time.time()     #동영상 촬영시간 측정
        self.sizecontrol = 0        #동영상 용량 조절
        self.clock = time.gmtime(time.time()) #동영상 이름 -> 현재시간
        self.name = str(self.clock.tm_year) +'.'+ str(self.clock.tm_mon) +'.'+ str(self.clock.tm_mday) +'.'+ str(self.clock.tm_hour + 9) +'.'+ str(self.clock.tm_min) +'.'+ str(self.clock.tm_sec)
        self.videooutput = 0        #동영상 녹화 변수 정의
        # 카메라에 접근하기 위해 VideoCapture 객체를 생성
        self.video = cv2.VideoCapture(idx)
        # exec('self.video{} = cv2.VideoCapture(idx)'.format(idx, idx))
        (self.grabbed, self.frame) = self.video.read()

    def __del__(self):
        # 현재까지의 녹화를 멈춘다.
        self.video.release()
        self.videooutput.release()
        cv2.destroyAllWindows()

    def write(self):
        # 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # 파일에 저장하기 위해 VideoWriter 객체를 생성
        self.videooutput = cv2.VideoWriter('output'+self.name+'.avi', fourcc, 10, (640, 480))

    def sizecon(self):
        # 동영상 용량 조정
        self.sizecontrol += 1
        if self.sizecontrol == 8000:
            self.sizecontrol = 0

    def storeframe(self, frame):
        # 동영상 프레임 실제 저장
        self.videooutput.write(frame)

store = VideoCamera(0) # 영상 저장을 위한 객체 생성
store.write() # 영상저장함수실행

while True:

    store.grabbed, store.frame = store.video.read()

    # 카메라로부터 이미지를 가져옴
    store.sizecon()
    # 캡쳐하는데 문제가 있으면 루프 중단
    if store.grabbed == False:
        break;

    # 이미지를 파일에 저장, VideoWriter 객체에 연속적으로 저장하면 동영상이 됨.
    if store.sizecontrol % 4 == 0:
        store.storeframe(store.frame)

    print(store.time) # 저장시간

    if (time.time() - store.time) > 10:
        store.__del__() # 현재까지 영상 저장
        store = VideoCamera(0)  # 영상 저장을 위한 객체 재생성
        store.write() # 영상저장함수실행

    # 화면에 이미지를 출력, 연속적으로 화면에 출력하면 동영상이 됨.
    cv2.imshow('frame', store.frame)

    # ESC 키 누르면 루프 중단
    if cv2.waitKey(1) & 0xFF == 27:
        break;
