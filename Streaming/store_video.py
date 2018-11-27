import cv2

# 카메라에 접근하기 위해 VideoCapture 객체를 생성

cap = cv2.VideoCapture(0)

# 코덱 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 파일에 저장하기 위해 VideoWriter 객체를 생성
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

while (cap.isOpened()):

    # 카메라로부터 이미지를 가져옴

    ret, frame = cap.read()

    # 캡쳐하는데 문제가 있으면 루프 중단
    if ret == False:
        break;

    # 이미지의 상하를 뒤집음
    # 이미지가 거꾸로 보인다면 주석처리
    #frame = cv2.flip(frame, 0)

    # 이미지를 파일에 저장, VideoWriter 객체에 연속적으로 저장하면 동영상이 됨.
    out.write(frame)

    # 화면에 이미지를 출력, 연속적으로 화면에 출력하면 동영상이 됨.

    cv2.imshow('frame', frame)

    # ESC 키 누르면 루프 중단
    if cv2.waitKey(1) & 0xFF == 27:
        break;

# 모든 자원을 해제함
cap.release()
out.release()
cv2.destroyAllWindows()
