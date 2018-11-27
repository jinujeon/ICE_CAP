import cv2 as cv

# 동영상 파일에 접근하기 위해 VideoCapture 객체를 생성
cap = cv.VideoCapture('output.avi')


while(cap.isOpened()):
    # 동영상 파일로부터 이미지를 가져옴
    ret, frame = cap.read()

    # 화면에 이미지를 출력, 연속적으로 화면에 출력하면 동영상이 됨.
    cv.imshow('frame',frame)

    # ESC 키 누르면 루프 중단
    if cv.waitKey(1) & 0xFF == 27:
        break;

# 모든 자원을 해제함
cap.release()
cv.destroyAllWindows()