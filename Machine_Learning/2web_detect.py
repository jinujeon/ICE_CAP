import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time

from utils import label_map_util
from utils import visualization_utils as vis_util


def create_image(h, w, d):
    image = np.zeros((h, w, d), np.uint8)
    color = tuple(reversed((0, 0, 0)))
    image[:] = color
    return image


def create_image_multiple(h, w, d, hcout, wcount):
    image = np.zeros((h * hcout, w * wcount, d), np.uint8)
    color = tuple(reversed((0, 0, 0)))
    image[:] = color
    return image


def showMultiImage(dst, src, h, w, d, col, row):
   dst[(col * h):(col * h) + h, (row * w):(row * w) + w] = src[0:h, 0:w]



def status_handler():
    # 현재 폴더의 디렉토리를 저장할수있게 추가합니다
    sys.path.append("..")
    # 객체 인식을 실행하는 프로그램의 디렉토리를 저장하여 프로그램을 실행하기 위한 유틸 파일을 사용할수 있게 설정합니다
    sys.path.insert(0, 'C:/tensorflow1/models/research/object_detection')
    # 학습 모델이 저장되어 있는 폴더의 이름을 저장합니다.
    MODEL_NAME = 'inference_graph'
    # 작업 중인 디렉토리를 저장합니다
    CWD_PATH = 'C:/tensorflow1/models/research/object_detection'

    # 학습 모델 파일을 지정합니다
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # 학습 모델을 사용하여 구분할 class명이 저장되어있는 폴더와 파일을 지정합니다.
    PATH_TO_LABELS = os.path.join(CWD_PATH,'labelmap.pbtxt')
    # 인식을 할 가짓수는 2가지입니다. (안전[Person], 위험[Warning])
    NUM_CLASSES = 90
    # 객체의 라벨과 클래스를 연관지어 저장합니다.
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # 머신 러닝한 모델을 사용하기 위해 텐서플로우를 메모리에 로드합니다.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    # 영상의 한 프레임마다 나오는 이미지에 대해 텐서를 지정합니다.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # 출력하는 텐서를 지정합니다.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # 영상 분석을 하여 나온 해당 객체의 이름(class)과 정밀도(scores)를 지정합니다.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # 총 인식된 객체의 수를 지정합니다.
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # 연결된 카메라의 초기설정

    ##### 코드 시작 ####
    video = cv2.VideoCapture(0)  # 카메라 생성
    video2 = cv2.VideoCapture(1)
    ret = video.set(3, 640)
    ret1 = video2.set(3, 640)

    if video.isOpened() == False:  # 카메라 생성 확인
        print('Can\'t open the CAM(%d)' % (0))
        exit()
    elif video2.isOpened() == False:  # 카메라 생성 확인
        print('Can\'t open the CAM(%d)' % ())
        exit()
    else:
        print('all clear')

    # 윈도우 생성 및 사이즈 변경
    cv2.namedWindow('multiView')

    present = time.time()
    timer = time.time()


    while (True):
        if int(timer - present) >= 3:
            print('1번 카메라')
            # 카메라에서 이미지 얻기
            ret, frame = video.read()
            ret1, frame1 = video2.read()

            # 이미지 높이
            height = frame.shape[0]

            # 이미지 넓이
            width = frame.shape[1]

            # 이미지 색상 크기
            depth = frame.shape[2]


            frame_expanded = np.expand_dims(frame, axis=0)
            # 객체에 씌울 경계선, 정밀도, 이름, 확률 값을 텐서에 넣어 지정합니다.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            # 유틸 프로그램에 해당 변수를 넣어 나온 결과물들을 저장합니다.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.60)  # 객체의 정밀도가 60% 이상일때만 화면에 표시합니다

            for i, b in enumerate(boxes[0]):
                if scores[0][i] >= 0.6:
                    mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                    mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                    cv2.putText(frame, "Core", (int(mid_x * 640), int(mid_y * 360)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)

            dstimage = create_image_multiple(height, width, depth, 1, 2)

            # 원하는 위치에 복사
            # 왼쪽 위에 표시(0,0)
            showMultiImage(dstimage, frame, height, width, depth, 0, 0)
            # 오른쪽 위에 표시(0,1)
            showMultiImage(dstimage, frame1, height, width, depth, 0, 1)
            timer = time.time()
            print(present)

        if int(timer - present) >= 4.5:
            print('2번카메라')
            frame_expanded1 = np.expand_dims(frame1, axis=0)
            # 객체에 씌울 경계선, 정밀도, 이름, 확률 값을 텐서에 넣어 지정합니다.
            (boxes_two, scores_two, classes_two, num_two) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded1})

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame1,
                np.squeeze(boxes_two),
                np.squeeze(classes_two).astype(np.int32),
                np.squeeze(scores_two),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.60)  # 객체의 정밀도가 60% 이상일때만 화면에 표시합니다
            # 최종 출력물을 이미지에 씌워 출력합니다.
            # 각 객체들의 core 좌표를 opencv를 통해 출력

            for i, b in enumerate(boxes[0]):
                if scores[0][i] >= 0.6:
                    mid_x_two = (boxes_two[0][i][1] + boxes_two[0][i][3]) / 2
                    mid_y_two = (boxes_two[0][i][0] + boxes_two[0][i][2]) / 2
                    cv2.putText(frame1, "Core", (int(mid_x_two * 640), int(mid_y_two * 400)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 화면에 표시할 이미지 만들기 ( 2 x 2 )

            dstimage = create_image_multiple(height, width, depth, 1, 2)

            # 원하는 위치에 복사
            # 왼쪽 위에 표시(0,0)
            showMultiImage(dstimage, frame, height, width, depth, 0, 0)
            # 오른쪽 위에 표시(0,1)
            showMultiImage(dstimage, frame1, height, width, depth, 0, 1)
            timer = time.time()
            print(present)

            if int(timer - present) >= 6:  # 객체 인식 3초 지속 후 초기화
                present = timer

            # 화면 표시
            cv2.imshow('multiView', dstimage)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print('3초이후')
            timer = time.time()  # 타이머를 갱신하며 3초를 count
            ret, frame = video.read()  # 타이머를 갱신중엔 객체 인식을 하지 않는 프레임 출력
            ret1, frame1 = video2.read()

            # 이미지 높이
            height = frame.shape[0]

            # 이미지 넓이
            width = frame.shape[1]

            # 이미지 색상 크기
            depth = frame.shape[2]


            # 화면에 표시할 이미지 만들기 ( 2 x 2 )

            dstimage = create_image_multiple(height, width, depth, 1, 2)

            # 원하는 위치에 복사
            # 왼쪽 위에 표시(0,0)
            showMultiImage(dstimage, frame, height, width, depth, 0, 0)
            # 오른쪽 위에 표시(0,1)
            showMultiImage(dstimage, frame1, height, width, depth, 0, 1)


            # 화면 표시
            cv2.imshow('multiView', dstimage)
            if cv2.waitKey(1) == ord('q'):
                break

    # 윈도우 종료
    video.release()
    cv2.destroyWindow('multiView')


if __name__ == '__main__':
    status_handler()