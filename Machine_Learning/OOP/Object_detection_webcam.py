######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description:
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a webcam feed.
# It draws boxes and scores around the objects of interest in each frame from
# the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import urllib.request
import time, threading
from multiprocessing import Process
import multiprocessing
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
# from utils import cam_class

def status_handler(location,id,virtual):
    cam = Cam(location,id,virtual)
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(cam.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    video = cv2.VideoCapture(0)
    ret = video.set(3,1280)
    ret = video.set(4,800)

    while(True):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, cam.frame = video.read()
        frame_expanded = np.expand_dims(cam.frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            cam.frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            cam.category_index,
            cam,
            use_normalized_coordinates=True,
            min_score_thresh=0.60)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

def status_handler1(location,id,virtual):
    cam = Cam(location,id,virtual)
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(cam.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    video = cv2.VideoCapture(0)
    ret = video.set(3,1280)
    ret = video.set(4,800)

    while(True):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, cam.frame = video.read()
        frame_expanded = np.expand_dims(cam.frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            cam.frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            cam.category_index,
            cam,
            use_normalized_coordinates=True,
            min_score_thresh=0.60)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

def status_handler2(location,id,virtual):
    cam = Cam(location,id,virtual)
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(cam.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize webcam feed
    video = cv2.VideoCapture(0)
    ret = video.set(3,1280)
    ret = video.set(4,800)

    while(True):
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, cam.frame = video.read()
        frame_expanded = np.expand_dims(cam.frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            cam.frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            cam.category_index,
            cam,
            use_normalized_coordinates=True,
            min_score_thresh=0.60)
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector({})'.format(cam.id), cam.frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

class Cam(threading.Thread):
    def __init__(self,location,id,virtual):
        threading.Thread.__init__(self, name='Cam({})'.format(id))
        self.url = "http://127.0.0.1:8000/home/change_stat"
        self.virtual = virtual
        self.frame = None
        self.id = id
        self.count_warn = 0
        self.count_trash = 0
        self.is_warning = False
        self.is_trash = False
        self.trash_timer = 0
        self.e_list = []
        self.c_list = []
        self.data = {'cam_id': id, 'cam_status': 'safe', 'cam_location': location, 'trash': False,'trusion': False}
        self.time = time.time()
        self.MODEL_NAME = 'inference_graph'
        self.CWD_PATH = os.getcwd()
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH, 'training', 'labelmap.pbtxt')
        self.NUM_CLASSES = 4
        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self. categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

def main():
    cam_info = (['1st_Floor', 0], ['2nd_Floor', 1], ['3rd_Floor', 2])
    procs = []
    multiprocessing.set_start_method('spawn')
    for cam_list in range(3):
        video = cv2.VideoCapture(cam_list)
        if video.isOpened() == True:  # 카메라 생성 확인
            if cam_list == 2:
                virtual = True
                proc = Process(target=status_handler2, args=(cam_info[cam_list][0], cam_info[cam_list][1], virtual))
                procs.append(proc)
                proc.start()
            else:
                virtual = False
                if cam_list == 0:
                    proc = Process(target=status_handler, args=(cam_info[cam_list][0], cam_info[cam_list][1], virtual))
                    procs.append(proc)
                    proc.start()
                else:
                    proc = Process(target=status_handler1, args=(cam_info[cam_list][0], cam_info[cam_list][1], virtual))
                    procs.append(proc)
                    proc.start()

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    # cam_info = (['1st_Floor',0],['2nd_Floor',1],['3rd_Floor',2])
    # for cam_list in range(3):
    #     video = cv2.VideoCapture(cam_list)
    #     if video.isOpened() == True:  # 카메라 생성 확인
    #         a = Cam(cam_info[cam_list][0],cam_info[cam_list][1])
    #         status_handler(a)
    #         # a.start()
    #     else:
    #         print('Can\'t open the CAM(%d)' % (cam_list))
    #         continue
    main()