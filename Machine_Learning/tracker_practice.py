import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time

from utils import label_map_util
from utils import visualization_utils as vis_util


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
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
global counter, x, y, w, h, trackwindow
x, y, w, h = 0, 0, 0, 0
trackwindow = (x, y, w, h)

counter = 0

# Setup the termination critera, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.5)

'''
def tracker(y, x, w, h, frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
'''

def status_handler():
    sys.path.insert(0, 'C:/tensorflow1/models/research/object_detection')
    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'inference_graph'

    # Grab path to current working directory
    CWD_PATH = 'C:/tensorflow1/models/research/object_detection'

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 90

    ## Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
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
        global w, h, x, y, trackwindow, counter
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        # Draw the results of the detection (aka 'visulaize the results')
        #print(classes)

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=0.60)
        # All the results have been drawn on the frame, so it's time to display it.
        for i, b in enumerate(boxes[0]):
         if scores[0][i] >= 0.6:
                 mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                 mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                 cv2.putText(frame, 'M', (int(mid_x * 1280), int(mid_y * 720)),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                 if classes[0][i] in category_index.keys():
                     class_name = category_index[classes[0][i]]['name']
                 if (class_name == 'person'):
                     if counter%30 == 0:
                         w = (int)((boxes[0][i][3] - boxes[0][i][1]) * 1280)
                         h = (int)((boxes[0][i][2] - boxes[0][i][0]) * 800)
                         x = (int)((boxes[0][i][1]) * 1280)
                         y= (int)((boxes[0][i][0]) * 800)
                         trackwindow = (x, y, w, h)
                         roi = frame[x:x + h, y:y + w]
                         hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                         mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                         roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                         cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                         # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
                         term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                         counter = counter + 1
                     else:
                         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                         dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                         # apply meanshift to get the new location
                         ret, trackwindow = cv2.CamShift(dst, trackwindow, term_crit)
                         # Draw it on image
                         pts = cv2.boxPoints(ret)
                         pts = np.int0(pts)
                         frame = cv2.polylines(frame, [pts], True, 255, 2)
                         counter = counter + 1


        #0.03, 0.916 & 0.917, 0.52
        #0.1 * 1280 & 0.1 * 720 = 128 & 72
        colist = [38, 660, 1174, 374]
        cv2.line(frame, (colist[0], colist[1]), (colist[2], colist[3]), (255,0,0), 2)
        cv2.imshow('Object detector', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    status_handler()