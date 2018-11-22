import collections
import functools
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import time
import tensorflow as tf
import json
import urllib.request
from object_detection.core import standard_fields as fields
import urllib.request
import intr_detect as intr
import multitracker

_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  if use_normalized_coordinates:
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
  else:
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default()

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)
  # mid_x = (right+left)/2
  # mid_y = (top+bottom)/2


  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.

  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)
    text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')

  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)



def _visualize_boxes(image, boxes, classes, scores, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image, boxes, classes, scores, category_index=category_index, **kwargs)


def _visualize_boxes_and_masks(image, boxes, classes, scores, masks,
                               category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      **kwargs)


def _visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints,
                                   category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      keypoints=keypoints,
      **kwargs)


def _visualize_boxes_and_masks_and_keypoints(
    image, boxes, classes, scores, masks, keypoints, category_index, **kwargs):
  return visualize_boxes_and_labels_on_image_array(
      image,
      boxes,
      classes,
      scores,
      category_index=category_index,
      instance_masks=masks,
      keypoints=keypoints,
      **kwargs)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         instance_masks=None,
                                         keypoints=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored.
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  # Additional channels are being ignored.
  images = images[:, :, :, 0:3]
  visualization_keyword_args = {
      'use_normalized_coordinates': use_normalized_coordinates,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 4
  }

  if instance_masks is not None and keypoints is None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [images, boxes, classes, scores, instance_masks]
  elif instance_masks is None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [images, boxes, classes, scores, keypoints]
  elif instance_masks is not None and keypoints is not None:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes_and_masks_and_keypoints,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [images, boxes, classes, scores, instance_masks, keypoints]
  else:
    visualize_boxes_fn = functools.partial(
        _visualize_boxes,
        category_index=category_index,
        **visualization_keyword_args)
    elems = [images, boxes, classes, scores]

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections,
                                  tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True):
  """Creates a side-by-side image with detections and groundtruth.

  Bounding boxes (and instance masks, if available) are visualized on both
  subimages.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_single_example().
    category_index: A category index (dictionary) produced from a labelmap.
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    A [1, H, 2 * W, C] uint8 tensor. The subimage on the left corresponds to
      detections, while the subimage on the right corresponds to groundtruth.
  """
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()
  instance_masks = None
  if detection_fields.detection_masks in eval_dict:
    instance_masks = tf.cast(
        tf.expand_dims(eval_dict[detection_fields.detection_masks], axis=0),
        tf.uint8)
  keypoints = None
  if detection_fields.detection_keypoints in eval_dict:
    keypoints = tf.expand_dims(
        eval_dict[detection_fields.detection_keypoints], axis=0)
  groundtruth_instance_masks = None
  if input_data_fields.groundtruth_instance_masks in eval_dict:
    groundtruth_instance_masks = tf.cast(
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_instance_masks], axis=0),
        tf.uint8)
  images_with_detections = draw_bounding_boxes_on_image_tensors(
      eval_dict[input_data_fields.original_image],
      tf.expand_dims(eval_dict[detection_fields.detection_boxes], axis=0),
      tf.expand_dims(eval_dict[detection_fields.detection_classes], axis=0),
      tf.expand_dims(eval_dict[detection_fields.detection_scores], axis=0),
      category_index,
      instance_masks=instance_masks,
      keypoints=keypoints,
      max_boxes_to_draw=max_boxes_to_draw,
      min_score_thresh=min_score_thresh,
      use_normalized_coordinates=use_normalized_coordinates)
  images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
      eval_dict[input_data_fields.original_image],
      tf.expand_dims(eval_dict[input_data_fields.groundtruth_boxes], axis=0),
      tf.expand_dims(eval_dict[input_data_fields.groundtruth_classes], axis=0),
      tf.expand_dims(
          tf.ones_like(
              eval_dict[input_data_fields.groundtruth_classes],
              dtype=tf.float32),
          axis=0),
      category_index,
      instance_masks=groundtruth_instance_masks,
      keypoints=None,
      max_boxes_to_draw=None,
      min_score_thresh=0.0,
      use_normalized_coordinates=use_normalized_coordinates)
  return tf.concat([images_with_detections, images_with_groundtruth], axis=2)


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    cam,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  cam.e_list = []
  cam.fxy_list= []
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  # data = {'cam_id': 1, 'alert': 'safe', 'trash': False}
  # data = {'cam_id': 1, 'cam_status': 'safe', 'cam_location': 'Engineering_Univ.1st.right_corridor', 'trash': False, 'intrusion':False}
  # for i, b in enumerate(boxes):
  #     if scores[i] >= 0.6:
  #         mid_x = (boxes[i][1] + boxes[i][3]) / 2
  #         mid_y = (boxes[i][0] + boxes[i][2]) / 2
  #         # cv2.putText(frame, 'M', (int(mid_x * 1280), int(mid_y * 720)),
  #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
  #         print(mid_x)
  #         print(mid_y)

  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None: # 인식하는 개체에 필수 변수가 존재하지 않는다면 재설정합니다.
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            # print(category_index)
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
              if class_name == 'person':
                  x, y = boxes[0][i][1], boxes[0][i][0]
                  w, h = boxes[0][i][3] - boxes[0][i][1], boxes[0][i][2] - boxes[0][i][0]
                  coord = (x, y, w, h)
                  cam.fxy_list.append(coord)
              cam.e_list.append(class_name) # 해당 화면에서 인식된 개체의 이름을 리스트에 저장합니다
            else:
              class_name = 'N/A' # 해당 화면에 인식된 객체의 이름이 존재하지 않는다면
            display_str = str(class_name) # 그 전까지 인식한 객체들만 표시

        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    if instance_masks is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
    if instance_boundaries is not None:
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  if cam.data['instrusion']:
      if 'person' in cam.e_list:
          fence = intr.fence()
          trackers = multitracker.multitracker()
          fence.fence_check(cam.fxy_list, cam.frame)
          if (fence.fence_warning):
              print("제한 구역 침입을 감지했습니다. 알림을 전송합니다.")
              if cam.data['instrusion'] == False:
                  cam.data['cam_status'] = 'warning'
                  cam.data['instrusion'] = True
                  params = json.dumps(cam.data).encode("utf-8")
                  req = urllib.request.Request(cam.url, data=params,
                                               headers={'content-type': 'application/json'})
                  response = urllib.request.urlopen(req)
                  print(response.read().decode('utf8'))
              fence.fence_warning = False
          else:
              print("해당 구역은 안전합니다.")
              if cam.data['instrusion'] == True:
                  cam.data['cam_status'] = 'safe'
                  cam.data['instrusion'] = False
                  params = json.dumps(cam.data).encode("utf-8")
                  req = urllib.request.Request(cam.url, data=params,
                                               headers={'content-type': 'application/json'})
                  response = urllib.request.urlopen(req)
                  print(response.read().decode('utf8'))

          if trackers.isFirst or (cam.count_tracking % 15 == 0):
              try:
                  trackers.settings(cam.fxy_list, cam.frame)
                  # now = 0
                  cam.count_tracking += 1
              except Exception as e:
                  print(str(e))
                  pass
          elif len(cam.fxy_list) != 0:
              try:
                  trackers.updatebox(cam.frame)
                  cam.count_tracking += 1
              except Exception as e:
                  print(str(e))
                  pass
          else:
              trackers.isFirst = True
              cam.count_tracking = 0

  # 쓰레기가 감지되었다면 감지된 시각을 초기화하고 그로부터 30분동안 지속적으로 감지 되었을때 쓰레기가 투기되었다고 생각하여 알림을 전송한다
  # 이따금씩 감지가 안될 때를 대비하여 별도의 카운터 변수를 선언, 10초의 기간을 두어 쓰레기가 실제로 사라졌는지를 확인한다.
  if 'trash' in cam.e_list or 'bottle' in cam.e_list: # 쓰레기를 감지했다면
      cam.trash_timer = 0 # 간헐적인 오감지를 방지하기 위해 카운터를 초기화하고
      trash_count(True,cam) # 10분동안 지속적으로 쓰레기가 감지되는지를 확인하기 위해 감지된 시각을 초기화합니다.
      trash_check(cam) # 실시간으로 시간을 재어 10분동안 쓰레기가 감지되면 상태를 업데이트합니다.
      if (int(time.time()) - cam.count_trash) % 10 == 0 and cam.count_trash != 0: # 10초에 한번씩 알림 발신
        print("쓰레기가 발견되었습니다. {}초 경과". format(int(time.time()) - cam.count_trash))
  else:
      if cam.is_trash:
          if cam.trash_timer == 0:
              cam.trash_timer = int(time.time())
          if int(time.time()) - cam.trash_timer > 10:
            print("쓰레기가 없습니다.")
            trash_count(False,cam)  # 없다면 카운터 초기화 및 현 상황 변동 확인
            trash_check(cam) # 확인된 위험이 없으므로 현재 CCTV의 상태를 return 합니다.
          else: print("사라진 쓰레기를 찾고있습니다. {}초 후 상태를 변경합니다.".format(10 - (int(time.time()) - cam.trash_timer)))

  if 'warning' in cam.e_list: # 해당 화면에 인식된 개체의 이름을 저장한 리스트에서 위험한 상황에 처한 객체가 있는지 확인합니다.
      emergency_count(True,cam)  # 있다면 카운터 증가
      emergency_check(cam)
      if (int(time.time()) - cam.count_fallen) < 10: # 10초의 시간을 기다린 후에도 위험한 상황이 인식된다면
        print("위험 상황을 확인했습니다. {}초 후 알림을 전송합니다.".format(10 - (int(time.time()) - cam.count_fallen)))
      else: # 알림 전송
          print("알림을 전송합니다. 위험 상황 발생 {}초 경과".format(int(time.time()) - cam.count_fallen))
      return image
  else:
      emergency_count(False,cam)  # 없다면 카운터 초기화 및 현 상황 변동 확인
      emergency_check(cam) # 확인된 위험이 없으므로 현재 CCTV의 상태를 return 합니다.
      print("확인된 위험이 없습니다. 위험상태: {}".format(cam.is_fallen))
      return image

def trash_count(stat,cam):
    if stat:  # 쓰레기 감지
        print("is_trash:{}".format(cam.is_trash))
        if not cam.is_trash: # 처음으로 쓰레기를 감지하였을때의 시간을 저장하고 상태를 변경
            cam.count_trash = int(time.time())
            cam.is_trash = True
    else: # 쓰레기가 없다고 판단
        cam.count_trash = 0 # 카운터 변수 초기화
        if cam.data['trash'] == True:
            cam.data['cam_status'] = 'safe'
            cam.data['trash'] = False  # 쓰레기가 없다고 판단하여 CCTV의 위험 상태를 safe로 바꿉니다
            params = json.dumps(cam.data).encode("utf-8")
            req = urllib.request.Request(cam.url, data=params,
                                         headers={'content-type': 'application/json'})
            response = urllib.request.urlopen(req)
            print(response.read().decode('utf8'))
        cam.is_trash = False  # CCTV의 상태 위험하지 않음으로 변경

def trash_check(cam):
    if cam.count_trash != 0:
        timer = int(time.time()) # 현재 시각을 확인하여 최초 쓰레기가 발견된 시각과 비교
    else: timer = 0
    if ((timer - cam.count_trash) % 60) == 0:
        print("알림을 전송합니다. 쓰레기 감지 {}분 경과".format(int((int(time.time()) - cam.count_trash) / 60)))
    if timer - cam.count_trash >= 600: # 일정 시간(10분) 이후에도 지속적인 위험 상황 확인
        if cam.data['trash'] == False:
            cam.data['cam_status'] = 'warning'
            cam.data['trash'] = True # 현재 CCTV 위치에 버려진 쓰레기가 감지되었으므로 상태를 변경합니다.
            params = json.dumps(cam.data).encode("utf-8")
            req = urllib.request.Request(cam.url, data=params,
                                         headers={'content-type': 'application/json'})
            response = urllib.request.urlopen(req)
            print(response.read().decode('utf8'))
            # 쓰레기가 CCTV에서 10분 이상 지속적으로 카운트 되었다.

def emergency_count(stat,cam): # 위험한 상태 확인, 일정 시간을 대기하기 위해 카운트하는 함수
    if stat: # 위험한 상황을 인식하였으므로 카운트 1 증가
        if not cam.is_fallen:
            cam.count_fallen = int(time.time())
            cam.is_fallen = True
    else:
        cam.count_fallen = 0
        if cam.data['fallen'] == True:
            cam.data['cam_status'] = 'safe' # 위험 상태가 없다고 판단하여 CCTV의 위험 상태를 safe로 바꿉니다
            cam.data['fallen'] = False
            params = json.dumps(cam.data).encode("utf-8")
            req = urllib.request.Request(cam.url, data=params,
                                         headers={'content-type': 'application/json'})
            response = urllib.request.urlopen(req)
            print(response.read().decode('utf8'))
        cam.is_fallen = False # CCTV의 상태 안전으로 변경

def emergency_check(cam):
    if cam.count_fallen != 0:
        timer = int(time.time())
    else: timer = 0
    if timer - cam.count_fallen > 10: # 일정 시간 이후에도 지속적인 위험 상황 확인
        if cam.data['fallen'] == False:
            cam.data['cam_status'] = 'warning' # 현재 위치에 있는 CCTV의 위험 상태를 위험으로 바꿉니다.
            cam.data['fallen'] = True
            params = json.dumps(cam.data).encode("utf-8")
            req = urllib.request.Request(cam.url, data=params,
                                         headers={'content-type': 'application/json'})
            response = urllib.request.urlopen(req)
            print(response.read().decode('utf8'))