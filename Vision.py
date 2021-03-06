####Object detection using Intel RealSense L515###
    #%% Stream from intel realsense
    
    # First import the library
import pyrealsense2 as rs

## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import numpy as np
import cv2
#import open3d as o3d
#import matplotlib.pyplot as plt

import os
import six.moves.urllib as urllib
#import sys
import tarfile
import tensorflow as tf
import csv

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# import rospy
# from std_msgs.msg import float64MultiArray
# from std_msgs.msg import Bool

# def Publish(data):
#     pub = rospy.Publisher(1.000, float64MultiArray, queue_size = 10)
#     rospy.init_mode('Publish_node', anonymous=True)
#     rate = rospy.Rate(10)
#     while not rospy.is_shutdown():
#         PublishFloat = data
#         rospy.loginfo(PublishFloat)
#         pub.publish(PublishFloat)
#         rate.sleep()

# def callback(data):
#     rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

# def Subscribe():
#     rospy.init_mode(False, anonymous = True)
#     rospy.Subscriber('Noe', Bool, callback)
#     rospy.spin()
    
os.chdir("C:/Users/jhesj/models/research/object_detection")

#Different models were tested
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())
        

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]  # change this value if you want to add more pictures to test

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pc = rs.pointcloud()

colorizer = rs.colorizer()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_init = np.asanyarray(color_frame.get_data())
        
try:
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
        
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                colorized = colorizer.process(frames)
                
                
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
        
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
        
                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape
        
                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
                
                
                
                Width_color = color_frame.get_width()         #Gets width in terms of pixels for color image
                Height_color = color_frame.get_height()       #Gets height in terms of pixels for color image
                
                Width_depth = depth_frame.get_width()         #Gets width in terms of pixels for depth image
                Height_depth = depth_frame.get_height()       #Gets height in terms of pixels for depth image
                # depth = depth_frame.get_distance(int(Width/2),int(Height/2))    #Les av distance
                # Distance.append(depth)          #Legger til distansem??ling i liste
                
                #rs.pointcloud.calculate(depth_frame)
                
                
                #''' ---------------object detection tensor-----------------------------'''
                image_np_expanded = np.expand_dims(color_image, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    color_image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                #
                Dist = []
                objects = []
                for index, value in enumerate(classes[0]):
                    object_dict = {}
                    if scores[0, index] > 0.8:
                        object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                            scores[0, index]
                        
                    objects.append(object_dict)
                    
                objects[:] = [a for a in objects if a] #removes "empty" objects in array
                #print (objects)
                for i, box in enumerate(boxes[0]):
                    
                    if scores[0, i] > 0.8:
                        
                        ymin, xmin, ymax, xmax = box    
                        x = (xmin + (xmax-xmin)/2)*Width_color
                        y = (ymin + (ymax-ymin)/2)*Height_color
                        #Makes sure x and y is within camera parameters
                        if x >= int(Width_color):
                            x = Width_color
                            
                        if y >= int(Height_color):
                            y = Height_color
                            
                        ratio_height = Height_depth/Height_color
                        ratio_width = Width_depth/Width_color
                        
                        x_depth = x * ratio_width
                        y_depth = y * ratio_height
                        
                        depth_intr = depth_frame.profile.as_video_stream_profile().intrinsics
                        color_intr = color_frame.profile.as_video_stream_profile().intrinsics
                        
                        Obj_dist = depth_frame.get_distance(int(x_depth), int(y_depth))
                        print(Obj_dist)
                        Coordinate_point = rs.rs2_deproject_pixel_to_point(color_intr,(int(x_depth),int(y_depth)) ,Obj_dist)
                        
                        print(Coordinate_point)
                        
                        
                        if(Coordinate_point != [0,0,0]):
                            Coord_number = 1
                            if(Coord_number == 1):                      #Only saving one coordinate
                                with open('Coord.csv', 'w') as csv_file:
                                    csv_writer = csv.writer(csv_file, delimiter = ',')
                                    csv_writer.writerow(Coordinate_point)
                                    Coord_number += 1
                        
                        
                        if Obj_dist != 0.0:
                            cv2.drawMarker(color_image,(int(x),int(y)), [0,0,255])
                            Dist.append(Obj_dist)
                        else:
                            continue
                    else:
                        pass
                
                #'''-------------------------------------------------------------'''

                
                
                # Show images
                #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow('RealSense1', cv2.WINDOW_AUTOSIZE)
                #cv2.namedWindow('RealSense2', cv2.WINDOW_AUTOSIZE)
                #cv2.namedWindow('RealSense3', cv2.WINDOW_AUTOSIZE)
                #cv2.imshow('RealSense',images)
                cv2.imshow('RealSense1',color_image)
                # cv2.imshow('RealSense2', res)
                # cv2.imshow('RealSense3', thresh)
                cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
