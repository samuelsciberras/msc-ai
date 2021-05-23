# Imports
from copy import copy
from core.Helpers.Classes.vehicledetection import VehicleDetection
import core.Helpers.general as generalFuncts
import core.Helpers.speedEstimation as speedEstFuncts
from numpy.core.fromnumeric import size
import os
import time
import copy


# 3rd Party Imports
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
# Comment to enable TF Logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Yolov4 & DeepSort Imports
from core.yolov4 import filter_boxes
import core.utils as utils
from core.config import cfg
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


# Define Flags (Arguments from command-line)
flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'Path to YoloV4 weights file')
flags.DEFINE_integer('size', 416, 'Image Size')
flags.DEFINE_integer('fps', 30, 'Frames per second of input')
flags.DEFINE_string('video', './data/video/sample.mp4', 'Path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'Path to output video')
flags.DEFINE_string('output_format', 'XVID', 'Codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.8, 'IoU threshold')
flags.DEFINE_float('score', 0.8, 'Score threshold for keeping Bounding Boxes')


def main(_argv):
    # Initialize params
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # Initialize DeepSORT
    model_filename = 'model_data/trainedModel.pb' # Path to trained DeepSORT model
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # Initialize cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # Initialize tracker
    tracker = Tracker(metric)

    # Load object detector configs
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # Load YoloV4 Weights
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # Read Input Video or Cam -> (0) for cam
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # If output flag is set, we save the video
    if FLAGS.output:
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    # Read Video/Webcam (for webcam, set VideoCapture(0))
    frame_num = 0
    FullDetectionList = []
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended')
            break
        
        frame_num +=1
        print('Frame #: ', frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # Run Detections
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # Convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # Format bounding boxes to YoloV4 standard annotation
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        pred_bbox = [bboxes, scores, classes, num_objects]

        # Get Class Names and convert to numpy array
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        names = np.array(list(class_names.values()))

        # Encode Yolo for tracking
        features = encoder(frame, bboxes)
        detections = []
        for iterator in range (0, pred_bbox[3]):
            detections.append(Detection(pred_bbox[0][iterator], pred_bbox[1][iterator], names[int(pred_bbox[2][iterator])], features[iterator]))

        # NMS
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Start tracking only after centroid is within x% of the frame (i.e. 90% of Width or Length)
        # Filter out detections that are not within this range
        filterDetectionIndices = generalFuncts.FilterWithPadding(detections, FLAGS.size, 0.85)
        for idx in sorted(filterDetectionIndices, reverse=True):
            detections.pop(idx)

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # Process Tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            
            trackBoundingBox = track.to_tlbr()
            yolobbox = track.yoloDetections[-1].to_tlbr()
            class_name = track.get_class()

            # Create vehicle detection object
            vehdet = VehicleDetection(track.track_id, class_name, track.yoloDetections[-1].tlwh, yolobbox, frame_num)
            
            speed, prev_frame = speedEstFuncts.EstimateSpeed(FullDetectionList, vehdet)
            if speed == 0:
                speed_str = "?"
            else:
                speed_str = str(speed)


            if prev_frame is not None:
                vehdet.HistoricalSpeed = prev_frame.HistoricalSpeed
                vehdet.HistoricalSpeed.append(speed)
            FullDetectionList.append(vehdet)

            # Draw on image output
            for i in range(0, vehdet.HistoricalCentroids.__len__()):
                if vehdet.HistoricalCentroids[i - 1] is not None:
                    cv2.line(frame,
                             (int(vehdet.HistoricalCentroids[i][0]), int(vehdet.HistoricalCentroids[i][1])),
                             (int(vehdet.HistoricalCentroids[i-1][0]), int(vehdet.HistoricalCentroids[i-1][1])),
                             (255, 0, 0),
                             1)

            cv2.rectangle(frame,
                        (int(yolobbox[0]), int(yolobbox[1])),
                        (int(yolobbox[2]), int(yolobbox[3])),
                        (0, 255, 0),
                        2)
            cv2.putText(frame,
                        f"{speed_str} m/s",
                        (int(yolobbox[0]), int(yolobbox[1]-20)),
                        0,
                        0.4,
                        (255, 255, 255),
                        1)
            cv2.putText(frame,
                        f"{class_name}-{track.track_id}",
                        (int(trackBoundingBox[0]), int(trackBoundingBox[1]-5)),
                        0,
                        0.4,
                        (255,255,255),
                        1)

        # Calculate Processing FPS
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Output Video", result)
        

        # Save if output is set
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
