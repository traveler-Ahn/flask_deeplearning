import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX

# Set the colors for the bounding boxes
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(classes), 3),
	dtype="uint8")

class VideoCamera(object):
    def __init__(self, input='cam', margin=None, resize_prop=(640, 480), obj_detector=None, 
                       face_detector=None, age_gener_detector=None, emotion_detector=None):
        if input == 'cam':
            self.input_stream = 0
            self.cap = cv2.VideoCapture(self.input_stream)
            self.resize_prop = resize_prop

        else:
            self.input_stream = input
            assert os.path.isfile(input), "Specified input file doesn't exist"
            self.cap = cv2.VideoCapture(self.input_stream)

        ret, self.frame = self.cap.read()
        cap_prop = self._get_cap_prop()
        self.margin = margin

        self.detections = obj_detector
        self.face_detector = face_detector
        self.age_gener_detector = age_gener_detector
        self.emotion_detector = emotion_detector
        self.img_width, self.img_height = (300, 300)    # SSD object detection inference image size
        
    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self, flip_code=None, is_object_detection=True, is_face_detection=True, 
                        is_age_gender_detection=True, is_emotions_detection=False, is_facial_landmarks_detection=False):

        ret, self.frame = self.cap.read()
        if not ret:
            return None
        self.frame = cv2.resize(self.frame, self.resize_prop)
        if self.input_stream == 0 and flip_code is not None:
            self.frame = cv2.flip(self.frame, int(flip_code))

        if is_object_detection:
            results_obj = self.detections.predict_result(self.frame)

            for box in results_obj[0]:
                frame_h, frame_w = self.frame.shape[:2]

                # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
                box_coord = box[2:6] * np.array([frame_w/self.img_width, frame_h/self.img_height, frame_w/self.img_width, frame_h/self.img_height])
                xmin, ymin, xmax, ymax = self._crop_face((box_coord.astype("int")))

                cv2.putText(self.frame, classes[int(box[0])], (xmin, ymin), font, 1, (255,255,0), 2)
                cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax), (int(COLORS[int(box[0])][0]), 
                                                                       int(COLORS[int(box[0])][1]), 
                                                                       int(COLORS[int(box[0])][2])), 2)

        if is_face_detection:
            small_fr = cv2.resize(self.frame, (0,0), fx=0.25, fy=0.25)    
            rgb_small_fr = small_fr[:,:,::-1]
            results = self.face_detector.detect_faces(rgb_small_fr)

            # For detected Faces
            for result in results:
                x1, y1, x2, y2 = self._crop_face(result['box'], is_face=True)

                # Emotion detection
                if is_emotions_detection:
                    gray_fr = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    fc = gray_fr[y1:y2, x1:x2]
                    roi = cv2.resize(fc, (48,48))
                    pred = self.emotion_detector.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                    cv2.putText(self.frame, pred, (x1,y2), font, 1, (255,255,0), 2)

                # Facial landmark detection
                if is_facial_landmarks_detection:
                    # Transform the predicted bounding boxes for the image*0.25 to the original image dimensions.
                    for key, val in result['keypoints'].items():
                        if key == "mouth_left" or key == "mouth_right":
                            continue
                        cv2.circle(self.frame, (val[0]*4, val[1]*4), 5, (0,0,255), -1)
                    cv2.line(self.frame, (result['keypoints']['mouth_left'][0]*4, result['keypoints']['mouth_left'][1]*4), 
                                         (result['keypoints']['mouth_right'][0]*4, result['keypoints']['mouth_right'][1]*4), 
                                         (0,0,255), 5)
                # Age & Gender detection
                if is_age_gender_detection:
                    roi = self.frame[y1:y2, x1:x2]
                    label = self.age_gener_detector.predict_agge(roi)
                    cv2.putText(self.frame, label, (x1,int((y1+y2)/2)), font, 1, (255,255,0), 2)
                                        
                cv2.putText(self.frame, (str(round(result['confidence'], 2)*100) + "%"), (x1,y1), font, 1, (255,255,0), 2)
                cv2.rectangle(self.frame, (x1,y1), (x2,y2),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', self.frame)

        return jpeg.tobytes()

    def _crop_face(self, box, is_face=False):

        if is_face:
            # Transform the predicted bounding boxes for the image*0.25 to the original image dimensions.
            x, y, w, h = box * np.array([4])
            margin = int(min(w, h) * self.margin / 100)

            x_a = x - margin
            y_a = y - margin
            x_b = x + w + margin
            y_b = y + h + margin
        else:
            x_a, y_a, x_b, y_b = box

        if x_a < 0:
            x_a = 0
        if y_a < 0:
            y_a = 0
        if x_b > self.frame.shape[1]:
            x_b = self.frame.shape[1]
        if y_b > self.frame.shape[0]:
            y_b = self.frame.shape[0]

        return x_a, y_a, x_b, y_b