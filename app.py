from flask import Flask, render_template, Response, jsonify, request
import sys
import json
import argparse
from camera import VideoCamera
from model import ObjectDetectionModel, FacialExpressionModel, FaceGenderAge
from mtcnn.mtcnn import MTCNN

app = Flask(__name__)

is_object_detection = True
is_face_detection = False
is_age_gender_detection = False
is_emotions_detection = False
is_facial_landmarks_detection = False
flip_code = None  # filpcode: 0,x-axis 1,y-axis -1,both axis

# Flask video streaming https://blog.miguelgrinberg.com/post/video-streaming-with-flask
def gen(camera):
    while True:
        frame = camera.get_frame(flip_code, is_object_detection, 
                                 is_face_detection, is_age_gender_detection,
                                 is_emotions_detection, is_facial_landmarks_detection)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/")
def index():
    return render_template('index.html',
                           flip_code=flip_code,
                           is_object_detection=is_object_detection)

@app.route('/video_feed')
def video_feed():
    camera = VideoCamera(input=args['input'], margin=20, obj_detector=obj_detector, face_detector=face_detector, 
                         age_gener_detector=age_gener_detector, emotion_detector=emotion_detector)
    return Response(
        gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection', methods=['POST'])
def detection():
    global is_object_detection
    global is_face_detection
    global is_age_gender_detection
    global is_emotions_detection
    global is_facial_landmarks_detection

    command = request.json['command']

    # Object detection or Face detection
    if command == "object_detection":
        is_object_detection = True
        is_face_detection = False
    if command == "face_detection":
        is_face_detection = True
        is_object_detection = False
    if command == "age_gender_detection" and not is_object_detection:
        is_age_gender_detection = not is_age_gender_detection
    if command == "emotions_detection" and not is_object_detection:
        is_emotions_detection = not is_emotions_detection
    if command == "facial_landmarks_detection" and not is_object_detection:
        is_facial_landmarks_detection = not is_facial_landmarks_detection
 
    result = {
        "flip_code": flip_code,
        "is_object_detection": is_object_detection,
        "is_face_detection": is_face_detection,
        "is_age_gender_detection": is_age_gender_detection,
        "is_emotions_detection": is_emotions_detection,
        "is_facial_landmarks_detection": is_facial_landmarks_detection
    }
    
    return jsonify(ResultSet=json.dumps(result))    

@app.route('/flip', methods=['POST'])
def flip_frame():
    global flip_code

    command = request.json['command']

    if command == "flip" and flip_code is None:
        flip_code = 0
    elif command == "flip" and flip_code == 0:
        flip_code = 1
    elif command == "flip" and flip_code == 1:
        flip_code = -1
    elif command == "flip" and flip_code == -1:
        flip_code = None

    return command

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        help="Path to video file or image. 'cam' for capturing video stream from camera",
        required=True,
        type=str,
        default=None)
    ap.add_argument(
        "-pt",
        "--prob_threshold",
        help="Probability threshold for object detections filtering",
        default=0.5,
        type=float)

    args = vars(ap.parse_args())
    # Model object instantiation
    obj_detector = ObjectDetectionModel(args['prob_threshold'])       # ObjectDetection model
    face_detector = MTCNN()                                           # FaceDetection & Landmark model 
    emotion_detector = FacialExpressionModel()                        # EmotionRecognition model
    age_gener_detector = FaceGenderAge()                              # Age&Gender Prediction model

    app.run()
