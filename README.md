# Flask_deeplearning

## 실행 화면 (App running)

- Face Detection (좋아하는 [연플리 시즌4](https://www.youtube.com/watch?v=PNuGLykXyzE):))
<img src="https://github.com/traveler-Ahn/flask_deeplearning/blob/master/img/main_1.png" width="50" height="300">

- Object Detection
<img src="https://github.com/traveler-Ahn/flask_deeplearning/blob/master/img/main_2.png" width="50" height="300">

- Etc... with Deeplearning
<img src="https://github.com/traveler-Ahn/flask_deeplearning/blob/master/img/main_3.png" width="50" height="300">

## 실행환경 (Environment)
```
Flask 1.1.1
tensorflow 1.14.0 (cpu/gpu) supported
keras 2.2.4
mtcnn 0.0.9 (FaceDetection, Landmark detection)
```

## 실행 코드 (Excution, linux(ubuntu) & window10 supported)
```
$ python app.py -i cam         (USB camera)

or

$ python app.py -i VIDEO_PATH  (Video file)

and Web browser(Firefox & Chrome supported)
http://localhost:5000
```

# Enjoy iT:)

### Reference
Flask Deeplearning: [https://github.com/kodamap/object_detection_demo](https://github.com/kodamap/object_detection_demo)><br>
Age & Gender: [https://github.com/Tony607/Keras_age_gender](https://github.com/Tony607/Keras_age_gender)<br>
Age & Gender: [https://github.com/asmith26/wide_resnets_keras](https://github.com/asmith26/wide_resnets_keras)<br>
ObjectDetection(SSD): [https://github.com/pierluigiferrari/ssd_keras](https://github.com/pierluigiferrari/ssd_keras)<br>
FaceDetection: [https://github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn)
LandMark: [https://github.com/ipazc/mtcnn](https://github.com/ipazc/mtcnn)<br>
