import cv2
import os
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.optimizers import Adam

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss

from models.wide_resnet import WideResNet

class ObjectDetectionModel(object):
    def __init__(self, confidence_threshold=0.5):

        self.confidence_th = confidence_threshold

        # 0: Set the image size.
        img_height = 300
        img_width = 300

        # 1: Build the Keras model
        self.loaded_model = ssd_300(image_size=(img_height, img_width, 3),
                                    n_classes=20,
                                    mode='inference',
                                    l2_regularization=0.0005,
                                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                            [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                            [1.0, 2.0, 0.5],
                                                            [1.0, 2.0, 0.5]],
                                    two_boxes_for_ar1=True,
                                    steps=[8, 16, 32, 64, 100, 300],
                                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                    clip_boxes=False,
                                    variances=[0.1, 0.1, 0.2, 0.2],
                                    normalize_coords=True,
                                    subtract_mean=[123, 117, 104],
                                    swap_channels=[2, 1, 0],
                                    confidence_thresh=0.5,
                                    iou_threshold=0.45,
                                    top_k=200,
                                    nms_max_output_size=400)

        # 2: Load the trained weights into the model.
        weights_path = 'models/VGG_VOC0712_SSD_300x300_iter_240000.h5'
        self.loaded_model.load_weights(weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
        self.loaded_model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

        # 4: make prediction (graph)
        self.loaded_model._make_predict_function()

    def predict_result(self, frame):
        inference_frame = cv2.resize(frame, (300, 300)) 
        inference_frame = inference_frame[:,:,::-1]

        img = image.img_to_array(inference_frame)
        img = img[np.newaxis,:,:,:] # Batch size axis add


        results = self.loaded_model.predict(img)
        results_obj = [results[k][results[k,:,1] >= self.confidence_th] for k in range(results.shape[0])]

        return results_obj

class FacialExpressionModel(object):
    EMOTION_LIST = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

    def __init__(self):
        
        model_json_file = 'models/model_4layer_2_2_pool.json'
        model_weights_file = 'models/model_4layer_2_2_pool.h5'

        # load model from JSON file     
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTION_LIST[np.argmax(self.preds)]

class FaceGenderAge(object):

    WRN_WEIGHTS_PATH = "https://github.com/Tony607/Keras_age_gender/releases/download/V1.0/weights.18-4.06.hdf5"

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        model_dir = os.path.join(os.getcwd(), "models")
        fpath = get_file('weights.18-4.06.hdf5',
                         self.WRN_WEIGHTS_PATH,
                         cache_subdir=model_dir)
        self.model.load_weights(fpath)
        self.model._make_predict_function()

    def predict_agge(self, frame):
        inference_frame = cv2.resize(frame, (self.face_size, self.face_size))
        inference_frame = inference_frame[np.newaxis,:,:,:]
        results = self.model.predict(inference_frame)
        
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        label = "Age: {}, Gender: {}".format(int(predicted_ages), "F" if predicted_genders[0][0] > 0.5 else "M")
        return label