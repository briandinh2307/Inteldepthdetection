import keras
from keras.models import Sequential, model_from_json, load_model
import keras_metrics as km
import time
import cv2
import numpy as np
resized_width = 640 // 4
resized_height = 480 // 4

lastUploaded_predict = time.strftime('%Y%m%d%H%M%S')

#load model
test_model = load_model('Data/human-vs-dog.model_3.h5', 
                        custom_objects={
                            'categorical_precision':km.categorical_precision(),
                            'categorical_recall':km.categorical_recall(),
                            'categorical_f1_score':km.categorical_f1_score()})

def predict(img, previous_pred_time):
    global lastUploaded_predict
    timestamp = time.strftime('%Y%m%d%H%M%S')
    result = None
    if (int(timestamp) - int(lastUploaded_predict) >= 1):
        if (previous_pred_time == None or int(timestamp) - int(previous_pred_time) >= 2):
            norm_image = cv2.resize(img, (resized_width, resized_height))
            norm_image = norm_image.astype(np.float32) / 255.0
            test_X = np.expand_dims(norm_image, axis=2)
            test_X = np.expand_dims(test_X, axis=0)     

            #predict
            prediction = test_model.predict_classes(test_X)
            if (prediction == 0):
                print('dog')
                result = 'dog'
            elif (prediction == 1):
                print('human')
                result = 'human'
            lastUploaded_predict = timestamp

    return result, timestamp
