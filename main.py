import cv2
import numpy as np
from tensorflow.keras.models import load_model, model_from_json
from utils import emotion_recognition

if __name__ == '__main__':
    model_file_json = './model_01_expressions.json'
    model_file = './model_01_expressions.h5'
    
    json_file = open(model_file_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_file)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('camera', cv2.WND_PROP_FULLSCREEN)
    img_counter = 0
    
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print('failed to grab frame.')
            break
        img = emotion_recognition(frame, loaded_model)

        #print(f'\n\nimg.shape: {img.shape}')
        cv2.imshow('camera', img)

        k = cv2.waitKey(1)
        if k & 0xFF == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()    
    