import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

def facial_detection(img:np.ndarray):
  cascade_faces = './haarcascade_frontalface_default.xml'
  face_detection = cv2.CascadeClassifier(cascade_faces)
  original = img.copy()
  faces = face_detection.detectMultiScale(original, scaleFactor=1.1, minNeighbors=3, minSize=(20,20))
  [cv2.rectangle(original, (faces[0,0],faces[0,1]), (faces[0,0]+faces[0,2], faces[0,1]+faces[0,3]), color=(0,0,255), thickness=3)for i in range(len(faces))]
  return faces, original
  

def roi_extractor(img:np.ndarray, faces:np.ndarray):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  roi = [gray[faces[0,1]:faces[0,1]+faces[0,-1], faces[0,0]:faces[0,0]+faces[0,-2]]for i in range(len(faces))]
  for i in range(len(roi)):
    roi[i] = cv2.resize(roi[i], (48,48))
  roi = np.array(roi)
  roi = img_to_array(np.array(roi)/255)
  roi = np.expand_dims(roi, axis = 0)
  return roi

def emotion_recognition(img:np.ndarray, model):

 
  emotions = ['Angry', 'disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

  faces, bbox = facial_detection(img)
  try:
    roi = roi_extractor(img, faces)
  except:
    return img

  preds = [model.predict(roi[0]) for i in range(len(roi))][0]
  label = [emotions[preds[0].argmax()]for i in range(len(preds))]  
  #plotting the results:
  [cv2.putText(bbox, f'face{i+1}-{label[0]}',(faces[0,0]+5, faces[0,1]-5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=bbox.shape[0]/450, color=(0,0,255), thickness=5, lineType=cv2.LINE_AA) for i in range(len(faces))]
  bbox= cv2.resize(bbox, (300,300))
  
  return bbox