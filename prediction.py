# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 01:49:01 2020

@author: Gaurav
"""
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

model = tf.keras.models.load_model('E:/AI Application Implementation/Face_mask_detection/mask_detection.model')


face = cv2.imread("WhatsApp Unknown 2020-07-13 at 3.34.59 AM/1 (4).jpeg")
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
face = cv2.resize(face, (224, 224))
face = img_to_array(face)
face = preprocess_input(face)
face = np.expand_dims(face, axis=0)
(mask, withoutMask) = model.predict(face)[0]
print(float(mask),float(withoutMask)) 



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels_dict={0:'without_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
        _, frame = video_capture.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
      
        for (x,y,w,h) in faces:
            
            face=frame
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            (mask, withoutMask) = model.predict(face)[0]
                
            if mask>0.5:
                label = "Mask" 
                color = (0, 255, 0) 
            elif withoutMask>=0.5:
                label = "No Mask"
            color=(0, 0, 255)
          
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		
        
            cv2.putText(frame, label, (x,y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()


