import numpy as np
import cv2
import os
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
import easyocr

import matplotlib
import matplotlib.pyplot as plt

modelpath='detect.tflite'
lblpath='labelmap.txt'
min_conf=0.5
cap = cv2.VideoCapture('demo.mp4')

reader = easyocr.Reader(['en'], gpu=False)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Choose the appropriate codec
output_video = cv2.VideoWriter('annotated_video.avi', fourcc, 30.0, (1114, 720)) 


interpreter = Interpreter(model_path=modelpath)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

with open(lblpath, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

while(True):
    ret, frame =cap.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imH, imW, _ = frame.shape

    print(imH, imW)
    #output_video = cv2.VideoWriter('annotated_video.avi', fourcc, 30.0, (imW, imH))
    
    
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std
        
    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    
    detections = []
    
    
    for i in range(len(scores)):
        if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            cropped_object = frame[ymin:ymax, xmin:xmax]
            image_filename = 'detected_object.jpg'
            cv2.imwrite(image_filename, cropped_object)
            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = reader.readtext(image_filename)  # Example: 'person: 72%'
            #labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            #label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin,ymin-20), (xmax,ymin), (10, 255, 0), cv2.FILLED)
            #cv2.rectangle(frame, xmin, ymax, (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            
            if label:
                label = label[0][1]
            else:
                label="License Plate"
            cv2.putText(frame, label, (xmin, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])
    
    output_video.write(frame)
    
    
    #cv2.imshow('output',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
    
cap.release()
cv2.destroyAllWindows()
    