# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 12:57:07 2021

@author: 91915
"""

import cv2
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
#cred = credentials.Certificate('path/to/serviceAccountKey.json')
cred = credentials.Certificate('count-people-22746-firebase-adminsdk-ath9b-5b61a19ac1.json')

# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://count-people-22746.firebaseio.com/'
})

ref = db.reference('/place/temple')

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#load input video stream
cap = cv2.VideoCapture('temple.mp4')
#instantiate a variable 'p' to keep count of persons
p = 0  
#initialize the writer
writer = None
(W, H) = (None, None)
starting_time = time.time()
frame_id = 0
while True:
    ret , frame= cap.read()
    frame_id += 1
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
 
    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    class_ids = []
    # loop over each of the layer outputs
    for out in outs:
        # loop over each of the detections
        for detection in out:
            # extract the class ID and confidence 
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # filter out weak predictions
            if confidence > 0.7:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
  
                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)
    #detecting persons
    if len(indexes) > 0:
        # loop over the indexes we are keeping
        for i in indexes.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            label = str(classes[class_ids[i]])
            if label == 'person':
                p=p+1
            else:
                continue
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = label + ':' + str(p)
            cv2.putText(frame, text, (x, y+30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #writer = cv2.VideoWriter("person_out1.mp4", fourcc, 30,(frame.shape[1], frame.shape[0]), True)
    elapsed_time = time.time() - starting_time
    ref.update({'count': p})
    print(p)
    p=0
    fps = frame_id / elapsed_time
    print(str(round(fps, 2)))
    cv2.imshow("frame", frame)
    #writer.write(frame)
    if cv2.waitKey(1) == 27:
        cap.release()
        #writer.release()
        break
    
cv2.destroyAllWindows()