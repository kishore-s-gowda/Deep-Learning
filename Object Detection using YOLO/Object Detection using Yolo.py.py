#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread


net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
with open('coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
output_layers=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]


img=cv2.imread('Traffic.jpg')
height, width, channels = img.shape


blob=cv2.dnn.blobFromImage(img,scalefactor=1/255.,size=(416,416),swapRB=True,crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


class_ids = []
confidences = []
boxes = []
for out in outs:
    print(out.shape)
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.3:
            #print("Object detected")
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
indexes


colors = np.random.uniform(0, 255, size=(len(classes), 3))


font=cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x,y,w,h=boxes[i]
        label=str(classes[class_ids[i]])
        color=colors[i]
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img,label,(x,y+200),font,5,color,thickness=5)
        print(label)


cv2.imwrite('Traffic_detect.jpg',img)

