#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import cv2
from PIL import Image, ImageOps
import time
import numpy as np
import math


font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 0, 0)
fontScale = 1
thickness = 2

protoFile = "C:/Users/kumar/Documents/knockknees/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "C:/Users/kumar/Documents/knockknees/pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]


st.write("""
         # Knock Knees prediction
         """
         )
st.write("This is a Web App to predict if you have knock knees or not by calculating angle using opencv.")


file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    img = np.asarray(image)
    st.image(image, use_column_width=True)
    frame = img
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")


    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    lh = points[11]
    lk = points[12]
    la = points[13]
    rh = points[8]
    rk = points[9]
    ra = points[10]

    if ((lh is None) or (lk is None) or (la is None)):
        left_angle = "Cannot Determine!"

    else:

        if (lk[0]-lh[0]) == 0:
            lh_lk = math.degrees(math.atan(abs(float('inf'))))
            lh_lk = 180 - lh_lk
        else:
            lh_lk = math.degrees(math.atan(abs(float(((-lk[1])-(-lh[1]))/(lk[0]-lh[0])))))
            lh_lk = 180 - lh_lk

        if (la[0]-lk[0]) == 0:
            lk_la = math.degrees(math.atan(abs(float('inf'))))
            lk_la = 180 - lk_la
        else:
            lk_la = math.degrees(math.atan(abs(float(((-la[1])-(-lk[1]))/(la[0]-lk[0])))))
            lk_la = 180 - lk_la

        left_angle = lh_lk + lk_la

    if ((rh is None) or (rk is None) or (ra is None)):
        right_angle = "Cannot Determine!"

    else:
        if (rk[0]-rh[0]) == 0:
            rh_rk = math.degrees(math.atan(abs(float('inf'))))
            rh_rk = 180 - rh_rk

        else:
            rh_rk = math.degrees(math.atan(abs(float(((-rk[1])-(-rh[1]))/(rk[0]-rh[0])))))
            rh_rk = 180 - rh_rk


        if (ra[0]-rk[0]) == 0:
            rk_ra = math.degrees(math.atan(abs(float('inf'))))
            rk_ra = 180 - rk_ra
        else:
            rk_ra = math.degrees(math.atan(abs(float(((-ra[1])-(-rk[1]))/(ra[0]-rk[0])))))
            rk_ra = 180 - rk_ra

        right_angle = rh_rk + rk_ra

    Left_angle_text = ('Left Angle = '+str(left_angle))
    right_angle_text = ('Right Angle = '+str(right_angle))

    if ((abs(right_angle - 180) > 20) or (abs(left_angle - 180) > 20)):
        result = "Result:- Knock Knees"
    else:
        result = "Result:- Normal Knees"


    cv2.putText(frame, Left_angle_text, (0, 75), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, right_angle_text, (0, 150), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(frame, result, (0, 300), font, fontScale, color, thickness, cv2.LINE_AA)


    cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    cv2.imwrite('Output-Skeleton.jpg', frame)

    print("Total time taken : {:.3f}".format(time.time() - t))
    
    
    st.image(frame, use_column_width=True)
    st.write(result)








# In[ ]:




