# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:46:57 2018

@author: RUAIRI
"""

import cv2
import numpy as np

cap = cv2.VideoCapture('My Video1.mp4')#'ShibuyaCrossingFullHD.mp4')
fgbg=cv2.createBackgroundSubtractorMOG2(history=250,detectShadows = True)#,varThreshold=3
fgbg2=cv2.createBackgroundSubtractorKNN()
Ubody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')

#roi
upper_left = (50, 50)
bottom_right = (300, 300)


kernelOp = np.ones((3,3),np.uint8)
kernelCl = np.ones((11,11),np.uint8)
areaTH = 500




while True:
    ret, frame = cap.read()
 
    people=Ubody_cascade.detectMultiScale(frame,20,4)
    
    for (x,y,w,h) in people:
        _, contours, _ =cv2.findContours(people,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cv2.cv2.drawContours(frame,contours,-1,(0,255,0),1)
        
        
 
    lower_Val_Jskin=np.array([0,0, 0])
    upper_Val_Jskin= np.array([30, 160, 255])
    
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hugh saturation value
    
    mask= cv2.inRange(hsv,lower_Val_Jskin,upper_Val_Jskin)
    
    res=cv2.bitwise_and(frame,frame, mask=mask)
    mask1=fgbg.apply(frame)
    mask2=fgbg2.apply(frame)
    
 #   mask=mask1+mask2+mask
    
   
 
    _, contours, _ =cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame,contours,-1,(125,0,125),1)
    faces=Ubody_cascade.detectMultiScale(mask1,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray=mask[y:y+h,x:x+w]
        img_item="myimg_item_imagr.png"
        cv2.imwrite(img_item,roi_gray)
        cv2.imshow('Frame',frame)
        stroke= 2
        color=(255,0,0)
        eyeboxW=x+w
        eyeboxH= y+h
        cv2.rectangle(frame,(x,y),(eyeboxW,eyeboxH),color,stroke)
    #mask=mask2+res
   # res=cv2.bitwise_and(mask,mask,mask2=mask2)

        roi=cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 5)
        rect_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
        #sketcher_rect = rect_img

        
      #  Fcrop=frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    #cv2.imshow('frame',frame)
    cv2.imshow('res',mask)
    cv2.imshow('Frame',frame)
   # cv2.imshow('maskMOG',mask)
    #cv2.imshow('maskKNN',mask2)
    
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
  
        cv2.destroyAllWindows()
        cap.release()