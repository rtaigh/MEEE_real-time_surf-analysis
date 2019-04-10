# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:52:14 2018

@author: RUAIRI
"""
import cv2
#import logging
import sys
import streamlink
#import os.path
import numpy as np
                                        #<400
fgbg=cv2.createBackgroundSubtractorMOG2(history=300,detectShadows = True)



def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7,7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask




    

try:
      # lower_white = np.array([0,0,0], dtype=np.uint8)
    #upper_white = np.array([0,0,255], dtype=np.uint8)
   #lower_Bswell_val=np.array([110, 100, 100,])
   # upper_Bswell_val= np.array([130,255, 255])
#white 
    lower_white_val=np.array([0, 0, 0,])
    upper_white_val= np.array([30,15, 255])
    
    #deep blueFor HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. Different softwares use different scales. So if you are comparing OpenCV values with them, you need to normalize these ranges.
    lower_blk_val=np.array([(155/255)*179, 255*.30, 255*.001,])
    upper_blk_val= np.array([179,255*.99, 255*.99])
   # lower_Per_val=np.array([0, 200, 0,])
    #upper_Per_val= np.array([100,255, 100])

    streams = streamlink.streams("https://youtu.be/bNLy-XXYxcw")
    quality='best'
    cap = cv2.VideoCapture(0)#'E:\\Utube_stock\\My Video.MP4')#'E:\\masters_Video_cap\\Pipeline.mp4')#'E:\\masters_Video_cap\\19_1_2019_5.mp4')#(streams[quality].to_url())#
    frame_time = int((1.0 / 30) * 1000.0)

    while True:
        try:
            ret, frame = cap.read()
            if ret:
            
             #   cv2.imshow('frame', frame)
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hugh saturation value 
                Wmask=cv2.inRange(hsv,lower_white_val,upper_white_val)
                BLKmask=cv2.inRange(hsv,lower_blk_val,upper_blk_val)
               
                mask1=fgbg.apply(frame)
                _, contours, _ =cv2.findContours(BLKmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(frame,contours,-1,(0,0,255),1)
                _, contours, _ =cv2.findContours(Wmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(frame,contours,-1,(26,155,242),1)
                
                
                mask=BLKmask+Wmask
               
                res=cv2.bitwise_and(frame,frame, mask= mask)
                cv2.imshow('waves',mask1)
                cv2.imshow('frame',frame) 
                cv2.imshow('stream',res)  
                cv2.imshow('sketch', sketch_transform(frame))
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
                    break
            else:
                break
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    cap.release()
 
except ImportError:
    sys.stderr.write("This example requires opencv-python is installed")
    raise