# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:46:37 2018

@author: RUAIRI
"""

import cv2
import logging
import sys
import streamlink
import os.path

from matplotlib import pyplot as plt
def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7,7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask

streams=streamlink.streams("https://youtu.be/DY5RYp4sxYc")#("https://youtu.be/F3Q1n_DpA9o')
cam_capture = cv2.VideoCapture('C:\\Users\\RUAIRI\\Downloads\\Pipeline.mp4')
cv2.destroyAllWindows()


while True:
    _, image_frame = cam_capture.read()
    
    #cv2.imshow('stream',streams )
   # upper_left = (0, 250)#works
   # bottom_right = (1080, 1080)#works
    
    b_upper_left = (0, 600)#works
    b_bottom_right = (1920, 2000)#works
    
    
    #Rectangle marker
    r = cv2.rectangle(image_frame, b_upper_left, b_bottom_right, (250, 0, 250), 5)
    rect_img = image_frame[b_upper_left[1] : b_bottom_right[1], b_upper_left[0] : b_bottom_right[0]]
    
    sketcher_rect = rect_img
    sketcher_rect = sketch_transform(sketcher_rect)
    
    #Conversion for 3 channels to put back on original image (streaming)
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    
    #Replacing the sketched image on Region of Interest
   # image_frame[b_upper_left[1] : b_bottom_right[1], b_upper_left[0] : b_bottom_right[0]] = sketcher_rect_rgb
    cv2.imshow("Sketcher ROI", image_frame)
    cv2.imshow("rect_img", rect_img)
    if cv2.waitKey(1) == 13:
        break
        
cam_capture.release()
cv2.destroyAllWindows()