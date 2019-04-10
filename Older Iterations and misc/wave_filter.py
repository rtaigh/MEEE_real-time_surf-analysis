# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 17:32:34 2019

@author: RUAIRI
"""

import cv2
import numpy as np
#fifo stack

cap = cv2.VideoCapture('19_1_2019_1.mp4')#'My Video1.mp4')#
fgbg=cv2.createBackgroundSubtractorMOG2(history=1000000,detectShadows = True)#,varThreshold=3
#fgbg2=cv2.createBackgroundSubtractorKNN()

term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1)


_,first_frame = cap.read()
height, width = first_frame.shape[:2]
#new_frame_size=first_frame
x,y,w,h=0,0,1280,100
x1,y1,w1,h1=0,0,1280,100
#rolling roi point
roi_reset=int(height/5)
init_roi_mog=roi_reset
init_roi_hsv=roi_reset
Roi_height=int(height/(height/25))
wave_numb= 0
lower_t_frm=3

def nothing(x):
    pass


def hvs_mask(frame, min,max):
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask_hsv=cv2.inRange(hsv,min,max)
    return  mask_hsv

'''
cv2.namedWindow("trackBars")
cv2.createTrackbar("X-sobel","trackBars",0,31, nothing)
cv2.createTrackbar("Y-sobel","trackBars",0,31, nothing)

cv2.namedWindow("trackBars")
cv2.createTrackbar("L-H","trackBars",0,179, nothing)
cv2.createTrackbar("L-S","trackBars",0,255, nothing)
cv2.createTrackbar("L-V","trackBars",0,255, nothing)
cv2.createTrackbar("U-H","trackBars",0,179, nothing)
cv2.createTrackbar("U-S","trackBars",0,255, nothing)
cv2.createTrackbar("U-V","trackBars",0,255, nothing)
'''

while True:
    # now to make an iterative loop
    
  #initial reading  
    _, frame = cap.read()
    
    #unaltered display frame
    frame_disp= frame
    cv2.imshow("clean frame",frame)
    frame_hsv=frame
    frame_mog=frame
    
    #redaction
    #frame_hsv=cv2.rectangle(frame_hsv,(0,0),(1280,y1),(0,0,0),-10)
    #remove above noise
    #frame_mog=cv2.rectangle(frame_mog,(0,0),(1280,y),(0,0,0),-10)
    
    '''
    Xsobel=cv2.getTrackbarPos("X-sobel","trackBars")
    if Xsobel % 2==0:
        Xsobel=Xsobel-1
    Ysobel=cv2.getTrackbarPos("Y-sobel","trackBars")
    if Ysobel%2==0:
        Ysobel=Ysobel-1
    
    l_h=cv2.getTrackbarPos("L-H","trackBars")
    l_s=cv2.getTrackbarPos("L-S","trackBars")
    l_v=cv2.getTrackbarPos("L-V","trackBars")
    u_h=cv2.getTrackbarPos("U-H","trackBars")
    u_s=cv2.getTrackbarPos("U-S","trackBars")
    u_v=cv2.getTrackbarPos("U-V","trackBars")

    

    hsv_min=np.array([l_h,l_s,l_v])
    #max 180,255,255
    hsv_max=np.array([u_h,u_s,u_v])
    '''
    
    new_frame_size=frame[y:720,0:1280]
    #  print('position values: ',x,y,w,h)
    #cv2.imshow('new frame',new_frame_size)
    
    
   #initial calibration
   #hsv_min=np.array([0,0,120])
   #hsv_max=np.array([180,30,150])
    
   # whiteonly: min[0,0,13],max[179,33,255]
   # blueonly: min[90,0,0] max[110,20,255]
   #max 180,255,255
    hsv_min=np.array([0,0,13])
    hsv_max=np.array([179,33,255])
    #hsv mask
  
    mask_hsv=hvs_mask(frame_hsv,hsv_min,hsv_max)
    
    
    
    
    

    laplacian=cv2.Laplacian(frame, cv2.CV_64F)
   # sobelx=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=0)
    sobely=sobelx=cv2.Sobel(frame_mog,cv2.CV_64F,0,1,ksize= 15)
    canny = cv2.Canny(frame_mog, 28,50)
    #canny edge detection
   # cv2.imshow('x',sobelx)# don't care about this one
    #cv2.imshow('y',sobely)
    cv2.imshow('canny',canny)
    

   # cv2.imshow('wave mask',mask_hsv)
    #mog2 mask
    mask1=fgbg.apply(frame_mog)
    
    #mask2=mask_hsv/mask1
    mask4=mask_hsv+mask1
    mask5=mask_hsv*mask1
    mask1=mask1+canny
    cv2.imshow("mog+canny",mask1)
    #cv2.imshow("hsv-mog",mask3)
    #cv2.imshow("hsv+mog",mask4)
  #  mask2=fgbg2.apply(frame)
    
    #mask=mask1+mask2
    
    ###########Aproximating conours#############################################################################
    _,contours,_ = cv2.findContours(mask1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >5.0:
            cv2.drawContours(frame_mog,contour,-1,(255,0,255),1)
         
                                   #knn2 best
    _, track_window_mog = cv2.meanShift(mask1,(0,init_roi_mog,1280,Roi_height),term_criteria)
    _, track_window_HSV = cv2.meanShift(mask_hsv,(0,init_roi_hsv,1280,Roi_height),term_criteria)
    _, track_window_ADD = cv2.meanShift(mask4,(0,init_roi_hsv,1280,Roi_height),term_criteria)
    _, track_window_mlt = cv2.meanShift(mask5,(0,init_roi_hsv,1280,Roi_height),term_criteria)
    # print('track_window_update : ',track_window)
    
    
    
    #intergare with HVS for wash?
    #if white present(wash) track wash 
   
    #search roi for waves with yolo?
    #create roi of previous frame with max height being peak of previous roi
    #function to initially trace screen from most mask
    #measure widest as descending speed
    #rooling frame size so roi cannot jump
    #update roi values
    x_mog1,y_mog1,w_mog1,h_mog1=x,y,w,h
    x_hvs1,y_hvs1,w_hvs1,h_hvs1=x1,y1,w1,h1
    x,y,w,h= track_window_mog
    x1,y1,w1,h1=track_window_HSV
    #uni-directional
    if y<y_mog1:
        y=int(y_mog1)
        #+(y1-y))#int ((y1+y)/2)#adds reverse difference
        
        
    if y<y_hvs1:
        y1=int(y_hvs1)
        #+(y1-y))#int ((y1+y)/2)#adds reverse difference
    
    #once reaching bottom of screen system reset
    if y+h>= height-height/lower_t_frm:
       y_mog1=roi_reset+1
       y=roi_reset
       wave_numb=wave_numb+1
       print('Number of waves (mog):', wave_numb)
    
    
    if y1+h>= height-height/lower_t_frm:
       y_hvs1=roi_reset+1
       y1=roi_reset
       wave_numb=wave_numb+1
       print('        Number of waves (hsv):', wave_numb)
    
    #roi height now pevious value
    init_roi_mog=y
    init_roi_hvs=y1
    
    
    
    #print(track_window)
    #tracking frame
    cv2.rectangle (frame_disp,(x,y),(x+w,y+h),(0,0,255),1)
   # cv2.rectangle (frame_disp,(x1,y1),(x1+w,y1+h),(255,0,0),1)
   # cv2.rectangle (frame_disp,(x1,y1),(x1+w,y1+h),(255,0,255),1)
    #new_frame_size=frame[y:720,0:1280]
    #cv2.imshow('crop',new_frame_size)
    cv2.imshow('frame',frame)
  
    
   
   
   
   
   

 
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
  
cv2.destroyAllWindows()
cap.release()