# -*- coding: utf-8 -*-
"""
wave filter with kmeans
"""

import cv2
import numpy as np
#import matplotlib.pyplot as plt



cap = cv2.VideoCapture('19_1_2019_1.mp4')#'My Video1.mp4')#
fgbg=cv2.createBackgroundSubtractorMOG2(history=-1,detectShadows = False)#,varThreshold=3

_,first_frame = cap.read()
height, width = first_frame.shape[:2]

frame2=cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

height, width = first_frame.shape[:2]
kernel_erode = np.ones((10,8),np.uint8)
kernel_dilate= np.ones((1,32),np.uint8)
kernel_erode_canny = np.ones((1,32),np.uint8)
fgbg=cv2.createBackgroundSubtractorMOG2(history=-1,detectShadows = True)

term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1)

    
#function for K_means filtering
def K_mask(frame,K):
    Z=frame.reshape((-1,3))
    Z=np.float32(Z)

    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

   #centers are the colors!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ret, label1,center1=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
   # print(label1,center1)
    center1=np.uint8(center1)
    res1=center1[label1.flatten()]
    output1=res1.reshape((frame.shape))
 
    
    
   
    #retval, threshold = cv2.threshold(output1, 200, 255, cv2.THRESH_BINARY)
    #erosions
    threshold = cv2.erode(output1,kernel_erode,iterations = 3)
    threshold = cv2.dilate(threshold,kernel_dilate,iterations = 3)
   # retval2,threshold2 = cv2.threshold(threshold,155,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   
    #cv2.imshow('thresh2',threshold2)
   # cv2.imshow('k_unfiltered',output1)

    return threshold,output1,center1 
    


def scharr_mask(kmask,frame):   

    
    scharr_otsu=cv2.Scharr(frame,-1,dx=0,dy=1,scale=0.5,delta=0,borderType=cv2.BORDER_DEFAULT)
    #scharr_otsu=fgbg.apply(scharr_otsu)
    scharr_otsu = cv2.Canny(scharr_otsu, 40,255)
    
  #  opening= cv2.morphologyEx(canny, cv2.MORPH_OPEN,ero_kernal)
    #cv2.imshow('opening',opening)
    closing= cv2.morphologyEx(canny, cv2.MORPH_CLOSE,ero_kernal)#canny

    _,contours,_=cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
     
    for contour in contours:
        
        x,y,w,h=cv2.boundingRect(contour)
        if w>width/5:# and h<height/2: 
            moment=cv2.moments(contour)
            print(moment)
           # x,y=moment
            area=cv2.contourArea(contour)
            print(area)
            perimeter = cv2.arcLength(contour,True)
            print(perimeter)
            epsilon = 0.1*cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour,epsilon,True)
            #DING!DING!DING!DING!DING!
            if cv2.isContourConvex(contour)==True:
               moment=cv2.moments(contour)
               print(moment) 
               print(approx)
             
            
            #cv2.rectangle(frame,(0,y),(1280,y+h),(255,0,0),6)
            cv2.drawContours(kmask, contour,-1,(0,0,255),1)
            cv2.circle(kmask,(x,y), 5, (0,0,255), -1)
        #print(contour)
    #cv2.imshow('contour filter',kmask)
    
    
    
    return kmask,contours
    
    

    
    
    
    
    
    
    
    
    
while True:
    

    #read initial frame
    ret, frame= cap.read()
    #grey scale
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    
    #crop frame to background region of interest
    frame=frame[0:180,0:1280]
    ero_kernal = np.ones((1,25),np.uint8)
    
    frame_mog=cv2.dilate(frame,ero_kernal,iterations = 1)
   
    mog=fgbg.apply(frame_mog)
    #cv2.imshow('mog',mog)
    canny = cv2.Canny(frame_mog, 20,100)+mog
    cv2.imshow('canny',canny)
        
    #k-means number of clusters
    K=2# with roi there should only be two key colours
    
    cv2.imshow('orig',frame)
    #original frame
    or_frame=frame
    #frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    threshold,kmask,colors = K_mask(frame, K)
    cv2.imshow('kmeans',threshold)
    scharr,sch_tours=scharr_mask(kmask,frame)
    kmask_canny=cv2.Canny(kmask, 20,100)
    
    _, track_window_mog = cv2.meanShift(threshold,(0,200,1280,2),term_criteria)
   # threshold2=cv2.cvtColor(threshold2,cv2.COLOR_BGR2HSV)
    _,contours,_=cv2.findContours(kmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour,)
        if w>width/5:# and h<height/2: 
            #cv2.rectangle(frame,(0,y),(1280,y+h),(255,0,0),6)
            cv2.drawContours(kmask, contour,-1,(255,0,255),1)
            [vx,vy,x,y] = cv2.fitLine(kmask, cv2.DIST_L2,0,0.01,0.01)
            rows,cols = kmask.shape[:2]
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            cv2.line(kmask,(cols-1,righty),(0,lefty),(0,255,0),2)
    
   
    processed=kmask #scharr+
    cv2.imshow('processed',processed)
    
    
    key=cv2.waitKey(1)
    if key ==27:
         break
     
cap.release()
cv2.destroyAllWindows()