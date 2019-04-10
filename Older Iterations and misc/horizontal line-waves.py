'''
this file is generate to erode vertically and dilute horzontally 
to the extent that all waves morph into lines

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('19_1_2019_1.mp4')#'My Video1.mp4')#
fgbg=cv2.createBackgroundSubtractorMOG2(history=-1,detectShadows = False)#,varThreshold=3

_,first_frame = cap.read()
height, width = first_frame.shape[:2]
ero_hor_kernal = np.ones((1,15),np.uint8)
dil_hor_kernal = np.ones((5,20),np.uint8)
dil_vert_kernal= np.ones((5,1),np.uint8)
ero_kernal= np.ones((2,4),np.uint8)
cny_ero_kernal= np.ones((1,20),np.uint8)
cny_dil_kernal= np.ones((1,20),np.uint8)
mog_dil_kernal= np.ones((1,30),np.uint8)

frame2=cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

height, width = first_frame.shape[:2]





#THRESHOLDING black& white  binary, zro,trunc
#erode after binary
#scharr
#histogram
#opening and closing

#morphologycal best on binary image
#kmeans seperates colors

#absolute difference
kernal = np.ones((1,5),np.uint8)
j=0






while True:
    #read initial frame
    ret, frame= cap.read()
    
    
    
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    Z=frame.reshape((-1,3))
    Z=np.float32(Z)

    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

    K=5
    ret, label1,center1=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center1=np.uint8(center1)
    res1=center1[label1.flatten()]
    output1=res1.reshape((frame.shape))
    cv2.imshow('kmeans',output1)
    #cv2.imshow('kmeans',frame)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    frame_mog=frame
    frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((4,4),np.uint8)
    t=60
    
    if j<t:
        j=j+1
        if j==t:
            #filter by previous frames?
            frame=cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 253, 13)
            d= cv2.absdiff(frame,frame2)
            d=cv2.dilate(d,kernal,iterations = 1)
            d = cv2.morphologyEx(d, cv2.MORPH_OPEN, kernel)
            cv2.imshow('movement',d)
            frame2=frame
            j=0
 

    
    
    
    
    kernel = np.ones((2,5),np.uint8)
    frame=cv2.erode(frame,ero_kernal,iterations = 5 )
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    cv2.imshow('frame open',frame)
    #frame = cv2.medianBlur(frame,17)
    #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray=frame                                            #~100-200
    retval, threshold = cv2.threshold(img_gray, 155, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary',threshold )
    #adaptive filter is good if not a bit noisy


    th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 253, 13)
    cv2.imshow('Adaptive threshold',th)
    retval2,threshold2 = cv2.threshold(img_gray,155,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('Otsu threshold',threshold2)
    #cv2.imshow('sobelx',sobelx)
    cv2.imshow('frame',frame)
 
    
     #median blur #not great

    
 
   
    #scharr much better that >>sobel
    scharr=cv2.Scharr(frame_mog,-1,dx=0,dy=1,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('Scharr',scharr)
    
    
    
    
    kernel = np.ones((4,10),np.uint8)
    opening = cv2.morphologyEx(threshold2, cv2.MORPH_OPEN, kernel)
    cv2.imshow('scharr_otsu_opening',opening)
    
    scharr_otsu=cv2.Scharr(opening,-1,dx=0,dy=1,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    cv2.imshow('scharr_otsu',scharr_otsu)

    '''
    #touch under pain of death

    '''
    erosion=cv2.erode(frame,ero_kernal,iterations = 5 )
    #cv2.imshow('erosion',erosion)
   # mask = cv2.Canny(erosion, 40,100) 
    #cv2.imshow('mask',mask)
    
    #apply mog2
    mog=fgbg.apply(frame)
    #cappy edge detection
    mask = cv2.Canny(frame_mog, 40,100)    
    
    
    dilation=cv2.dilate(frame,ero_kernal,iterations = 2)
    #cv2.imshow('dilation',dilation)

    erosion= cv2.erode(dilation,ero_kernal,iterations = 2)
    #cv2.imshow('erosion',erosion)
    mask = cv2.Canny(dilation, 40,100)  
   # cv2.imshow('canny',mask)
    mog=fgbg.apply(dilation)
    #cv2.imshow('mog',mog)
    
    
 #   _,contours,_ = cv2.findContours(Erode_hor,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    
 #   for contour in contours:
       # area = cv2.contourArea(contour)
        
  #      cv2.drawContours(Erode_hor,contour,-1,(255,0,255),1)
    
   #     cv2.imshow('frame_mog',Erode_hor)
    
    
    
    
    













    
    '''

    _,contours, _ =cv2.findContours( sobely ,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    mt=0
    for contour in contours:
        area = cv2.contourArea(contour)
       # if area>mt :
        #    mt=area
       # if area==mt:
        cv2.drawContours(frame, contour,-1,(0,0,255),1)
   # print(mt)
     
    
   # cv2.drawContours(frame, contours,-1,(0,0,255),1)
    cv2.imshow('',frame)
    '''
    key=cv2.waitKey(1)
    if key ==27:
         break
     
cap.release()
cv2.destroyAllWindows()