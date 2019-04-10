##convwx hull
import cv2
import numpy as np


roi_base=180
threshold_y=0
Former_BB=0 # occlusion prevention


#import matplotlib.pyplot as plt

###search for contours in ROI the track

cap = cv2.VideoCapture('19_1_2019_1.mp4')#'My Video1.mp4')#     ITERATION THEN SHIFT AMOUNT
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,5)
_,first_frame = cap.read()
height, width = first_frame.shape[:2]



kernel_erode = np.ones((10,8),np.uint8)
kernel_dilate= np.ones((1,32),np.uint8)
kernel_erode_canny = np.ones((1,32),np.uint8)


def m_roi(frame):
    print (x1,y1,w1,h1)
    m_ROI=frame[y1:y1+h1,x1:x1+w1]
    cv2.imshow('new frame',m_ROI)
    #new roi and convex hull!!!
    
    return frame




def convex_hull (frame):
    _,contours,heirarchy=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.imshow('remov',remov)
   
    hull = [cv2.convexHull(c) for c in contours]
    x1,y1,w1,h1=cv2.boundingRect(swell_cons)
    final =cv2.drawContours(frame,hull,-1,(255,0,255))
    cv2.drawContours(r_frame,hull,-1,(255,0,255))
    cv2.imshow('Sub_Hull',r_frame)
    

    return frame
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

    return threshold,output1,center1 

def swell_filter (frame):
    # canny threshold?
    swell_cons=frame
    can_test=cv2.Canny(frame, 40,100)
    swell_cons=can_test
    cv2.imshow('swell_cons',swell_cons)
    return swell_cons,can_test










while True:
    
   
    #read initial frame
    ret, frame= cap.read()
    #grey scale
    r_frame=frame
    frame=frame[0:roi_base,0:1280]
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame orig',frame)
    
    
    K=2
    
    threshold,kmask,colors = K_mask(frame, K)
    
    
    
    #cv2.imshow('kmask',kmask)
    #NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    remov=kmask+frame
    swell_val,swell_cons=swell_filter (frame)
   # cv2.imshow('remov_raw',remov)
    ret,remov=cv2.threshold(remov,140,255,cv2.THRESH_BINARY_INV)
    
    remov=remov+swell_val
    '''
    ####no problems up to here===========================================================================
    =====================================================================================================
    =====================================================================================================
    '''
    #window tracker, may need to be looped or threaded
    '''maybe create a method to loop through trackers'''
    _, track_window_break = cv2.meanShift(remov,(0,100,1280,20),term_criteria)
    #window tracker values 
    xt,yt,wt,ht=track_window_break

    

    if yt<threshold_y:#if the barrier wants to move up split into smaller ROI's
        yt=threshold_y
   
    threshold_y=yt
        

   # print (xt,yt,wt,ht)
    cv2.rectangle(frame,(0,yt),(1280,(yt+ht)),(0,255,0),2)
    #centre_line= cv2.line(r_frame,(0,int(yt+(ht/2))),(1280,yt+int(ht/2)),(0,255,0),2)
    
    
    _,contours,heirarchy=cv2.findContours(remov,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    
    cv2.imshow('remov',remov)
   
    
    hull = [cv2.convexHull(c) for c in contours]
    #moments = cv2.moments(hull)
    
    
            #index of contour =-1

    x1,y1,w1,h1=cv2.boundingRect(remov)#swell_cons)
    if threshold_y>y1:
        y1=threshold_y
        m_roi(remov)
        threshold_y=y1
        
        # print (x1,y1,w1,h1)
        #m_ROI=frame[y1:y1+h1,x1:x1+w1]
        #cv2.imshow('new frame',m_ROI)
        #new roi and convex hull!!!
        
        
    
    
    
    final =cv2.drawContours(frame,hull,-1,(255,0,255))
    cv2.drawContours(r_frame,hull,-1,(255,0,255))
#else print previous convex hulls
    
    x,y,w,h=cv2.boundingRect(final)#got area use previous to stop occlusion
    cv2.rectangle(r_frame, (x1,y1),( x1+w1,y1+h1), (0,255,0), 1)    #tracker box

    #max height
    #rolling height
    #draw horizontal centre line
    M = cv2.moments(final)
  
  
   # print ('centre point',M)

 
    cv2.imshow('frame',frame)
    cv2.imshow('R_frame',r_frame)
    key=cv2.waitKey(20)
    if key ==27:
         break
     
cap.release()
cv2.destroyAllWindows()
























