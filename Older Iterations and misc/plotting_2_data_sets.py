#data  plotting

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


##convwx hull
import cv2
import numpy as np
import datetime
import time









   


roi_base=180
Former_BB=0 # occlusion prevention

startyt=120
ht=roi_base
yt= startyt  
ycan=startyt 
hcan=roi_base
threshold_y=yt
threshold_ycan=ycan
threshold_ymog=yt
wave_count=0;
bb_org=25
BB_size=bb_org
BB_size_2=bb_org
BB_size_3=bb_org
period_Per_min=0
period_average=0
waves_all_day=0
mint_cnt=datetime.time.minute
lastMarker=time.time()
mixcout=0
cannycount=0
mogcount=0
tracker=[]
peak=[]
base=[]
time=[]
max_base=30






##plotting GUI

style.use('fivethirtyeight')

fig=plt.figure()
ax1=fig.add_subplot(1,1,1)


###search for contours in ROI the track
##set contour at top of page as start point/ roi
##mog for flase positives?
##filter contour by size!!!! for surfers
#try just canny
#canny with kmask at bottom of ROI?

cap = cv2.VideoCapture('19_1_2019_1.mp4')#'My Video1.mp4')#     ITERATION THEN SHIFT AMOUNT
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,1,1) #high value means more FP #changed iterations from 10 to 2 
_,first_frame = cap.read()
height, width = first_frame.shape[:2]

fgbg = cv2.createBackgroundSubtractorMOG2()

kernel_erode = np.ones((10,8),np.uint8)
kernel_dilate= np.ones((1,32),np.uint8)
#kernel_erode_canny = np.ones((1,32),np.uint8)


    
        
    

def convex_hull (frame,yt):#, #y_val):
    x,y=frame.shape[:2]
    contours,heirarchy=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    #cv2.drawContours(r_frame,hull,-1,(255,0,255)) 
     
    contours,heirarchy=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    cv2.drawContours(r_frame[yt:yt+y,0:1280],hull,-1,(255,0,0)) 
    area=0    
    for c in contours:
        thisArea=cv2.contourArea(c)
        if thisArea>area:
            area=thisArea
#    for c in contours:
        #calibrate for surfers,
       # x, y, w, h = cv2.boundingRect(c)
        #if h<10 and w<40:
         #   print('swell detected')
            
            
    #print('present area size',area)
    for c in contours:
        if cv2.contourArea(c)==area:
            cv2.fillPoly(r_frame[yt:yt+y,0:1280], pts =[c], color=(255,255,255))
            #store in an array and find the largest area for peak wave size
    
   # cv2.imshow('new',frame[y_val:roi_base,0:1280])#new image
    #cv2.imshow('old',frame[0:y_val,0:y])#old image
   
    hull = [cv2.convexHull(c) for c in contours]
    
    x1,y1,w1,h1=cv2.boundingRect(swell_cons)
    
    
    
   # final =cv2.drawContours(frame,hull,-1,(255,0,255))
   # cv2.drawContours(r_frame,hull,-1,(255,0,255))
    #cv2.imshow('Sub_Hull',r_frame)
    

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
    cv2.imshow('canny filter',can_test)
    swell_cons=can_test
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
    remov2=remov
    remov= swell_val+remov
    cv2.imshow('data frame',remov)
    
        #mavbe add mog to k mask but leave canny, thus leaving clear lines at the swell and outline as 
    #with the break, caviate may be residual noise 
    
    #remov=fgbg.apply(remov)
    
    
    '''
    ####no problems up to here===========================================================================
    =====================================================================================================
    =====================================================================================================
    '''
    #create timer
    
    
    cv2.imshow('base filter',remov)
    
    
    
    
    _, track_window_break_1 = cv2.meanShift(remov,(0,yt,1280,BB_size),term_criteria)#remov

    xt,yt,wt,ht=track_window_break_1
    

    
    
    
   # print(xt,yt,wt,ht)
    #reset value
    if yt+ht==roi_base:
        BB_size=BB_size-5
        yt=yt+5
        if BB_size==5:
            yt=100
            xt,yt,wt,ht=0,yt,1280,BB_size
            track_window_break=(0,100,1280,BB_size)
            ht=BB_size
            threshold_y=100
            mixcout=mixcout+1
            #print('mixcout',mixcout,time.time())
            BB_size=bb_org
            period_Per_min=period_Per_min+1
            waves_all_day=waves_all_day+1
            marker=time.time()#recieves time of crossing
            #print('marker',(marker-lastMarker))
            frequency=(marker-lastMarker)
            lastMarker=marker
            #print('frequency',frequency)
   
    '''uni-directional movement'''
    if threshold_y>yt:
         yt=threshold_y
    threshold_y=yt
#added
    cv2.rectangle(r_frame, (0,yt),( 1280,yt+ht), (0,255,0), 2)
    
    
    
    
    
    remov_tracker=remov[yt:roi_base,xt:xt+wt]  
    _, track_window_break_2 = cv2.meanShift(remov_tracker,(0,0,wt,BB_size),term_criteria)
    xt2,yt2,wt2,ht2=track_window_break_2
    #print(xt2,yt2,wt2,ht2)
    #cv2.imshow('remov_tracker',remov_tracker)
    #convex_hull(remov_tracker,yt)
    cv2.rectangle(r_frame, (wt2,yt+yt2),( wt2+ht2,yt+yt2+ht2), (255,0,0), 2)
    



    centre_line= cv2.line(r_frame,(0,roi_base),(1280,roi_base),(0,0,255),2)

    '''
    ============================================================================
    Good up to here, following an attempt at MOG filtering for False positives
    ============================================================================
    '''

        #canny------------> contour 
        #if contour width>height or width< screenwidth/x  : remove from filter
    con_mask = np.ones(first_frame.shape[:2], dtype="uint8") * 255  
    contours,_ = cv2.findContours(remov,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    max_line=720
    max_con=contours[0]
    #loop through all contours
    for contour in contours:
            #get bounding boxes
          xcon,ycon,wcon,hcon=cv2.boundingRect(contour)
          #draw boxes for trouble shooting
          #if wcon*hcon>100 or wcon>40:
             # cv2.rectangle(r_frame,(xcon,ycon),(xcon+wcon,ycon+hcon),(255,255,30),3)
         # if (wcon*hcon<100 or wcon<20)and wcon*hcon>20:
             # cv2.rectangle(r_frame,(xcon,ycon),(xcon+wcon,ycon+hcon),(255,30,255),1)
          
            #if the bounding box is wider than x 
          if wcon>40 and wcon*hcon>100:#width/10 :#and hcon<30:
              #fitline # get highest line
              
              # if present y value is less than all previous
              if (ycon)<max_line:
                 max_line2=max_line 
                 max_line=ycon
                 max_base2=max_base
                 max_base=hcon
               # maxX2=maxX
                 maxX=xcon
              #  maxW2=maxW
                 maxW=wcon
                 max_con=contour
              cv2.drawContours(con_mask, contour, -1, 0, -1) 
    rolling_adverage=remov[max_line:max_line+max_base,0:width] 
    convex_hull(rolling_adverage,max_line)
    cv2.line(r_frame,(0,max_line),(width,max_line), (30,255,255),2)
    cv2.line(r_frame,(0,max_line+max_base),(width,max_line+max_base), (150,255,255),2)
    
    #cv2.line(r_frame,(0,max_line2),(width,max_line2), (0,0,255),2)
    #cv2.line(r_frame,(0,max_line2+max_base2),(width,max_line2+max_base2), (0,0,255),2)

    cv2.imshow('contour filter',con_mask)

         
    
    
    
    
    
    
    
    
    
    
    
    break_mask = np.ones(first_frame.shape[:2], dtype="uint8") * 255  
    contours_2,_ = cv2.findContours(remov,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    
    for contour in contours_2:
          xcon,ycon,wcon,hcon=cv2.boundingRect(contour)
          #if wcon<width/5 :
          if hcon<wcon and wcon>width/10 and hcon>30:
              cv2.drawContours(break_mask, contour, -1, 0, -1)   
               
    cv2.imshow('other than swell filter',break_mask)
         
    time.append(time)
    tracker.append(yt)
    peak.append(ycon)
    base.append(hcon)
    ani = animation.FuncAnimation(fig,animate(tracker,peak,base,time),interval=1000)
    plt.show()
    
    
    
    
    
    
    
    print('readings_Tracker:\t',yt,ht,'\t\tReadings_max and Min:\t',max_line,max_base)
    
    
    cv2.imshow('frame',frame)
    cv2.imshow('R_frame',r_frame)    
    
    key=cv2.waitKey(1)
    if key ==27:
         break
     
     
cap.release()
cv2.destroyAllWindows()

    


    
    
    