
import cv2
import numpy as np
import datetime
import time
from collections import deque






roi_base=180
max_line=97
base_cut=99
startyt=base_cut
Drift_tracker_X=[]
Drift_tracker_Y=[]
maxline_ary=[]
minline_ary=[]
min_h_ary=deque([])
threshold_base=[]
threshold_max=[]
threshold_min=[]
occlusion_counter=0
blw_avg=False
abv_avg=False
max_thres =(roi_base-base_cut)/2
min_thres=(roi_base-base_cut)/2
swell_average=deque([])
threshold_y=base_cut
bb_org=15
BB_size=bb_org
ht_fix=5#(25/180)*(roi_base-base_cut)#roi_base
break_detect=False
swell_detect=False
Break_abv=False
Break_below=False
Wave_avg=deque([])
wave_count=0

local_ar=deque([])
counter_abv=0
counter_blw=0
one=0




cap = cv2.VideoCapture('masters_Video_cap/19_1_2019_1.mp4')#28_12_2018_8.mp4')#29_12_2018_0.mp4')#'6_1_2018_4.mp4')#19_1_2019_1.mp4')#'My Video1.mp4')#     ITERATION THEN SHIFT AMOUNT
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,1,1) #high value means more FP #changed iterations from 10 to 2 
_,first_frame = cap.read()
height, width = first_frame.shape[:2]



kernel_erode = np.ones((10,8),np.uint8)
kernel_dilate= np.ones((1,32),np.uint8)
#kernel_erode_canny = np.ones((1,32),np.uint8)


    
        
    
#grand
def convex_hull (frame,yt,ht,colour,centroid):#, #y_val):
    con_frame=empty_frame
    x,y=frame.shape[:2]
    contours,_=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    #cv2.drawContours(r_frame,hull,-1,(255,0,255)) 
     
    contours,heirarchy=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    cv2.drawContours(con_frame[base_cut+yt:base_cut+yt+y,0:1280],hull,-1,(255,0,0)) 
    area=0    
    for c in contours:
        thisArea=cv2.contourArea(c)
        if thisArea>area:
            area=thisArea
#    for c in contours:
        #calibrate for surfers
       # x, y, w, h = cv2.boundingRect(c)
        #if h<10 and w<40:
         #   print('swell detected')
            
            
    #print('present area size',area)
    for c in contours:
        if cv2.contourArea(c)==area:
            cv2.fillPoly(con_frame[base_cut+yt:base_cut+yt+y,0:1280], pts =[c], color=(colour))
            M=cv2.moments(c)
            
            # calculate x,y coordinate of center
            try:
                cX = int(M["m10"] / M["m00"])
                cY = yt+int(ht/2)
            except:
                cX = int(M["m10"] /1)
                cY = int(M["m01"] / 1)
            
            # put text and highlight the center
            if centroid==True:
                cv2.circle(con_frame, (cX, base_cut+cY), 5, (0, 0, 255), -1)
                cv2.putText(con_frame, "moment", (cX - 25, base_cut+cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                Drift_tracker_X.append(cX)
                Drift_tracker_Y.append(base_cut+cY)
    #
            #store in an array and find the largest area for peak wave size
    
   # cv2.imshow('new',frame[y_val:roi_base,0:1280])#new image
    #cv2.imshow('old',frame[0:y_val,0:y])#old image
   
    hull = [cv2.convexHull(c) for c in contours]
    
    x1,y1,w1,h1=cv2.boundingRect(swell_cons)
    
    
    
   # final =cv2.drawContours(frame,hull,-1,(255,0,255))
   # cv2.drawContours(r_frame,hull,-1,(255,0,255))
    #cv2.imshow('Sub_Hull',r_frame)
    

    return area
#function for K_means filtering
    


#Grand -dynamic colour filter
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




#simple canny filter (Grand)
def swell_filter (frame):
    # canny threshold?
    swell_cons=frame
    can_test=cv2.Canny(frame, 40,100)
    #cv2.imshow('canny filter',can_test)
    swell_cons=can_test
    return swell_cons,can_test




































def FiFo_brackets(remov,blw_avg,occlusion_counter,abv_avg, max_thres,min_thres):

    global Break_abv
    global Break_below
    global base_cut
    global wave_count
    global one
    global max_line
    
    #tracker(base_cut+10,BB_size,threshold_y,roi_base)
    contours,_= cv2.findContours(remov,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#SIMPLE
    blw_avg=blw_avg
    
    fifo_frame=empty_frame
    
    max_line=height
    min_line=height

    max_base=0
   
    minbase=0
    min_line=0

    
    
 
    for contour in contours:
             
          xcon,ycon,wcon,hcon=cv2.boundingRect(contour)
          
          
          
              # width and area used to filter noise
          if wcon>40 and wcon*hcon>100:#width/10 :#and hcon<30:
        
              
              # if present y value is less than all previous (appearshigher on screen)
              if (ycon)<max_line:
                  #old value for swell transformation
                 #Swell_threshold=max_base+max_line
                # past_max=max_line
                 '''#finds the heighest contour on screen'''
                 max_line=ycon
                 max_base=hcon
                 maxline_ary.append(ycon) 
              if max_line>base_cut-10:
                     base_cut=max_line-10
              if base_cut<max_line-10:
                     base_cut=max_line-10
              
              #adds maxline of each frame to an array
              
       
          
              '''lowest contour on screen'''
              if (ycon)>min_line and wcon>width/5:
                 min_line=ycon 
                 minbase=hcon
         
                 min_h_ary.append(hcon)
                 one= sum(min_h_ary)/len(min_h_ary)
                 #threshold_base.append(hcon)
                 
              '''  
              ============================================================================================================
              ============================================================================================================
              ============================================================================================================
              ============================================================================================================
              '''    
              #if swell is true create tracker to average 
                 
              if hcon+2<one :
               swell_average.append(0)
               
          
                    

                  #  print('below average')
              if hcon-2>one:  
                    swell_average.append(1)
                 
                    
                    
     
                   
                   
              if len(swell_average)>10:
                  swell_average.popleft()
            
              
             # print (sum(swell_average))#/len(swell_average))#/len(swell_average))
              if sum(swell_average)>=5:
                  Break_abv=True
                  
              if sum(swell_average)==0:   
                  Break_below=True
                  
          #    if Break_abv==False:
          #       Break_below=False
                
              if Break_abv==False and Break_below==True:
                  Break_below=False
               #]   Break_below==False
                
              if Break_abv==True and Break_below==True and hcon>one: 
                    
                    Break_abv=False
                    Break_below=False
                    wave_count=wave_count+1
                    print('wave ',wave_count,' detected')
                
             
             
             
              if len(min_h_ary)>=150:
                    # Wave_avg.popleft()
              #if len(min_h_ary)>=30:#implement standard deviation?
                     min_h_ary.popleft()
                  
                     
             # 
             # old_local=local
      
              cv2.line(fifo_frame,(0,int(roi_base-one)),(width,int(roi_base-one)), (0,0,255),2)    
        
                 

              try:
                  sum_base=sum(threshold_base)
                  no_base=len(threshold_base)
                  base_average=sum_base/no_base

              except:
                  base_average=20
                  
                  
                  
                  
              colour_min=255,0,0  
              colour_max=255,125,125     
              rolling_adverage_min =remov[min_line:min_line+minbase,0:width] 
              rolling_adverage_max =remov[max_line:max_line+max_base,0:width] 
              swell_area =convex_hull(rolling_adverage_max,max_line,max_base,colour_min, False)
              Break_area =convex_hull(rolling_adverage_min,min_line,minbase,colour_max, True) 
              
    
                 

    #top contour
    cv2.line(fifo_frame,(0,base_cut+max_line),(width,base_cut+max_line), (30,255,255),2)
  
    cv2.line(fifo_frame,(0,base_cut+max_line+max_base),(width,base_cut+max_line+max_base), (150,255,255),2)
    
    #bottom contour
    cv2.line(fifo_frame,(0,base_cut+min_line),(width,base_cut+min_line), (255,30,255),2)
    cv2.line(fifo_frame,(0,base_cut+min_line+minbase),(width,base_cut+min_line+minbase), (255,30,255),2)

 
    cv2.imshow('fifo_frame',fifo_frame)
    
    average_base=0
    return abv_avg, blw_avg,occlusion_counter, max_thres,min_thres,average_base






















#Grand  
    '''
    Return value needed i.e. drifting left/ drifting right
    '''
def  centroid_analyses(r_frame):
    
    F_centroid=empty_frame
    try:
        average= sum(Drift_tracker_X)/len(Drift_tracker_X)

        
        #cv2.line(F_centroid,(leftb,botb),(leftb,topb), (255,255,255),2)
        #cv2.line(F_centroid,(rightb,botb),(rightb,topb), (255,255,255),2)
        cv2.line(r_frame,(int(average),base_cut),(int(average),roi_base), (0,255,0),2)
        cv2.line(r_frame,(int(width/2),base_cut),(int(width/2),roi_base), (255,255,0),2)
        cv2.putText(r_frame," Screen centre",(int(width/2),roi_base-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 1) 
        cv2.putText(r_frame," Mean shift",(int(average),base_cut+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    except:
       1+1

    
    cv2.imshow('Deviation',F_centroid)







# starting poit, trackbox size, passed value as reference for uni-direction
def tracker(yt,BB_size,threshold_y,base):
 
    tracke_frame=empty_frame
    _, track_window_break_1 = cv2.meanShift(remov,(0,yt,width,BB_size),term_criteria)
    
#location of tracker window
    xt,yt,wt,ht=track_window_break_1
   
# print(xt,yt,wt,ht)
    #reset value
    #set tracker at mean Point of the wave!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if yt+ht+base_cut>=roi_base-10:
        remove_c=int((25/180)*5)+1
       
        BB_size=BB_size-remove_c
        print('BB_size:',BB_size)
        yt=yt+remove_c
        '''
        if BB_size<=remove_c+3:
            print('BB_size',BB_size)
            yt=startyt 
            xt,yt,wt,ht=0,startyt,width,bb_org
            track_window_break=(0,100,1280,bb_org)
            ht=ht_fix
            threshold_y=int((81/180)*100)
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
         '''
   # uni-directional movement
    if threshold_y>yt:
         yt=threshold_y
    threshold_y=yt
#added
   # print(yt,yt+ht)                  #yt+ht
    try:
       cv2.rectangle(r_frame, (0,yt),( 1280,yt+ht), (0,255,0), 2)
    #base_cut+base_cut+
    except:
       cv2.rectangle(r_frame, (0,base_cut),( 1280,base_cut+yt), (0,255,0), 2)
    
    
    '''
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    APEX is moment of convex hull, side movement can be derived by tracking moment across screen
    '''
    
   # remov_tracker=remov[yt:roi_base,xt:xt+wt]  

    cv2.imshow('tracker',tracke_frame)
    return yt,BB_size,threshold_y


    '''
  _, track_window_break_1 = cv2.meanShift(remov,(0,yt,1280,BB_size),term_criteria)

    xt,yt,wt,ht=track_window_break_1
   
    

    
    
    
   # print(xt,yt,wt,ht)
    #reset value
    #set tracker at mean Point of the wave!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if yt+ht+base_cut>=roi_base-10:
        remove_c=int((25/180)*5)+1
       
        BB_size=BB_size-remove_c
        print('BB_size:',BB_size)
        yt=yt+remove_c
        
        if BB_size<=remove_c+3:
            print('BB_size',BB_size)
            yt=startyt 
            xt,yt,wt,ht=0,startyt,1280,bb_org
            track_window_break=(0,100,1280,bb_org)
            ht=ht_fix
            threshold_y=int((81/180)*100)
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
   
   # uni-directional movement
    if threshold_y>yt:
         yt=threshold_y
    threshold_y=yt
#added
   # print(yt,yt+ht)                  #yt+ht
    try:
       cv2.rectangle(r_frame, (0,base_cut+yt),( 1280,base_cut+yt+ht), (0,255,0), 2)
    #base_cut+base_cut+
    except:
       cv2.rectangle(r_frame, (0,base_cut),( 1280,base_cut+yt), (0,255,0), 2)


    '''








    '''
    oldCut=base_cut
    base_cut= max_line-5
    while (base_cut % 3>=1 or base_cut>100)and base_cut>=oldCut :
        base_cut=base_cut-1
    #if base_cut<=0:
    while (base_cut % 3>=1 or base_cut <=0  or base_cut<50) and base_cut<=oldCut:
            base_cut=base_cut+1
    if base_cut>roi_base:
       base_cut= oldCut
    print(base_cut)
    '''
        
    #cv2.imshow('dyno_crop',frame)
    #return frame, base_cut 












'''#######################################################################################
Loop starts here ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
==========================================================================================
'''




while True:
    

    #read initial frame
    ret, frame= cap.read()
    #grey scale
    empty_frame=frame
    r_frame=frame
    
 #   try:
 #       base_cut=max(maxline_ary)-7
 #   except:
    frame=frame[base_cut:roi_base,0:1280]
    cv2.imshow('frame',frame)
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame orig',frame)
    
    
    K=2
    
    threshold,kmask,colors = K_mask(frame, K)
    
    #cv2.imshow('kmask',kmask)
    
    #cv2.imshow('kmask',kmask)
    #NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    remov=kmask+frame
    swell_val,swell_cons=swell_filter (frame)
   # cv2.imshow('remov_raw',remov)
    ret,remov=cv2.threshold(remov,100,255,cv2.THRESH_BINARY_INV)
   #remov=cv2.adaptiveThreshold(remov,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,3)
    
    
    
    
    
    
    
    
    
    
    #cv2.imshow("remov filter",remov)
    remov2=remov
    remov= swell_val+remov
   # cv2.imshow('data frame',remov)
    
        #mavbe add mog to k mask but leave canny, thus leaving clear lines at the swell and outline as 
    #with the break, caviate may be residual noise 
    
    #remov=fgbg.apply(remov)
    
    
    '''
    ####no problems up to here===========================================================================
    =====================================================================================================
    =====================================================================================================
    '''

      
    con_mask = np.ones(first_frame.shape[:2], dtype="uint8") * 255  
    '''Commented out for testing'''
    abv_avg, blw_avg,occlusion_counter,max_thres,min_thres,average=FiFo_brackets(remov,blw_avg,occlusion_counter,abv_avg, max_thres,min_thres)# call to add bottom and top trackers
    centroid_analyses(r_frame)
    '''testing ROI_base complete'''
    
    #yt,BB_size,threshold_y=tracker(yt,BB_size,threshold_y,average)
    #yt,BB_size,threshold_y=tracker(base_cut,5,threshold_y,45)

    #cv2.imshow('frame',frame)
    cv2.imshow('R_frame',r_frame)    
    
    

    
    
   # base_cut=dyno_crop(base_cut)
    key=cv2.waitKey(30)
    if key ==27:
         break
     
     
cap.release()
cv2.destroyAllWindows()

    


    
    
    