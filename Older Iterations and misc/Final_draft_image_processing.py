
import cv2
import numpy as np
import datetime
import streamlink
import time
from collections import deque

from datetime import datetime
import json
import time



filename = 'E:\\masters_Video_cap\\test.AVI'#set video file name


streams = streamlink.streams("http://46.4.36.73/hls/bundoransurfco/playlist.m3u8")#"https://youtu.be/bNLy-XXYxcw")
quality='best'
image= cv2.VideoCapture(streams[quality].to_url())#
fps=image.get(cv2.CAP_PROP_FPS);
length=image.get(cv2.CAP_PROP_FRAME_WIDTH)
hight=image.get(cv2.CAP_PROP_FRAME_HEIGHT)
res=(length,hight)



roi_base=180
abv_avg=False
blw_avg=False
maxline_ary=[]
base_cut=99
dsp_array=deque([])
max_base=0
min_h_ary=deque([])
hcon_ary=deque([])
Drift_tracker_X=[]
Drift_tracker_X.append(1)
Drift_tracker_Y=[]
timer=0
cY=0
min_line=0
minbase=0
'''
occlusion_counter=0
abv_avg=False
max_thres =60
min_thres=30
one=0
swell_average=deque([])
drift_average=deque([])
Break_below=False
min_h_ary=deque([])
swell_check=deque([])
swell_ther=False
Swell=False
break_=False
swell_switch=False
swellcount=0
Break_abv=False
clean_Wave_count=1
wave_swell=True
Drift_reset=False
wave_count=0
base_height=0
Break_abv2=False
Break_below2=False
max_base=0
butterworth=[]
cX=0
timer=0
minute_Data_stor=deque([])
'''


#'masters_Video_cap//31_12_2018_3.mp4'28_12_2018_10   #20_1_2019_4.mp4 no good, zoomed in image #night vison fails too

#'masters_Video_cap//29_12_2018_10.mp4'--- issue with occlusion and lighting #29_12_2018_1
cap = cv2.VideoCapture('masters_Video_cap//19_1_2019_1.mp4')#31_12_2018_3.mp4')#6_1_2018_4.mp4')#6_1_2018_3.mp4')#28_12_2018_8.mp4')#29_12_2018_0.mp4')#'6_1_2018_4.mp4')#19_1_2019_1.mp4')#'My Video1.mp4')#     ITERATION THEN SHIFT AMOUNT
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,1,1) #high value means more FP #changed iterations from 10 to 2 
_,first_frame = cap.read()
height, width = first_frame.shape[:2]

fgbg = cv2.createBackgroundSubtractorMOG2()

kernel_erode = np.ones((10,8),np.uint8)
kernel_dilate= np.ones((1,32),np.uint8)
#kernel_erode_canny = np.ones((1,32),np.uint8)



def plotter():
    global max_base
    global timer
    global cX
    global minute_Data_stor

    
    
    dt = datetime.now()
    Daily_data=(str(dt.year) +'_'+str(dt.month)+'_'+str(dt.day) +'_Full_day_data.txt')
    
    if timer==0:
        try:
            data=open(Daily_data,"r").read()
            lines=data.split('\n')
            if len(lines)>1:
                v,vs,_,t=lines[len(lines)-1].split(',')
                timer=float(t)
        except:
            print('New file created')
           # data.close()
    Daily_data=open(Daily_data,"a")
    Daily_data.write('\n'+str(int(base_height))+','+str(int(max_base))+','+str(cX)+','+str(timer))#dsp_array[i])) 

    timer+=(1/30)
  #  if time==10 call dsp classs
    Daily_data.close()
  #  minute_data.close()



def convex_hull (frame,yt,ht,colour,centroid):#, #y_val):|
    global cX
    global cY
    x,y=frame.shape[:2]
    _,contours,_=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    #cv2.drawContours(r_frame,hull,-1,(255,0,255)) 
     
    _,contours,heirarchy=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    cv2.drawContours(r_frame[base_cut+yt:base_cut+yt+y,0:1280],hull,-1,(255,0,0)) 
    area=0    
    for c in contours:
        thisArea=cv2.contourArea(c)
        if thisArea>area:
            area=thisArea



    for c in contours:
        if cv2.contourArea(c)==area:
            cv2.fillPoly(r_frame[base_cut+yt:base_cut+yt+y,0:1280], pts =[c], color=(colour))
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
                cv2.circle(r_frame, (cX, base_cut+cY), 5, (0, 0, 255), -1)
                cv2.putText(r_frame, "Centre-point", (cX - 25, base_cut+cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                Drift_tracker_X.append(cX)
                Drift_tracker_Y.append(base_cut+cY)



   
    hull = [cv2.convexHull(c) for c in contours]
    
    x1,y1,w1,h1=cv2.boundingRect(swell_cons)
    if centroid==True or (min_line==max_line and minbase==max_base):
        
        cv2.circle(r_frame, (cX, base_cut+cY), 5, (0, 0, 255), -1)
        cv2.putText(r_frame, "Centre-point", (cX - 25, base_cut+cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
               
    

    return area
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



#def FiFo_brackets(remov,blw_avg,occlusion_counter,abv_avg, max_thres,min_thres):
def FiFo_brackets(blw_avg):
  
    global base_cut 
    global base_height
    global Swell
    global break_
    global max_line 
    global Break_abv
    global Break_below
    global wave_swell
    global dsp_array
    global hcon_ary
    global max_base
    global min_line
    global minbase
   # tracker(base_cut+10,BB_size,threshold_y,roi_base)
    _,contours,_= cv2.findContours(remov,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#SIMPLE
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
          if wcon>40 and wcon*hcon>100 and wcon>1280/10:#width/10 :#and hcon<30:
        
              
              # if present y value is less than all previous (appearshigher on screen)
              if (ycon)<max_line:
                  #old value for swell transformation
                 #Swell_threshold=max_base+max_line
                # past_max=max_line
                 '''#finds the heighest contour on screen'''
                 max_line=ycon
                 max_base=hcon
                 maxW=wcon
                 maxline_ary.append(ycon) 
              if max_line>base_cut-10:
                     base_cut=max_line-10
              if base_cut<max_line-10:
                     base_cut=max_line-10
              
              #adds maxline of each frame to an array
              
       
          
              '''lowest contour on screen'''
              if (ycon)>min_line :
                 min_line=ycon 
                 minbase=hcon
                 dsp_array.append(hcon)
                 base_height=hcon
                 min_h_ary.append(hcon)
                 hcon_ary.append(hcon)
                 
              if max_line>min_line and max_line<min_line+minbase :
                  max_line=min_line
                  max_base=minbase
    
    if min_line==max_line and minbase==max_base:
         cv2.putText(fifo_frame, "Occlusion Zone - Lock", (5, base_cut+min_line -10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,127,255), 2)
         cv2.line(fifo_frame,(0,base_cut+min_line),(1280,base_cut+min_line), (0,127,255),2)
         cv2.line(fifo_frame,(0,base_cut+min_line+minbase),(1280,base_cut+min_line+minbase), (0,127,255),2)
         
         
    else:             
        #Top contour 
        cv2.putText(fifo_frame, "New Swell Inbound", (5, base_cut+max_line-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)            
        cv2.line(fifo_frame,(0,base_cut+max_line),(1280,base_cut+max_line), (30,255,255),2)
        cv2.line(fifo_frame,(0,base_cut+max_line+max_base),(1280,base_cut+max_line+max_base), (150,255,255),2)
    
        #bottom contour
        cv2.putText(fifo_frame, "Break Occlusion Zone", (5,base_cut+min_line+minbase-10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)   
        cv2.line(fifo_frame,(0,base_cut+min_line),(1280,base_cut+min_line), (255,30,255),2)
        cv2.line(fifo_frame,(0,base_cut+min_line+minbase),(1280,base_cut+min_line+minbase), (255,30,255),2)
    

    
    

    colour_min=255,0,0  
    colour_max=255,125,125     
    rolling_adverage_min =remov[min_line:min_line+minbase,0:1280] 
    rolling_adverage_max =remov[max_line:max_line+max_base,0:1280] 
    swell_area =convex_hull(rolling_adverage_max,max_line,max_base,colour_min, False)
    Break_area =convex_hull(rolling_adverage_min,min_line,minbase,colour_max, True) 
              
    
    swell_area=convex_hull(rolling_adverage_max,max_line,max_base,colour_min, False)  


 
  #  cv2.imshow('fifo_frame',fifo_frame)
    
 
    return  blw_avg#,occlusion_counter, max_thres,min_thres,average_base,abv_avg,
  




#dynamic cropping of ROI
def dyno_crop(base_cut):
    global max_line

  #  print('max_line : ',max_line)
    if max_line<=10:
        if base_cut<=0:
            base_cut=3
        else:
         base_cut=base_cut-1
         while base_cut % 3>=1 :
             base_cut=base_cut-1
             if base_cut<=0:
                 base_cut=3
    if  max_line>10: 
            base_cut=base_cut+1
            while base_cut % 3>=1 :
                base_cut=base_cut+1
            
   # print('max_line',max_line)
  #  print('base_cut :', base_cut)
    if base_cut>99:
        base_cut=99
    return base_cut






'''#######################################################################################
Loop starts here ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
==========================================================================================
'''




while True:
    

    #read initial frame
    #read from file
    ret, frame= cap.read()
    
    #Ready from stream
    #ret, frame=image.read()
    #grey scale
    r_frame=frame
    empty_frame=r_frame


    ##Dynamic RoI
    try:
        base_cut=dyno_crop(base_cut)
        
        frame=frame[base_cut:roi_base,0:1280]
        #HowToGetSeaSick
        cv2.imshow('frame oscillate',frame)
    except:
    ## Generic RoI
        frame=frame[99:roi_base,0:1280]
  

    #Further filtering     
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame orig',frame)
    
    #implementing dynamic color filter
    K=2
    threshold,kmask,colors = K_mask(frame, K)
    
    cv2.imshow('kmask',kmask)
    
    #cv2.imshow('kmask',kmask)
    #NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    
    #Overlay mask on RoI
    remov=kmask+frame
    swell_val,swell_cons=swell_filter (frame)
    cv2.imshow('remov_raw',remov)
    #Binaries image
    ret,remov=cv2.threshold(remov,100,255,cv2.THRESH_BINARY_INV)
   #remov=cv2.adaptiveThreshold(remov,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,3)
    

    cv2.imshow("remov filter",remov)
    remov2=remov
    
    #Add contour values
    remov= swell_val+remov
    cv2.imshow('data frame',remov)
    
        #mavbe add mog to k mask but leave canny, thus leaving clear lines at the swell and outline as 
    #with the break, caviate may be residual noise 
    
    #remov=fgbg.apply(remov)
    
    
    '''
    ####End of base filter===========================================================================
    =====================================================================================================
    =====================================================================================================
    '''
    #create timer
    
    
    cv2.imshow('base filter',remov)
    

    '''
    ============================================================================
    Good up to here, following an attempt at MOG filtering for False positives
    ============================================================================
    '''
 

  
    # call to add bottom and top trackers 
    abv_avg=FiFo_brackets(abv_avg)
   
 
    #Function to add mask over break and find wave moment
 #   Occlusion_counter()

    

    cv2.imshow('frame',frame)
    cv2.imshow('R_frame',r_frame)    
    

    
   #Store Data As external file
    plotter()
    
    
    
    #Controled_exit
    key=cv2.waitKey(1)
    if key ==27:
         break
     

     
cap.release()
cv2.destroyAllWindows()

    


    
    
    