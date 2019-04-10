
import cv2
import numpy as np
import datetime
import streamlink
import time
from collections import deque
import requests
from datetime import datetime
import json
import time
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, butter,lfilter, freqz
import os


#plt.rcParams['animation.html']='jshtml'



#import csv





filename = 'E:\\masters_Video_cap\\test.AVI'#set video file name


streams = streamlink.streams("http://46.4.36.73/hls/bundoransurfco/playlist.m3u8")#"https://youtu.be/bNLy-XXYxcw")
quality='best'
image= cv2.VideoCapture(streams[quality].to_url())#
fps=image.get(cv2.CAP_PROP_FPS);
length=image.get(cv2.CAP_PROP_FRAME_WIDTH)
hight=image.get(cv2.CAP_PROP_FRAME_HEIGHT)
res=(length,hight)














roi_base=180
blw_avg=False
occlusion_counter=0
abv_avg=False
max_thres =60
min_thres=30
maxline_ary=[]
base_cut=99
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
Drift_tracker_X=[]
Drift_tracker_X.append(1)
Drift_tracker_Y=[]
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
dsp_array=deque([])
butterworth=[]
timer=0
hcon_ary=deque([])
cX=0
timer=0
minute_Data_stor=deque([])
#plot setup
#fig=plt.figure()
#ax=fig.add_subplot(111)
#fig.show()


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

'''
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

'''



def plotter():
    global max_base
    global timer
    global cX
    global minute_Data_stor
    x=[]
    y=[]
    order = 3
    fs = 1800   # sample rate, Hz
    cutoff = 300
    i=len(dsp_array)
    
    
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
    #minute_Data_stor.append('\n'+str(int(base_height))+','+str(int(max_base))+','+str(cX)+','+str(timer))
  #  try:
   #     if timer % 10==0:
  #          os.remove('Minute_history.txt')
    #except:
   # open('Minute_history.txt', 'w').close()
   # minute_data=open('Minute_history.txt',"a") 
    
   # for sec in minute_Data_stor:
   #     minute_data.write(sec)    
    
  #  if len(minute_Data_stor)>5400:
  #      minute_Data_stor.popleft()
    timer+=(1/30)
  #  if time==10 call dsp classs
    Daily_data.close()
  #  minute_data.close()

        
    
    
    
    
'''
def plotter():
    global max_base
  #  x=[]
  #  y=[]
 #   order = 3
#    fs = 1800   # sample rate, Hz
#    cutoff = 300
#    i=len(dsp_array)
    #print(len(dsp_array))
            # x.append(i)
   # y.append(dsp_array[i])
   # print(dsp_array[i])
            #ax.plot(1, 1 ,color='r') 
    dt = datetime.now()
    
   # try:
    data=open("DSP_data_AVG.txt","a")#.write(str(y[i]))
    data.write('\n'+str(int(one))+','+str(max_base)+','+str(cX)+','+str((dt.minute*60)+dt.second))#dsp_array[i])) 
           
    data.close()
  #  except:
  #              print('nope')
    
            #if len(y)>=60:
  #  lp = butter_lowpass_filter(y, cutoff, fs, order)
   # fig.clear()    
 #   ax.plot(x, lp ,color='r')
        
 #   fig.canvas.draw()
              
    #ax.set_xlim(left=max(0,i-400),right=i+5)
        #    i+=1

       #     timer=0
        
       # timer+=1
 #   print(timer)
        

   # if len(dsp_array)==3600:
   #     dsp_array.popleft()
   #     x.clear()
   #     y.clear()

'''        
    
   
   
   
   
   
   
   
   

def convex_hull (frame,yt,ht,colour,centroid):#, #y_val):|
    global cX
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
#    for c in contours:
        #calibrate for surfers
       # x, y, w, h = cv2.boundingRect(c)
        #if h<10 and w<40:
         #   print('swell detected')
            
            
    #print('present area size',area)
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
                cv2.putText(r_frame, "centroid", (cX - 25, base_cut+cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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



def FiFo_brackets(remov,blw_avg,occlusion_counter,abv_avg, max_thres,min_thres):

    global Break_abv
    global Break_below
    global base_cut
    global swell_switch
    global break_switch
    global one
    global swell_ther
    global swell_brok
    global swellcount
    global FiFo_detect
    global Swell
    global break_
    global max_line
    global base_height
    global clean_Wave_count
    global wave_swell
    global dsp_array
    global hcon_ary
    global max_base
    
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
              if (ycon)>min_line and wcon>1280/4:
                 min_line=ycon 
                 minbase=hcon
                 dsp_array.append(hcon)
                 base_height=hcon
                 min_h_ary.append(hcon)
                 hcon_ary.append(hcon)
                 one= sum(min_h_ary)/len(min_h_ary)
                 minbase=int(sum(hcon_ary)/len(hcon_ary))
                 
                 #threshold_base.append(hcon)
              #else:
             #     dsp_array.append(sum(dsp_array)/len(dsp_array))

                 
              '''  
              ============================================================================================================
              ============================================================================================================
              ============================================================================================================
              ============================================================================================================
              '''    
              #if swell is true create tracker to average 
                 
              if minbase<one :
               swell_average.append(-1)
               
          
                    

                  #  print('below average')
              if minbase>one:  
                    swell_average.append(1)#2
                 
                    
                    
     
                   
                   
              if len(swell_average)>10:
                  swell_average.popleft()
            
              
             # print (sum(swell_average))#/len(swell_average))#/len(swell_average))
              if sum(swell_average)>0:
                  Break_abv=True
                  
              if sum(swell_average)<0:   
                  Break_below=True
                  
              if Break_below==False:
                 Break_abv=False
            

                
              if Break_abv==True and Break_below==True and max_base>=one:
                    
                    Break_abv=False
                    Break_below=False
                   # print('FIFO - wave detected')
                
                    break_=True
                    FiFo_detect=True
                
          
             
                 
              if len(hcon_ary)>=3:
                     hcon_ary.popleft()
                    # Wave_avg.popleft()
              #if len(min_h_ary)>=30:#implement standard deviation?
                     min_h_ary.popleft()
             
              if len(min_h_ary)>=150:
                    # Wave_avg.popleft()
              #if len(min_h_ary)>=30:#implement standard deviation?
                     min_h_ary.popleft()
                  
                     
             # 
             # old_local=local
      
              cv2.line(fifo_frame,(0,int(roi_base-one)),(1280,int(roi_base-one)), (0,0,255),1)    
        
                 

              try:
                  sum_base=sum(threshold_base)
                  no_base=len(threshold_base)
                  base_average=sum_base/no_base

              except:
                  base_average=20
                  
                  
                  
                  
              colour_min=255,0,0  
              colour_max=255,125,125     
              rolling_adverage_min =remov[min_line:min_line+minbase,0:1280] 
              rolling_adverage_max =remov[max_line:max_line+max_base,0:1280] 
              swell_area =convex_hull(rolling_adverage_max,max_line,max_base,colour_min, False)
              Break_area =convex_hull(rolling_adverage_min,min_line,minbase,colour_max, True) 
              
    
              swell_area=convex_hull(rolling_adverage_max,max_line,max_base,colour_min, False)   

              if swell_area<maxW and swell_area!=Break_area and max_base<30:# and max_line+max_base<roi_base-one:
                         #print('swell height: ',sum(swell_check))
                         swell_switch=True
                         break_switch=False
                         swell_check.append(1)
              else: 
                  swell_check.append(0)
                         
              if len(swell_check)>66:
                             swell_check.popleft()
              if sum(swell_check)>=15:
                             swell_check.clear()
                             swell_ther=True
              if sum(swell_check)<5:# and max_line>=min_line:
                             swell_brok=True
              if minbase<5 and min_line<100:
                  swell_brok=False
                             
              if  swell_ther==True :# swell_brok==True and
                  swellcount=swellcount+1
                  swell_ther=False
                  swell_brok=False
                  #print('swells:',swellcount)
                  Swell=True
                  break_=False
                  
         #     if   max_line+max_base >min_line and max_line+max_base <minbase+min_line:
       #            break_=False
                  
                  
              if Swell==False and break_==True:
                  break_=False
                  
              if Swell==True and break_==True and min_line>one and max_line>one and wave_swell==True:
                  print('clean_Wave',clean_Wave_count, 'detected')
                  clean_Wave_count=clean_Wave_count+1
                  Swell=False
                  break_=False
                  wave_swell=False
                  
              if swell_switch==True:
                     cv2.putText(r_frame, "Status: Swell",(50,int(height-height/3)-25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                     swell_switch=False

    #top contour
    cv2.line(fifo_frame,(0,base_cut+max_line),(1280,base_cut+max_line), (30,255,255),2)
    cv2.line(fifo_frame,(0,base_cut+max_line+max_base),(1280,base_cut+max_line+max_base), (150,255,255),2)
    
    #bottom contour
    cv2.line(fifo_frame,(0,base_cut+min_line),(1280,base_cut+min_line), (255,30,255),2)
    cv2.line(fifo_frame,(0,base_cut+min_line+minbase),(1280,base_cut+min_line+minbase), (255,30,255),2)

 
  #  cv2.imshow('fifo_frame',fifo_frame)
    
    average_base=0
    return abv_avg, blw_avg,occlusion_counter, max_thres,min_thres,average_base















def  centroid_analyses(r_frame):
    global Drift_reset
    F_centroid=empty_frame

    average= sum(Drift_tracker_X)/len(Drift_tracker_X)
    if len(drift_average)==0:
        drift_average.append(average)
    if Drift_reset==True:
        Drift_tracker_X.clear()
        Drift_tracker_X.append(average)
        drift_average.append(average)
        Drift_reset=False
    if len(drift_average)>100:
        drift_average.popleft()
        
    drift_avg= sum(drift_average)/len(drift_average)
    
    
        #cv2.line(F_centroid,(leftb,botb),(leftb,topb), (255,255,255),2)
        #cv2.line(F_centroid,(rightb,botb),(rightb,topb), (255,255,255),2)
    cv2.line(r_frame,(int(drift_avg),base_cut),(int(drift_avg),roi_base), (0,255,0),2)
    cv2.line(r_frame,(int(1280/2),base_cut),(int(1280/2),roi_base), (255,255,0),2)
    cv2.putText(r_frame," Screen centre",(int(1280/2),roi_base-5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,0), 1) 
    cv2.putText(r_frame," Mean shift",(int(drift_avg),base_cut+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
   

    
    #cv2.imshow('Deviation',F_centroid)






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




def Occlusion_counter():
    global Drift_reset
    global Break_abv2
    global Break_below2
    global base_cut2
    global wave_count2
    global one
    global min_line
    global blw_avg
    global base_height
    global wave_count
    global wave_swell
    
    '''    
    
def API_Weather():
    
    new_api= 'http://magicseaweed.com/api/a6e57b6c5788cde8560cac105c2e9344/forecast/?spot_id=50&units=eu'
    api_address= 'http://api.openweathermap.org/data/2.5/weather?appid=b2d41430bd5b5403291b78b6b1fe12e0&units=metric&q=bundoran'
    jsonDayDat=requests.get(api_address).json()
    jsonDat=requests.get(new_api).json()#import and convert to json
  #  print (jsonDat)
    time_stamps=jsonDat[0]['timestamp']
    break_min=jsonDat[0]['swell']['minBreakingHeight']
    break_max=jsonDat[0]['swell']['maxBreakingHeight']
    period_val=jsonDat[0]['swell']['components']['primary']['period']
    print('period',period_val)
    print('Wave_Max',break_max)
    print('Wave_Min',break_min)
    
    
    Sun_set=jsonDayDat['sys']['sunset']
    Sun_rise=jsonDayDat['sys']['sunrise']

    print('Sunrise',datetime.utcfromtimestamp(Sun_rise).strftime('%Y-%m-%d %H:%M:%S'))
    '''    
    
    
    
    '''
    ============================================================================================================
    ============================================================================================================
    ============================================================================================================
    '''

                   
    #print("average",sum(swell_average))
             # print (sum(swell_average))#/len(swell_average))#/len(swell_average))
    if sum(swell_average)>=3:
                  Break_abv2=True
                  
    if sum(swell_average)<=-6:   
                  Break_below2=True
                  
          #    if Break_abv==False:
          #       Break_below=False
                
    if Break_abv2==False and Break_below2==True:
                  Break_below2=False
               #]   Break_below==False
                
    if Break_abv2==True and Break_below2==True and base_height>=one: 
                    
                    Break_abv2=False
                    Break_below2=False
                    wave_count=wave_count+1
                    print('wave ',wave_count,' detected')
                   # wave_swell=True
                    Drift_reset=True
             












'''#######################################################################################
Loop starts here ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
==========================================================================================
'''




while True:
    

    #read initial frame
    ret, frame= cap.read()
    #timer+=1
   # ret, frame=image.read()
    #grey scale
    r_frame=frame
    empty_frame=r_frame
 #   try:
 #       base_cut=max(maxline_ary)-7
 #   except:
    try:
        base_cut=dyno_crop(base_cut)
        
        frame=frame[base_cut:roi_base,0:1280]
    except:
        frame=frame[99:roi_base,0:1280]
    try:
        cv2.imshow('frame oscillate',frame)
    except:
        1+1
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame orig',frame)
    
    
    K=2
    
    threshold,kmask,colors = K_mask(frame, K)
    
    cv2.imshow('kmask',kmask)
    
    #cv2.imshow('kmask',kmask)
    #NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    remov=kmask+frame
    swell_val,swell_cons=swell_filter (frame)
    cv2.imshow('remov_raw',remov)
    ret,remov=cv2.threshold(remov,100,255,cv2.THRESH_BINARY_INV)
   #remov=cv2.adaptiveThreshold(remov,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,33,3)
    
    
    
    
    
    
    
    
    
    
    cv2.imshow("remov filter",remov)
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
    
 
#    yt,BB_size,mixcout,threshold_y=tracker(yt,BB_size,mixcout,threshold_y) 




    centre_line= cv2.line(r_frame,(0,roi_base),(1280,roi_base),(0,0,255),2)

    '''
    ============================================================================
    Good up to here, following an attempt at MOG filtering for False positives
    ============================================================================
    '''
    ''' CREATE IVERTED VERSION!!! ie. base up'''
        #canny------------> contour 
        #if contour width>height or width< screenwidth/x  : remove from filter
    con_mask = np.ones(first_frame.shape[:2], dtype="uint8") * 255  
  
    
    abv_avg, blw_avg,occlusion_counter,max_thres,min_thres,average=FiFo_brackets(remov,blw_avg,occlusion_counter,abv_avg, max_thres,min_thres)# call to add bottom and top trackers
    Occlusion_counter()
    #cv2.line(r_frame,(0,max_line2),(width,max_line2), (0,0,255),2)
    #cv2.line(r_frame,(0,max_line2+max_base2),(width,max_line2+max_base2), (0,0,255),2)

   # cv2.imshow('contour filter',con_mask)


    cv2.imshow('frame',frame)
    cv2.imshow('R_frame',r_frame)    
    
    
    '''testing ROI_base'''
    centroid_analyses(r_frame)
    '''testing ROI_base complete'''
    
   # if len(dsp_array) % 1800:
    plotter()
    
    key=cv2.waitKey(1)
    if key ==27:
         break
     

     
cap.release()
cv2.destroyAllWindows()

    


    
    
    