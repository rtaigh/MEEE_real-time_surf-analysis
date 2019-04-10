
import cv2
import numpy as np
import datetime
import time
#import csv


roi_base=180
Former_BB=0 # occlusion prevention

base_cut=93
startyt=120-base_cut
ht_fix=(25/180)*(roi_base-base_cut)#roi_base
ht=ht_fix
yt= startyt  
ycan=startyt 
hcan=roi_base
threshold_y=yt
threshold_ycan=ycan
threshold_ymog=yt
bb_org=int(ht_fix)
BB_size=bb_org
period_Per_min=0
period_average=0
waves_all_day=0
mint_cnt=datetime.time.minute
lastMarker=time.time()
mixcout=0
FPS=30
swell_switch= False
break_switch= False
swell_average= []
Drift_tracker_X=[]
Drift_tracker_Y=[]
maxline_ary=[]

threshold_base=[]
threshold_max=[]
threshold_min=[]
max_thres =60
min_thres=30
global blw_avg
blw_avg=False
abv_avg=False
occlusion_counter=0

###search for contours in ROI the track
##set contour at top of page as start point/ roi
##mog for flase positives?
##filter contour by size!!!! for surfers
#try just canny
#canny with kmask at bottom of ROI?
'''Bottom value is much smoother, may reduce false positives of top threshold but add validity'''
''' Check frame speed'''

cap = cv2.VideoCapture('masters_Video_cap/19_1_2019_1.mp4')#28_12_2018_8.mp4')#29_12_2018_0.mp4')#'6_1_2018_4.mp4')#19_1_2019_1.mp4')#'My Video1.mp4')#     ITERATION THEN SHIFT AMOUNT
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,1,1) #high value means more FP #changed iterations from 10 to 2 
_,first_frame = cap.read()
height, width = first_frame.shape[:2]

fgbg = cv2.createBackgroundSubtractorMOG2()

kernel_erode = np.ones((10,8),np.uint8)
kernel_dilate= np.ones((1,32),np.uint8)
#kernel_erode_canny = np.ones((1,32),np.uint8)


    
        
    

def convex_hull (frame,yt,ht,colour,centroid):#, #y_val):
    x,y=frame.shape[:2]
    contours,_=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    #cv2.drawContours(r_frame,hull,-1,(255,0,255)) 
     
    contours,heirarchy=cv2.findContours(frame,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
    contours,_= cv2.findContours(remov,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#SIMPLE
    blw_avg=blw_avg
    '''
    edited
    '''
    
    max_line=height
    min_line=height
          #max_con=contours[0]
          #loop through all contours
    max_base=0
    maxX=0
    maxW=0
    minbase=1
    min_line=1
    minX=0
    minW=0
    
    
 
    for contour in contours:
            #get bounding boxes
          xcon,ycon,wcon,hcon=cv2.boundingRect(contour)
          #xcon,ycon,wcon,hcon=0,0,0,0
          
          
          
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
                  #old value for swell transformation
                 Swell_threshold=max_base+max_line
                 past_max=max_line
                 max_line=ycon
                 
                 maxline_ary.append(ycon)
                 
                 max_base=hcon
                
                 maxX=xcon
                 maxW=wcon
                 max_con=contour
                 cv2.drawContours(con_mask, contour, -1, 0, -1) 
                 swell_average.append(hcon)
                 hcon_average=sum(swell_average)/len(swell_average)
                
               #  print('max_line',max_line)
                 
                 

                 
              
              if (ycon)>min_line and wcon>width/2:
                 min_line=ycon 
                 #if min_line<=base_cut+1:
                  #  min_line=roi_base 
                 minbase=hcon
                 threshold_base.append(hcon)
                
              #   print('avg',avg)
               #  print('hcon',hcon)
                 minX=xcon
                 minW=wcon
                 threshold_base.append(ycon)
                 sum_base=sum(threshold_base)
                 no_base=len(threshold_base)
                 base_average=sum_base/no_base
                # print('base average',base_average)
                 cv2.line(r_frame,(0,int(base_cut+base_average)),(1280,int(base_cut+base_average)), (255,255,255),2)
                 
              swell_switch=False
              break_switch=False                  
                 
              colour_min=255,0,0  
              colour_max=255,125,125              
              rolling_adverage_min=remov[min_line:min_line+minbase,0:width] 
              rolling_adverage_max=remov[max_line:max_line+max_base,0:width] 
              swell_area=convex_hull(rolling_adverage_max,max_line,max_base,colour_min, False)
              Break_area=convex_hull(rolling_adverage_min,min_line,minbase,colour_max, True) 
              
             # print('swell area',swell_area)    

                 
                 
              if swell_area<maxW/2 or max_base<10 :
                         swell_switch=True
                         break_switch=False 

                         
            
              elif max_line>=min_line and min_line<20 or min_line<max_line+max_base or Break_area==swell_area :
                     swell_average.clear()
                    # print('swell height',hcon_average)
                     swell_switch=False
                     break_switch=True
                     
              if (max_line+max_base)>min_line:
                  swell_switch=False
                     
              if swell_switch==True:
                     cv2.putText(r_frame, "Status: Swell",(50,int(height-height/3)-25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                 
              if break_switch==True:
                     swell_switch==False
                     cv2.putText(r_frame, "Status: Breaking wave", (50,int(height-height/3)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                 
                 
                    
        
                    
    '''            
    colour_min=255,0,0  
    colour_max=125,0,0              
    rolling_adverage_min=remov[min_line:min_line+minbase,0:width] 
    rolling_adverage_max=remov[max_line:max_line+max_base,0:width] 
    convex_hull(rolling_adverage_max,max_line,max_base,colour_min)
    convex_hull(rolling_adverage_min,min_line,minbase,colour_max)
    ''' 
    #top contour
  #  cv2.line(r_frame,(0,base_cut+max_line),(width,base_cut+max_line), (30,255,255),2)
  #  cv2.line(r_frame,(0,base_cut+max_line+max_base),(width,base_cut+max_line+max_base), (150,255,255),2)
    
    #bottom contour
    cv2.line(r_frame,(0,base_cut+min_line),(width,base_cut+min_line), (255,30,255),2)
    cv2.line(r_frame,(0,base_cut+min_line+minbase),(width,base_cut+min_line+minbase), (255,30,255),2)

    #blw_avg=False
    #abv_avg=False
    #minbase
    #threshold_base=[]
    
    average_base=sum(threshold_base)/len(threshold_base)
    cv2.line(r_frame,(0,int(base_cut+average_base)),(1280,int(base_cut+average_base)), (255,255,255),2)
   # print('base',average_base)
  #  if abv_avg==True and blw_avg==False:
   #     abv_avg==False#

    if minbase<max_thres:
       # print('blw_avg=True')
        blw_avg=True
        print(minbase , min_thres,max_thres )
    if minbase>=min_thres:
        abv_avg=True
       # print('abv_avg=True')
        
       
    if blw_avg==True and abv_avg==False:
       blw_avg==False
        
        
    if blw_avg==True and abv_avg==True:
        base_av=average_base
        threshold_max.append(max(threshold_base))
        min_thres =base_av+int((max(threshold_max)-base_av)/5)
        threshold_min.append(min(threshold_base))
        max_thres =int(min(threshold_min)+((average_base-min(threshold_min))/2))
        threshold_base.clear()
        threshold_base.append(base_av)
        blw_avg=False
        abv_avg=False
        occlusion_counter=occlusion_counter+1
        cv2.line(r_frame,(0,int(base_cut+min_thres)),(1280,int(base_cut+min_thres)), (255,0,0),2)
        
        print('Occluded waves',occlusion_counter)
    #print('Min',min_thres)
    #print('Max',max_thres)
        
    cv2.line(r_frame,(0,int(base_cut+max_thres)),(1280,base_cut+int(max_thres)), (255,0,0),2)
    
   # cv2.putText(r_frame,"below"+str(blw_avg),(50,int(base_cut+min_thres-2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
   # cv2.putText(r_frame,"above"+str(abv_avg),(150,int(base_cut+min_thres-2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
   # cv2.putText(r_frame, "min threshold",(250,int(base_cut+min_thres-2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
   # cv2.line(r_frame,(0,int(base_cut+min_thres)),(1280,int(base_cut+min_thres)), (255,0,0),2)
   # cv2.putText(r_frame,"max threshold",(50,base_cut+int(max_thres)-2),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return abv_avg, blw_avg,occlusion_counter, max_thres,min_thres







def  centroid_analyses(r_frame):
    
    F_centroid= r_frame
#        cv2.line(r_frame,(minX,min_line),(minX+minW,min_line), (255,30,255),2)
       # cv2.circle(r_frame, (cX, cY), 5, (0, 0, 255), -1)
        #Drift_tracker_X=[]
       # Drift_tracker_Y=[]
    '''Only adds new points'''
    try:  
        leftb=min(Drift_tracker_X)
        rightb=max(Drift_tracker_X)
        
        topb=min(Drift_tracker_Y)
        botb=max(Drift_tracker_Y)
        
        cv2.line(F_centroid,(leftb,botb),(leftb,topb), (255,255,255),2)
        cv2.line(F_centroid,(rightb,botb),(rightb,topb), (255,255,255),2)
        for points in Drift_tracker_Y:
         if Drift_tracker_Y.index(points) %100:
             
             cv2.circle(F_centroid, (Drift_tracker_X[Drift_tracker_Y.index(points)], points), 3, (0,0,255), -1) 
             #print('co_ordinates',Drift_tracker_Y[Drift_tracker_Y.index(points)], points)#'Locations of moments',Drift_tracker_X(Drift_tracker_Y.index(points)),
        cv2.imshow('centroids',F_centroid)
     

    except:
         return

























'''#######################################################################################
Loop starts here ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
==========================================================================================
'''




while True:
    

    #read initial frame
    ret, frame= cap.read()
    #grey scale
    r_frame=frame
 #   try:
 #       base_cut=max(maxline_ary)-7
 #   except:
    frame=frame[base_cut:roi_base,0:1280]
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
    
 
    _, track_window_break_1 = cv2.meanShift(remov,(0,yt,1280,BB_size),term_criteria)

    xt,yt,wt,ht=track_window_break_1
   
    

    
    
    
   # print(xt,yt,wt,ht)
    #reset value
    #set tracker at mean Point of the wave!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if yt+ht+base_cut>=roi_base-10:
        remove_c=int((25/180)*5)+1
       
        BB_size=BB_size-remove_c
        #print('BB_size:',BB_size)
        yt=yt+remove_c
        
        if BB_size<=remove_c+3:
            #print('BB_size',BB_size)
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
   
    '''uni-directional movement'''
    if threshold_y>yt:
         yt=threshold_y
    threshold_y=yt
#added
  # print(yt,yt+ht)                  #yt+ht
  #  try:
  #     cv2.rectangle(r_frame, (0,base_cut+yt),( 1280,base_cut+yt+ht), (0,255,0), 2)
    #base_cut+base_cut+
  #  except:
  #     cv2.rectangle(r_frame, (0,base_cut),( 1280,base_cut+yt), (0,255,0), 2)
    
    
    '''
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    APEX is moment of convex hull, side movement can be derived by tracking moment across screen
    '''
    
    remov_tracker=remov[yt:roi_base,xt:xt+wt]  
 #   _, track_window_break_2 = cv2.meanShift(remov_tracker,(0,0,wt,BB_size),term_criteria)
    #xt2,yt2,wt2,ht2=track_window_break_2
    #print(xt2,yt2,wt2,ht2)
    #cv2.imshow('remov_tracker',remov_tracker)
    #convex_hull(remov_tracker,yt)
   # cv2.rectangle(r_frame, (wt2,base_cut+yt+yt2),( wt2+ht2,base_cut+yt+yt2+ht2), (255,0,0), 2)
    



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
  
    
    #cv2.line(r_frame,(0,max_line2),(width,max_line2), (0,0,255),2)
    #cv2.line(r_frame,(0,max_line2+max_base2),(width,max_line2+max_base2), (0,0,255),2)

    cv2.imshow('contour filter',con_mask)


    
    abv_avg, blw_avg,occlusion_counter,max_thres,min_thres=FiFo_brackets(remov,blw_avg,occlusion_counter,abv_avg, max_thres,min_thres)# call to add bottom and top trackers
    

    cv2.imshow('frame',frame)
    cv2.imshow('R_frame',r_frame)    
    
    
    '''testing ROI_base'''
    #centroid_analyses(r_frame)
    '''testing ROI_base complete'''
    
    
    
    key=cv2.waitKey(20)
    if key ==27:
         break
     
     
cap.release()
cv2.destroyAllWindows()

    


    
    
    