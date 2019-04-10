import cv2
import numpy as np
import streamlink
import datetime
import os



#MP4file='E:\\masters_Video_cap\\6_1_2018_0.mp4'#data collection file
MP4file='E:\\Utube_stock\\My Video8.MP4'
#'E:\\Utube_stock\\My Video7.MP4'
#'E:\\Utube_stock\\My Video6.MP4'
#'E:\\Utube_stock\\My Video5.MP4'
#'E:\\Utube_stock\\My Video4.MP4'
#'E:\\Utube_stock\\My Video3.MP4'
#'E:\\Utube_stock\\My Video2.MP4'

#'E:\\Utube_stock\\My Video1.MP4'#'E:\\masters_Video_cap\\Pipeline.mp4'#E:\\masters_Video_cap\\6_1_2018_3.mp4'
#6_1_2018_2
#6_1_2018_1
#6_1_2018_4
#8_1_2019_0
#8_1_2019_4
#8_1_2019_4
#7_1_2019_1
#28_12_2018_7
#28_12_2018_3
#6_1_2018_0q
#6_1_2018_2
timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
print(timeVal)
#unique value for each saved frame
#cv2.imwrite('whirldata.png',img)


#cap = cv2.VideoCapture('E:\\masters_Video_cap\\28_12_2018_2.mp4')

cap = cv2.VideoCapture(MP4file)
#Reading the first frame
(grabbed, frame) = cap.read()



#for streaming real time
#streams = streamlink.streams("http://46.4.36.73/hls/bundoransurfco/playlist.m3u8")#"https://youtu.be/bNLy-XXYxcw")
#quality='best'
image=cap #cv2.VideoCapture(MP4file)#
cv2.namedWindow("Data Collection Program")   
system_status= 'select mode'
# image= image1[100,20,400,100]





while True:
    #image=image1(426,240,720,1280)
   
    width = image.get(cv2.CAP_PROP_FRAME_WIDTH)
    height =image.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #print(width,height)
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    key=cv2.waitKey(33)#30 fps
    
    
    if key==ord('q'):
        timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        ROI_size=0
        print('q pressed\n')
        filePath='E:/hawaii_Images/'
        filename=timeVal+'_image.jpg'
        name_fpath=os.path.join(filePath,filename)
        _, frame=image.read()
        cv2.imshow("Data Collection Program",frame)
        _, frame=image.read()
        system_status= 'Surfer waiting mode'
        cv2.rectangle(frame,(226,480),(1080,0),(0,255,0),2)
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
        #cv2.waitKey(0)
        print("filePath; ",name_fpath)
        try:
            print("attempt here!")
            (grabbed, frame) = cap.read()

            cv2.namedWindow('Data Collection Program')
            
      

    #drawing rectangle
           
               
    
            print("Clone created")
                #check if correct. 

            photo=cv2.imwrite(name_fpath,frame)#ROI)
            cv2.imshow('1',frame)
                #end of problem area
            print('Saved')
                #cv2.destroyWindow("roi highlight")
            cv2.destroyWindow('1')
            key!=ord('q')
         
        except:
            print("nope! try again")
            cv2.waitKey(500)
        
        
    if key==ord('w'):
        cv2.waitKey(15)
        _, frame=image.read()
        cv2.rectangle(frame,(226,480),(1080,0),(0,255,0),2)#(426,480),(1280,0),(0,255,0),2)
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
    
    if key ==27:#escape key
        print('escape\n')
        image.release()
        cv2.destroyAllWindows()
        break
    
  #  elif 
    #image1=image 
    _, frame=image.read()
    cv2.rectangle(frame,(226,480),(1080,0),(0,255,0),2)#(426,480),(1280,0),(0,255,0),2)
    SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Data Collection Program",frame)
       