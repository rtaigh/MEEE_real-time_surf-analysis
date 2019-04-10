import cv2
import numpy as np
import streamlink
import datetime
import os



#MP4file='E:\\masters_Video_cap\\6_1_2018_0.mp4'#data collection file
MP4file='E:\\Utube_stock\\My Video1.mp4'
    #\\masters_Video_cap\\.mp4'
#6_1_2018_2
#6_1_2018_0
#5_1_2018_5
#5_1_2018_0
#31_12_2018_4
#30_12_2018_5
#30_12_2018_2
#29_12_2018_3
#29_12_2018_7
#29_12_2018_0
#28_12_2018_5
#28_12_2018_4
#28_12_2018_3
#28_12_2018_2
#6_1_2018_0.mp4
#28_12_2018.mp4'
#'E:\\masters_Video_cap\\6_1_2018_0.mp4'#complete
timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
print(timeVal)
#unique value for each saved frame
#cv2.imwrite('whirldata.png',img)

rect = (0,0,0,0)
startPoint = False
endPoint = False

def on_mouse(event,x,y,flags,params):

    global rect,startPoint,endPoint

    # get mouse click
    if event == cv2.EVENT_LBUTTONDOWN:

        if startPoint == True and endPoint == True:
            startPoint = False
            endPoint = False
            rect = (0, 0, 0, 0)

        if startPoint == False:
            rect = (x, y, 0, 0)
            startPoint = True
        elif endPoint == False:
            rect = (rect[0], rect[1], x, y)
            endPoint = True

#cap = cv2.VideoCapture('E:\\masters_Video_cap\\28_12_2018_2.mp4')

cap = cv2.VideoCapture(MP4file)
#Reading the first frame
(grabbed, frame) = cap.read()



#for streaming real time
#streams = streamlink.streams("http://46.4.36.73/hls/bundoransurfco/playlist.m3u8")#"https://youtu.be/bNLy-XXYxcw")
#quality='best'
image=cap #cv2.VideoCapture(MP4file)#
cv2.namedWindow("Data Collection Program")
cv2.setMouseCallback('Data Collection Program', on_mouse)    
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
        filePath='E:/Utube_stock/surferLib_n_Water/'
        filename=timeVal+'_surfer_water.jpg'
        name_fpath=os.path.join(filePath,filename)
        _, frame=image.read()
        cv2.imshow("Data Collection Program",frame)
        _, frame=image.read()
        system_status= 'Surfer waiting mode'
        cv2.rectangle(frame,(226,480),(1080,0),(0,255,0),2)
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
        cv2.waitKey(0)
        print("filePath; ",name_fpath)
        try:
            print("attempt here!")
            (grabbed, frame) = cap.read()

            cv2.namedWindow('Data Collection Program')
            
            cv2.setMouseCallback('Data Collection Program', on_mouse)    

    #drawing rectangle
            if startPoint == True and endPoint == True:
                startPoint == False and endPoint == False #reset
                #new loop for multiple selects?
                #ROI=[]
                #ROI.append(frame[rect[0]:rect[2],rect[1]:rect[3]])
                x=rect[0]
                y=rect[1]
                hight=rect[2]
                width=rect[3]
                
                print("Point1: ",x)
                print("Point2: ",y)
                print("Point3: ",hight)
                print("Point4: ",width)
                
               
                
                copy=frame[y:width,x:hight].copy()
                print("Clone created")
                #check if correct. 
                roif=cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
                point1=cv2.putText(frame,"point 1"+system_status,(rect[0],rect[1]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                point2=cv2.putText(frame,"point 2"+system_status,(rect[2],rect[3]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow("roi highlight",frame)
                 
                print('Crop Attempt')
                #Crop_ROI=copy(x,y, width,hight)
               
                print('Crop successful')
               # photo=cv2.imwrite(name_fpath,0)
                photo=cv2.imwrite(name_fpath,copy)#ROI)
                cv2.imshow('1',copy)
                #end of problem area
                print('Saved')
                #cv2.destroyWindow("roi highlight")
                cv2.destroyWindow('1')
                rect = (0, 0, 0, 0)
                key!=ord('q')
         
        except:
            print("nope! try again")
            cv2.waitKey(500)
        
        
    #elif cv2.waitKey(0)==ord('w'):
    if key==ord('w'):
   
        print('w pressed\n')
        timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        filePath='E:/Utube_stock/surferLib_Riding/'
        filename= timeVal+'_surfer_riding.jpg'
        name_fpath=os.path.join(filePath,filename)
        cv2.imshow("Data Collection Program",frame)
        _, frame=image.read()
        system_status= 'Surfer on wave mode'
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
        cv2.waitKey(0)
        try:
            print("attempt here!")
            (grabbed, frame) = cap.read()

            cv2.namedWindow('Data Collection Program')
            
            cv2.setMouseCallback('Data Collection Program', on_mouse)    

    #drawing rectangle
            if startPoint == True and endPoint == True:
                startPoint == False and endPoint == False #reset
                #new loop for multiple selects?
                #ROI=[]
                #ROI.append(frame[rect[0]:rect[2],rect[1]:rect[3]])
                x=rect[0]
                y=rect[1]
                hight=rect[2]
                width=rect[3]
                
                print("Point1: ",x)
                print("Point2: ",y)
                print("Point3: ",hight)
                print("Point4: ",width)
                
               
                
                copy=frame[y:width,x:hight].copy()
                print("Clone created")
                #check if correct. 
                roif=cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
                point1=cv2.putText(frame,"point 1"+system_status,(rect[0],rect[1]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                point2=cv2.putText(frame,"point 2"+system_status,(rect[2],rect[3]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow("roi highlight",frame)
                 
                print('Crop Attempt')
                #Crop_ROI=copy(x,y, width,hight)
               
                print('Crop successful')
               # photo=cv2.imwrite(name_fpath,0)
                photo=cv2.imwrite(name_fpath,copy)#ROI)
                cv2.imshow('1',copy)
                #end of problem area
                print('Saved')
                #cv2.destroyWindow("roi highlight")
                cv2.destroyWindow('1')
                rect = (0, 0, 0, 0)
                key!=ord('w')
         
        except:
            print("nope! try again")
            cv2.waitKey(500)
        
    #elif cv2.waitKey(0)==ord('e'):
    if key==ord('e'):
        timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        print('e pressed\n')
        filePath='E:/Utube_stock/Swell/'
        filename= timeVal+'_Swell.jpg'
        name_fpath=os.path.join(filePath,filename)
        cv2.imshow("Data Collection Program",frame)
        _, frame=image.read()
        system_status= 'Wave Swell mode'
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
        cv2.waitKey(0)
        try:
            print("attempt here!")
            (grabbed, frame) = cap.read()

            cv2.namedWindow('Data Collection Program')
            
            cv2.setMouseCallback('Data Collection Program', on_mouse)    

    #drawing rectangle
            if startPoint == True and endPoint == True:
                startPoint == False and endPoint == False #reset
                #new loop for multiple selects?
                #ROI=[]
                #ROI.append(frame[rect[0]:rect[2],rect[1]:rect[3]])
                x=rect[0]
                y=rect[1]
                hight=rect[2]
                width=rect[3]
                
                print("Point1: ",x)
                print("Point2: ",y)
                print("Point3: ",hight)
                print("Point4: ",width)
                
               
                
                copy=frame[y:width,x:hight].copy()
                print("Clone created")
                #check if correct. 
                roif=cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
                point1=cv2.putText(frame,"point 1"+system_status,(rect[0],rect[1]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                point2=cv2.putText(frame,"point 2"+system_status,(rect[2],rect[3]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow("roi highlight",frame)
                 
                print('Crop Attempt')
                #Crop_ROI=copy(x,y, width,hight)
               
                print('Crop successful')
               # photo=cv2.imwrite(name_fpath,0)
                photo=cv2.imwrite(name_fpath,copy)#ROI)
                cv2.imshow('1',copy)
                #end of problem area
                print('Saved')
                #cv2.destroyWindow("roi highlight")
                cv2.destroyWindow('1')
                rect = (0, 0, 0, 0)
                key!=ord('e')
         
        except:
            print("nope! try again")
            cv2.waitKey(500)
        
    #elif cv2.waitKey(0)==ord('r'): 
    if key==ord('r'):
        timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        print('r pressed\n')
        filePath='E:/Utube_stock/PointBrake/'
        filename= timeVal+'_brake.jpg'
        name_fpath=os.path.join(filePath,filename)
        cv2.imshow("Data Collection Program",frame)
        _, frame=image.read()
        system_status= 'Wave break mode'
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
        cv2.waitKey(0)
        try:
            print("attempt here!")
            (grabbed, frame) = cap.read()

            cv2.namedWindow('Data Collection Program')
            
            cv2.setMouseCallback('Data Collection Program', on_mouse)    

    #drawing rectangle
            if startPoint == True and endPoint == True:
                startPoint == False and endPoint == False #reset
                #new loop for multiple selects?
                #ROI=[]
                #ROI.append(frame[rect[0]:rect[2],rect[1]:rect[3]])
                x=rect[0]
                y=rect[1]
                hight=rect[2]
                width=rect[3]
                
                print("Point1: ",x)
                print("Point2: ",y)
                print("Point3: ",hight)
                print("Point4: ",width)
                
               
                
                copy=frame[y:width,x:hight].copy()
                print("Clone created")
                #check if correct. 
                roif=cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
                point1=cv2.putText(frame,"point 1"+system_status,(rect[0],rect[1]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                point2=cv2.putText(frame,"point 2"+system_status,(rect[2],rect[3]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow("roi highlight",frame)
                 
                print('Crop Attempt')
                #Crop_ROI=copy(x,y, width,hight)
               
                print('Crop successful')
               # photo=cv2.imwrite(name_fpath,0)
                photo=cv2.imwrite(name_fpath,copy)#ROI)
                cv2.imshow('1',copy)
                #end of problem area
                print('Saved')
                #cv2.destroyWindow("roi highlight")
                cv2.destroyWindow('1')
                rect = (0, 0, 0, 0)
                key!=ord('r')
         
        except:
            print("nope! try again")
            cv2.waitKey(500)
        
    #elif cv2.waitKey(0)==ord('t'):
    if key==ord('t'):
        timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        print('t pressed\n')
        filePath='E:/Utube_stock/Wash/'
        filename=timeVal+'_surfer.jpg'
        name_fpath=os.path.join(filePath,filename)
        cv2.imshow("Data Collection Program",frame)
        _, frame=image.read()
        system_status= 'Sea bird mode'
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
        cv2.waitKey(0)
        try:
            print("attempt here!")
            (grabbed, frame) = cap.read()

            cv2.namedWindow('Data Collection Program')
            
            cv2.setMouseCallback('Data Collection Program', on_mouse)    

    #drawing rectangle
            if startPoint == True and endPoint == True:
                startPoint == False and endPoint == False #reset
                #new loop for multiple selects?
                #ROI=[]
                #ROI.append(frame[rect[0]:rect[2],rect[1]:rect[3]])
                x=rect[0]
                y=rect[1]
                hight=rect[2]
                width=rect[3]
                
                print("Point1: ",x)
                print("Point2: ",y)
                print("Point3: ",hight)
                print("Point4: ",width)
                
               
                
                copy=frame[y:width,x:hight].copy()
                print("Clone created")
                #check if correct. 
                roif=cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
                point1=cv2.putText(frame,"point 1"+system_status,(rect[0],rect[1]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                point2=cv2.putText(frame,"point 2"+system_status,(rect[2],rect[3]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                cv2.imshow("roi highlight",frame)
                 
                print('Crop Attempt')
                #Crop_ROI=copy(x,y, width,hight)
               
                print('Crop successful')
               # photo=cv2.imwrite(name_fpath,0)
                photo=cv2.imwrite(name_fpath,copy)#ROI)
                cv2.imshow('1',copy)
                #end of problem area
                print('Saved')
                #cv2.destroyWindow("roi highlight")
                cv2.destroyWindow('1')
                rect = (0, 0, 0, 0)
                key!=ord('t')
         
        except:
            print("nope! try again")
            cv2.waitKey(500)
        
    #elif cv2.waitKey(0)==ord('y'):
    if key==ord('y'): 
        timeVal= datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        print('y pressed\n')
        filePath='E:/Utube_stock/birdsOfAFeather/'
        filename=timeVal+'_surfer.jpg'
        name_fpath=os.path.join(filePath,filename)
        cv2.imshow("Data Collection Program",frame)
        _, frame=image.read()
        system_status= 'Bird Capture mode'
        SystemStatus_text=cv2.putText(frame,"System status: "+system_status,(10,80), font, 0.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow("Data Collection Program",frame)
        cv2.waitKey(0)
        try:
            print("attempt here!")
            (grabbed, frame) = cap.read()

            cv2.namedWindow('Data Collection Program')
            
            cv2.setMouseCallback('Data Collection Program', on_mouse)    

    #drawing rectangle
            if startPoint == True and endPoint == True:
                startPoint == False and endPoint == False #reset
                #new loop for multiple selects?
                #ROI=[]
                #ROI.append(frame[rect[0]:rect[2],rect[1]:rect[3]])
                x=rect[0]
                y=rect[1]
                hight=rect[2]
                width=rect[3]
                
                print("Point1: ",x)
                print("Point2: ",y)
                print("Point3: ",hight)
                print("Point4: ",width)
                
               
                
                copy=frame[y:width,x:hight].copy()
                print("Clone created")
                #check if correct. 
                roif=cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
                point1=cv2.putText(frame,"point 1"+system_status,(rect[0],rect[1]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                point2=cv2.putText(frame,"point 2"+system_status,(rect[2],rect[3]), font, 0.5,(0,255,0),2,cv2.LINE_AA)
                
                 
                print('Crop Attempt')
                #Crop_ROI=copy(x,y, width,hight)
               
                print('Crop successful')
               # photo=cv2.imwrite(name_fpath,0)
             
                photo=cv2.imwrite(name_fpath,copy)#ROI)
                cv2.imshow("roi highlight",frame)
            
                
                cv2.imshow('1',copy)
                #end of problem area
                print('Saved')
                #cv2.destroyWindow("roi highlight")
                cv2.destroyWindow('1')
                rect = (0, 0, 0, 0)
      
        except:
            print("nope! try again")
            cv2.waitKey(500)


    
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
       