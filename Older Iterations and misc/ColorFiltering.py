qimport cv2
import numpy as np

cap = cv2.VideoCapture('ShibuyaCrossingFullHD.mp4')

while True:
    _, frame = cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)#hugh saturation value
    
    lower_Val_Jskin=np.array([0, 30, 60,])
    upper_Val_Jskin= np.array([20, 150, 255])
    
    lower_Val_Road=np.array([150, 20, 140,])
    upper_Val_Road=np.array([155, 30, 150,])
    
    lower_AlVal=np.array([0, 0, 0,])
    upper_AlVal= np.array([255, 255, 255])
    
    lower_Val_white=np.array([255, 255, 255])
    upper_Val_white= np.array([255, 255, 255])   
    lower_Val_blk=np.array([0, 0, 0])
    upper_Val_blk= np.array([0, 0, 0])  
    
    
   
    mask_Road=cv2.inRange(hsv,lower_Val_Road,upper_Val_Road)
    All_Val=cv2.inRange(hsv, lower_AlVal,upper_AlVal)
    mask_BLK=cv2.inRange(hsv,lower_Val_blk,upper_Val_blk)
    mask_Wht=cv2.inRange(hsv,lower_Val_white,upper_Val_white)
    maskSkin= cv2.inRange(hsv,lower_Val_Jskin,upper_Val_Jskin)
    mask=mask_BLK+mask_Wht+maskSkin-mask_Road
    
  #  res=cv2.bitwise_and(frame,frame, mask_W = mask_W)#,mask_BLK=mask_BLK)
    res=cv2.bitwise_and(frame,frame, mask= mask)#,mask_BLK = mask_BLK)#,mask_Wht=mask_Wht )
    
 #not great- average   
 #   kernal= np.ones((2,2),np.float32)/4
 #   smoothed=cv2.filter2D(res,-1,kernal)
    
    
    #gaussian blur?- crap- add pixels around heads
    
    blur=cv2.GaussianBlur(res,(5,5),0)
    #still bad
    median=cv2.medianBlur(res,5)
    #Edge detectors----------------------------------------------------------
    edgeD=cv2.Canny(frame,100,100);
    
    
    #MOG____________________________Change detection
    fgbg=cv2.createBackgroundSubtractorMOG2()
    fgbgMask=fgbg.apply(frame)
    
    
    cv2.imshow('frame',frame)
  #  cv2.imshow('mask',mask)
    cv2.imshow('res',res)
     
    cv2.imshow('smoothed',fgbgMask)
  #  cv2.imshow('res2',res2)
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
  #crop to view
        cv2.destroyAllWindows()
        cap.release()