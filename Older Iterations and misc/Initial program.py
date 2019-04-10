import numpy as np
import cv2
#import matplotlib.pyplot as plt
face_cascade=cv2.CascadeClassifier('cascades/data/harrcascade_frontal_face_alt2.xml')

cap=cv2.VideoCapture(0)

def make_1080p():
       cap.set(3,920)
       cap.set(4,1080)
       
def make_480p():
       cap.set(3,640)
       cap.set(4,480)
      
def change_res(width,height):
       cap.set(3,width)
       cap.set(4,height)
        
def rescale_frame(frame,percent=75):
    scale_percent =75
    width=int(frame.shape[1]*scale_percent/100)
    height=int(frame.shape[0]*scale_percent/100)
    dim=(width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
# resolution
make_480p()

while (True):
    

   # Typical image
    ret, frame = cap.read()
    frame=rescale_frame(frame,percent=10)
    cv2.imshow('frame',frame)
   
    
   # Grey scale
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     #rescale
    grey=rescale_frame(grey,percent=750) 
    cv2.imshow('frame with greyScale',grey)
    faces=face_cascade.detectMultiScale(grey,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
         print (x,y,w,h)
     
    #mask
 ##   img=cv2.imread(0,cv2.IMREAD_REDUCED_GRAYSCALE_8)
   # plt.imshow(img,cmap='plot',interpolation='bicubic')
    #plt.plot([50,100],[80,100],'c',linewidth=5)
     
    #plt.show()
    #print(cv2.__file__)#find file locations
     
     
    if cv2.waitKey(20) & 0xFF == ord('q'):
     cap.release()
     cv2.destroyAllWindows()
     break