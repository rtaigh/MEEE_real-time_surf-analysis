##Contador de personas
##Federico Mejia
import numpy as np
import cv2
import time

#Will need altering!!!!!!!!
import Person #change to wave
#wave counter
Wave_counter= 0        #counts people down

#wave classifier
Ubody_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')#new___________________________________
#change footage to live stream
cap = cv2.VideoCapture('peopleCounter.avi')#file location #'ShibuyaCrossingFullHD.mp4')#

#Propiedades del video
##cap.set(3,160) #Width   
##cap.set(4,120) #Height
#Imprime las propiedades de captura a consola 
#Print the capture properties to console
for i in range(19):
    print (i, cap.get(i))

w = cap.get(3)#open cv capture width
h = cap.get(4)#open cv capture width
frame_Area = h*w
areaTH = frame_Area/250
print ('Area Threshold', areaTH)

#defining threshold lines
line_up = int(2*(h/5)) 
line_down   = int(3*(h/5))
up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))

line_0=int(1*(h/11))
line_1=int(2*(h/11))
line_2=int(3*(h/11))
line_3=int(4*(h/11))
line_4=int(5*(h/11))
line_5=int(6*(h/11))
line_6=int(7*(h/11))
line_7=int(8*(h/11))
line_8=int(9*(h/11))
line_9=int(10*(h/11))
line_10=int(11*(h/11))




ptl00=[0, line_0];
ptl01=[w, line_0];
ptl10=[0, line_1];
ptl11=[w, line_1];
ptl20=[0, line_2];
ptl21=[w, line_2];
ptl30=[0, line_3];
ptl31=[w, line_3];
ptl40=[0, line_4];
ptl41=[w, line_4];
ptl50=[0, line_5];
ptl51=[w, line_5];
ptl60=[0, line_6];
ptl61=[w, line_6];
ptl70=[0, line_7];
ptl71=[w, line_7];
ptl80=[0, line_8];
ptl81=[w, line_8];
ptl90=[0, line_9];
ptl91=[w, line_9];
ptl100=[0, line_10];
ptl101=[w, line_10];

pts_L00 = np.array([ptl00,ptl01], np.int32)
pts_L01 = np.array([ptl10,ptl11], np.int32)
pts_L02 = np.array([ptl20,ptl21], np.int32)
pts_L03 = np.array([ptl30,ptl31], np.int32)
pts_L04 = np.array([ptl40,ptl41], np.int32)
pts_L05 = np.array([ptl50,ptl51], np.int32)
pts_L06 = np.array([ptl60,ptl61], np.int32)
pts_L07 = np.array([ptl70,ptl71], np.int32)
pts_L08 = np.array([ptl80,ptl81], np.int32)
pts_L09 = np.array([ptl90,ptl91], np.int32)
pts_L10 = np.array([ptl100,ptl101], np.int32)











#when to start tracking objects
print ("Red line y:",str(line_down))
print ("Blue line y:", str(line_up))
line_down_color = (255,0,0)#blue
line_up_color = (0,0,255)#red
pt1 =  [0, line_down];#start at left pixel of the screen
pt2 =  [w, line_down];#with same as screen
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))


#When to stop tracking objects
pt5 =  [0, up_limit];
pt6 =  [w, up_limit];
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit];
pt8 =  [w, down_limit];
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#may be unessessary?
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#Elementos estructurantes para filtros morfoogicos
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while(cap.isOpened()):
##for  in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    #Lee una imagen de la fuente de video
    ret, frame = cap.read()
##    frame = image.array

    
    #Aplica substraccion de fondo
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#grayscale#-----------------------new---------------------------------------
    Bod=Ubody_cascade.detectMultiScale(gray,scaleFactor=10,minNeighbors=10)#-----------------------new------------------------------------

    
    for(x,y,w,h) in Bod:
        print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]
        img_item="myimg_item_imagr.png"
        cv2.imwrite(img_item,roi_gray)
        cv2.imshow('Frame',frame)
        stroke= 2
        color=(255,0,0)
        eyeboxW=x+w
        eyeboxH= y+h
        cv2.rectangle(frame,(x,y),(eyeboxW,eyeboxH),color,stroke)
        
        
        
        
        
        
        
        
        
    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,400,cv2.THRESH_BINARY)
        #Opening (erode->dilate) para quitar ruido.
      #  mask= cv2.erode(imBin, cv2.MORPH_OPEN,kernelOp)
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)
        #Closing (dilate -> erode) para juntar regiones blancas.
        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print ('DOWN:',Wave_counter)
        break


    _, contours0, hierarchy = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        if area > areaTH:

            
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                        # el objeto esta cerca de uno que ya se detecto antes
                        new = False
                        i.updateCoords(cx,cy)   #actualiza coordenadas en el objeto and resets age
                      
                        if i.going_DOWN(line_down,line_up) == True:
                            Wave_counter += 1;
                            print ("ID:",i.getId(),'crossed going down at',time.strftime("%c"))
                        break
                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()
                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()
                    if i.timedOut():
                        #sacar i de la lista persons
                        index = persons.index(i)
                        persons.pop(index)
                        del i     #liberar la memoria de i
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     

            cv2.circle(frame,(cx,cy), 5, (0,0,255), -1) #dot in centre of box
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
        
    for i in persons:

        cv2.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
        
  
    #################
    str_down = 'Waves Passed: '+ str(Wave_counter)
    
    #draw lines
    frame = cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
    frame = cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
    
    
    frame = cv2.polylines(frame,[pts_L00],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L01],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L02],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L03],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L04],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L05],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L06],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L07],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L08],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L09],False,(255,255,255),thickness=1)
    frame = cv2.polylines(frame,[pts_L10],False,(255,255,255),thickness=1)
    
   # cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
   # cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv2.LINE_AA)




            #diplay footage and mask
    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)    
    
    #preisonar ESC para salir
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

   
#################

#################
cap.release()
cv2.destroyAllWindows()