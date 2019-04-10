import cv2
import numpy as np
import streamlink
#import os


res = '720p'
filename = 'E:\\masters_Video_cap\\test.AVI'#set video file name


STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)
    
    
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height



streams = streamlink.streams("http://46.4.36.73/hls/bundoransurfco/playlist.m3u8")#"https://youtu.be/bNLy-XXYxcw")
quality='best'
image= cv2.VideoCapture(streams[quality].to_url())#
fps=image.get(cv2.CAP_PROP_FPS);
length=image.get(cv2.CAP_PROP_FRAME_WIDTH)
hight=image.get(cv2.CAP_PROP_FRAME_HEIGHT)
res=(length,hight)
#out=cv2.VideoWriter(filename,'avi', fps,get_dims(image, res))
out=cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'XVID'), 1,get_dims(image,(1280, 720)))
#out=cv2.VideoWriter()
#frame_time = int((1.0 / 30) * 1000.0)

while True:
    ret, frame = image.read()
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
image.release()
out.release()
cv2.destroyAllWindows()