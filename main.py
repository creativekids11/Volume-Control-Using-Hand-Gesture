import cv2
import mediapipe as mp  
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import *

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
vc=cast(interface,POINTER(IAudioEndpointVolume))
Range=vc.GetVolumeRange()
minR,maxR=Range[0],Range[1]

mpHands = mp.solutions.hands
Hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
PTime=0
vol=0
volBar=400
volPer=0
cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    lmList=[]
    success , img = cap.read() 
    converted_image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    results = Hands.process(converted_image)

    if results.multi_hand_landmarks:
        for hand_in_frame in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,hand_in_frame, mpHands.HAND_CONNECTIONS)
        for id,lm in enumerate(results.multi_hand_landmarks[0].landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            lmList.append([cx,cy])

        if len(lmList)!=0:
            x1,y1=lmList[4][0],lmList[4][1]
            x2,y2=lmList[8][0],lmList[8][1]

            cv2.circle(img,(x1,y1),15,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(255,0,0),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3,cv2.FILLED)
            length=math.hypot(x2-x1-30,y2-y1-30)

            vol=np.interp(length, [50,300], [minR,maxR])
            volBar=np.interp(length, [50,300], [400,150])
            volPer=np.interp(length, [50,300], [0,100])

            cv2.rectangle(img,(50,150),(85,400),(255,0,0))
            cv2.rectangle(img,(50,int(volBar)),(85,400),(255,0,0),cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (85,450),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
            vc.SetMasterVolumeLevel(vol,None)
    
    CTime=time.time()
    fps=1/(CTime-PTime)
    PTime=CTime
    cv2.putText(img,str(int(fps)),(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)

    cv2.imshow("Hand Tracking", img) 

    if cv2.waitKey(1) == 113: # 113 - Q
        break