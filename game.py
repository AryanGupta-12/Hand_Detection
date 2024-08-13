import mediapipe as mp
import time as time
import cv2
import handTrackingModule as hm

cap = cv2.VideoCapture(0)
last_sum_time = time.time()
detector = hm.HandDetector()
delay = 4.0
score_left = 0
score_right = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    img = cv2.flip(img,1)
    lmlist = detector.findPosition(img,allHands=True,draw = False)
    
    tipID = [4,8,12,16,20]
    fingers1 = []
    fingers2 = []
    if (len(lmlist)==42) and (lmlist[21][0]=='Right'): 
        if (lmlist[25][2]<lmlist[24][2]):
            fingers1.append(1)
        else:
            fingers1.append(0)
        for i in range(1,5):
            if (lmlist[21+tipID[i]][3]<=lmlist[19+tipID[i]][3]):
                fingers1.append(1)
            else: 
                fingers1.append(0)
    if (len(lmlist)==42) and (lmlist[0][0]=='Left'):
        if (lmlist[4][2]>lmlist[3][2]):
                fingers2.append(1)
        else:
            fingers2.append(0)
        for i in range(1,5):
            if (lmlist[tipID[i]][3]<=lmlist[tipID[i]-2][3]):
                fingers2.append(1)
            else: 
                fingers2.append(0)
    current_time = time.time()
    instant_sum1 = 0
    instant_sum2 = 0
    if(current_time - last_sum_time)>=delay:
        for i in range(0,5):
            if(len(fingers1)==5) and (len(fingers2)==5):
                instant_sum1 += fingers1[i]
                instant_sum2 += fingers2[i]
        if(instant_sum1 == 0 and instant_sum2==5):
            score_right+=1
        elif(instant_sum1==5 and instant_sum2==2):
            score_right+=1
        elif(instant_sum1==2 and instant_sum2==0):
            score_right+=1
        elif(instant_sum1==instant_sum2):
            pass
        else:
            score_left+=1
        last_sum_time=current_time
    cv2.putText(img,str(int(score_left)),(100,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),3)
    cv2.putText(img,str(int(score_right)),(500,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
    cv2.putText(img,str('Timer:'+str(int(current_time - last_sum_time))),(250,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
