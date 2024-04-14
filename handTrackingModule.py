import cv2
import mediapipe as mp
import time
class HandDetector():
    def __init__(self, mode= False, maxHands = 2, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode= self.mode, max_num_hands= self.maxHands, min_detection_confidence= self.detectionConf, min_tracking_confidence=self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:    
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,id_=0,allHands = False,draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            if not allHands:
                handlms = self.results.multi_hand_landmarks[0]
                for id, lm in enumerate(handlms.landmark):
                        h,w,c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmlist.append([id,cx,cy])
                        if draw and id==id_:
                            cv2.circle(img,(cx,cy),10,(255,255,0), cv2.FILLED)
            else:
                hand_data = list(zip(self.results.multi_hand_landmarks, self.results.multi_handedness))
                def sort_by_handedness(hand_info):
                    handlms, handedness = hand_info
                    return handedness.classification[0].label
        
                hand_data.sort(key=sort_by_handedness)
                for handlms, handedness in hand_data:
                    for id, lm in enumerate(handlms.landmark):
                        hand_id = handedness.classification[0].label
                        h,w,c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmlist.append([hand_id,id,cx,cy])
                        if draw and id==id_:
                            cv2.circle(img,(cx,cy),10,(255,255,0), cv2.FILLED)
        return lmlist

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img,3,allHands=True)
        if(len(lmlist)!=0):
            print(lmlist,'\n')
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(30,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)
        cv2.imshow("Video",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
    