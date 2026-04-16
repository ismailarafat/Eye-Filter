import cv2
import mediapipe as mp
import math
import numpy as np
import time
import pygame
import platform
# Mesure eye --> create mask in overlay video --> overlay video will play after first blink --> mp3 wil also play

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


system = platform.system()
if system == "Darwin":
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
elif system == "Windows": 
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("sharingan_eye.mp4")


pygame.mixer.init()
pygame.mixer.music.load("Quotes.mp3")
music_started = False
blink_start_time = 0


def distance(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)


blink_count = 0
prev_closed = False


def overlay(video, frame, x, y , alpha_scale = 1):
    h, w = video.shape[:2]
    x_end = int(min(x+w, frame.shape[1]))
    y_end = int(min(y+h, frame.shape[0]))
    x =  int(max(x,0))
    y =  int(max(y,0))
    crop_w = x_end - x
    crop_h = y_end - y
    if crop_w <= 0 or crop_h <= 0:
        return frame
    video_crop = video[:crop_h, :crop_w]
    if video.shape[2] == 4:
        alpha = np.clip((video_crop[:,:,3]/255.0) * alpha_scale, 0, 1)
        for c in range(3):
            frame[y: y_end, x: x_end, c] = (alpha * video_crop[:,:,c] + (1-alpha) * frame[y: y_end, x: x_end, c]).astype(np.uint8)
    else:
        mask = np.zeros((crop_h, crop_w), dtype = np.float32)
        cx, cy = crop_w //2 , crop_h//2
        radius = min(cx, cy)
        Y, X = np.ogrid[:crop_h, :crop_w]
        distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = np.clip(1- (distance / radius), 0, 1) * alpha_scale
        mask = np.clip(mask , 0, 1)
        for c in range(3):
            frame[y: y_end, x: x_end, c] = (mask * video_crop[:,:,c] + (1-mask ) * frame[y: y_end, x: x_end, c]).astype(np.uint8)
    return frame



def subtitle(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    h, w, _ = frame.shape
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - text_w) // 2
    y = h - 50
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


with mp_face_mesh.FaceMesh(
    refine_landmarks = True,
    min_detection_confidence = 0.8,
    min_tracking_confidence = 0.8
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        text = ""
        ret2, frame2 = cap2.read()
        if not ret2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = cap2.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                topL    = face_landmark.landmark[159]
                bottomL = face_landmark.landmark[145]
                leftL   = face_landmark.landmark[33]
                rightL  = face_landmark.landmark[133]
                earL = distance(topL, bottomL) / distance(leftL, rightL)

                topR    = face_landmark.landmark[386]
                bottomR = face_landmark.landmark[374]
                leftR   = face_landmark.landmark[263]
                rightR  = face_landmark.landmark[362]
                earR = distance(topR, bottomR) / distance(leftR, rightR)

                now_closed = earL<0.2  and earR< 0.2
                if prev_closed ==  False and now_closed == True:
                    blink_count += 1
                prev_closed = now_closed
                if blink_count >= 1:
                    if not music_started:
                        pygame.mixer.music.play()
                        blink_start_time = time.time()
                        music_started = True
                    current_time = time.time() - blink_start_time
                    if 1 <= current_time <= 3:
                        text = "EACH OF LIVES DEPENDANCE"
                    elif 4 <= current_time <= 6:
                        text = "AND BOUND BY OUR INDIVIDUAL KNOWLEDGE"
                    elif 7 <= current_time <= 8:
                        text = "AND OUR AWARENESS"
                    elif 9 <= current_time <= 12:
                        text = "ALL THAT IS WHAT WE CALL REALITY"
                    elif 14 <= current_time <= 16:
                        text = "BOTH KNOWLEDGE AND AWARENESS ARE EQUIVOCAL"
                    elif 17 <= current_time <= 20:
                        text = "ONE'S REALITY MIGHT BE ANOTHER'S ILLUSION"
                    elif 21 <= current_time <= 23:  
                        text = "WE ALL LIVE INSIDE OUR FANTASIES"

                h, w, _ = frame.shape
                iris_points_L = [468, 469, 470, 471, 472]
                points_L = [] 
                for i in iris_points_L:
                    p = face_landmark.landmark[i]
                    x, y = int(p.x * w), int(p.y * h)
                    points_L.append((x,y))
                cxL = sum(p[0] for p in points_L) // len(points_L)
                cyL = sum(p[1] for p in points_L) // len(points_L)
                distanceListL = [math.sqrt((p[0] - cxL)**2 + (p[1] - cyL)**2) for p in points_L]
                radiusL = sum(distanceListL) // len(distanceListL)
                sizeL = int(radiusL * 2)
                xL = cxL - radiusL
                yL = cyL - radiusL

                if not now_closed and 0 <= xL < w and 0 <= yL < h and blink_count >= 1 and (time.time()-blink_start_time) <= 26:
                    videoL = cv2.resize(frame2, (sizeL, sizeL))
                    frame = overlay(videoL, frame, xL, yL, alpha_scale=1.0)

                iris_points_R = [473, 474, 475, 476, 477]
                points_R = []
                for i in iris_points_R:
                    p = face_landmark.landmark[i]
                    x, y = int(p.x * w), int(p.y * h)
                    points_R.append((x, y))
                cxR = sum(p[0] for p in points_R) // len(points_R)
                cyR = sum(p[1] for p in points_R) // len(points_R)
                distanceListR = [math.sqrt((p[0]-cxR)**2 + (p[1]-cyR)**2) for p in points_R]
                radiusR = sum(distanceListR) // len(distanceListR)
                sizeR = int(radiusR * 2)
                xR = cxR - radiusR
                yR = cyR - radiusR

                if not now_closed and 0 <= xR < w and 0 <= yR < h and blink_count >= 1 and (time.time()-blink_start_time) <= 26:
                    videoR = cv2.resize(frame2, (sizeR, sizeR))
                    frame = overlay(videoR, frame, xR, yR, alpha_scale=1.0)
                
        frame  = cv2.flip(frame, 1)
        if text:
            subtitle(frame, text)
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


cap.release()
cap2.release()
cv2.destroyAllWindows()


        