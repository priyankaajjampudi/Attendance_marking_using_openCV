import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime 

path = "/media/pavansince1999/Windows/Users/pavan's jarvis/Desktop/mini pro/images"
images = []
classNames = []
l = os.listdir(path)
#print(l)
for img in l:
    curImg = cv2.imread(f'{path}/{img}')
    images.append(curImg)
    classNames.append(os.path.splitext(img)[0])

#print(classNames)  

def find_enc(images):
    enclist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)[0]
        enclist.append(enc)
    return enclist

def mark_attendance(name):
    with open('attendance.csv','r+') as f:
        myData = f.readlines()
        name_list = []
        for line in myData:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')
            
#mark_attendance('shantanu')

enclist_found = find_enc(images)
print("Encoding completed successfully")

cap = cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    cam_img = cv2.resize(img,(0,0),None,0.25,0.25)
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    
    faces_current_frame = face_recognition.face_locations(cam_img)
    enc_current_frame = face_recognition.face_encodings(cam_img,faces_current_frame)
    
    for encodeFace,faceLoc in zip(enc_current_frame,faces_current_frame):
        matches = face_recognition.compare_faces(enclist_found,encodeFace)
        faceDis = face_recognition.face_distance(enclist_found,encodeFace)
        #print(faceDis) 
        match_index = np.argmin(faceDis)
        
        if matches[match_index]:
            name = classNames[match_index]
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            mark_attendance(name)
    
    cv2.imshow('webcam',img)
    cv2.waitKey(1)  
