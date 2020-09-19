import cv2
import numpy as np
import face_recognition

imgOrig = face_recognition.load_image_file("/media/pavansince1999/Windows/Users/pavan's jarvis/Desktop/mini pro/Passport photo.jpg")
imgOrig = cv2.cvtColor(imgOrig,cv2.COLOR_BGR2RGB)
imgtest = face_recognition.load_image_file("/media/pavansince1999/Windows/Users/pavan's jarvis/Desktop/mini pro/IMG_20190210_094300_380.jpg")
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)


faceLoc = face_recognition.face_locations(imgOrig)[0]
encOrig = face_recognition.face_encodings(imgOrig)[0]
cv2.rectangle(imgOrig,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLoctest = face_recognition.face_locations(imgtest)[0]
enctest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLoctest[3],faceLoctest[0]),(faceLoctest[1],faceLoctest[2]),(255,0,0),2) 

res = face_recognition.compare_faces([encOrig],enctest)
cv2.putText(imgtest,f'{res}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Orig',imgOrig)
cv2.imshow('Test',imgtest)
cv2.waitKey(0)
