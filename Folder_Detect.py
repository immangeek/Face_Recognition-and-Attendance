import cv2 as cv
import numpy as np
import face_recognition as fc
import os
from datetime import datetime

path = "Attendance"
images = []
ClassNames = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curimage = cv.imread(f'{path}/{cl}')
    images.append(curimage)
    ClassNames.append(os.path.splitext(cl)[0])
print(ClassNames)

def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

        #print(myDataList)




encodelistknown = findEncodings(images)
print("Encoded Succesfully")

cap = cv.VideoCapture(0)

while True:
    sucess, img = cap.read()
    imgs = cv.resize(img,(0,0),None,0.25,0.25)
    imgs = cv.cvtColor(imgs,cv.COLOR_BGR2RGB)

    facesCurFrame = fc.face_locations(imgs)
    encodeCurFrame = fc.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip (encodeCurFrame,facesCurFrame):
        matches = fc.compare_faces(encodelistknown,encodeFace)
        faceDis = fc.face_distance(encodelistknown,encodeFace)
    
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = ClassNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4 
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv.FILLED)
            cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)



    cv.imshow('webcam',img)
    cv.waitKey(1)





