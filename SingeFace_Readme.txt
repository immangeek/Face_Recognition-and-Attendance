#importing Libraries
from base64 import encode
from unittest import result
from xml.sax.handler import feature_namespace_prefixes
import cv2 as cv
import numpy as np
import face_recognition as fc

#import Image & Convert Black & White
imgimman = fc.load_image_file('Reference/imman.jpg')
imgimman = cv.cvtColor(imgimman,cv.COLOR_BGR2RGB)

imgtest = fc.load_image_file('Reference/imman_test.jpg')
imgtest = cv.cvtColor(imgtest,cv.COLOR_BGR2RGB)

#Encoding Face
faceLoc = fc.face_locations(imgimman)[0]
encodeimman = fc.face_encodings(imgimman)[0]
cv.rectangle(imgimman,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = fc.face_locations(imgtest)[0]
encodeTest = fc.face_encodings(imgtest)[0]
cv.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

#Comparing Faces for Results
results = fc.compare_faces([encodeimman],encodeTest)
faceDis = fc.face_distance([encodeimman],encodeTest)

print(results,faceDis)
cv.putText(imgtest,f"{results} {round(faceDis[0],2)}",(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv.imshow('Imman',imgimman)
cv.imshow('Imman Test',imgtest)

cv.waitKey(0)
