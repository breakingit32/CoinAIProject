import imghdr
from tkinter import Toplevel
from cv2 import destroyAllWindows, imshow
import numpy as np
import cv2
import os
import time
import random

width = 1920
height = 1080
cam=cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
x=0
while x != 100:
    ignore, frame = cam.read()
    img = frame
    x=x+1
while True:
    ignore, frame = cam.read()
    img = frame
    path = r"C:/Users/break/Documents/Python/Penny"
    #Generating Random numbers to be used as the save name
    ranNum=random.sample(range(1, 1000), 3)
    x=0
    y=0
    i=0
    topLeft =()
    bottomRight=()
    #cv2.imshow('camera', frame)
    #cv2.waitKey(0)
    #setting up for Hough Circle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    #cv2.waitKey(0)
    blurred = cv2.medianBlur(gray, 37)
    #cv2.imshow('blur', blurred)
    #cv2.waitKey(0)#cv2.bilateralFilter(gray,10,50,50)
    # Hough Circle Prams
    #Old Hough circle Prams

    minDist = 250
    param1 = 30 #500
    param2 = 50 #200 #smaller value-> more false circles
    minRadius = 140
    maxRadius = 200 #10

    #Current Hough Circle Prams
    #IMPORTANT!!! My input images have been 6000x4000 resolution. With the coins being 950ish pixels wide. 
    #If using different image size, or coin size. Prams will need to be updated.
    #minDist = 500
    #param1 = 50
    #param2 = 20 
    #minRadius = 875
    #maxRadius = 1000
    #Detecting circles
    circlesLocations=[]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)



        
        #Converting to HSV
    imgHSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Drawing cicles on the new imgHSV image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]: #i = x,y, Radius
            circlesLocations.append(i[0:3])
            
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
            cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 2)
            #cv2.putText(imgHSV, str(y), (i[0]-40, i[1]+40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            y=y+1

    #IMPORTANT!!! Line thickness must be -1. To fill in the circle completely
    #Setting HSV Range
    y=0
    lowerBound=np.array([0,0,0])
    upperBound=np.array([1,1,1])
    #Making the mask
    myMask=cv2.inRange(imgHSV,lowerBound, upperBound)
    #Applying the mask
    myObject=cv2.bitwise_and(img, img, mask=myMask)
    print(myObject.shape)
    for num in range(0,len(circlesLocations)) :
        radius = circlesLocations[num][2]
        circleX = circlesLocations[num][0]
        circlesY = circlesLocations[num][1]
        topLeft=[(circlesY-radius),(circlesY+radius)]
        bottomRight=[(circleX-radius),(circleX+radius)]
            #Debuging info
        print("")
        print("Circle Number: ", y)
        print("CircleLocation: ", circlesLocations[num])
        print("Top Left: +/- " + str(radius) + " ", topLeft)
        print("Bottom Right: +/- " + str(radius) + " ", bottomRight)
        
        print("CircleRadius: ", circlesLocations[num][2])

        #Checking to see if coin is at edge of the image
        if topLeft[0] > myObject.shape[0]:
            topLeft[0] = 0
        
        if bottomRight[0] > myObject.shape[1]:
            bottomRight[0]= 0
        
        if topLeft[1] > myObject.shape[0]:
            topLeft[1] = myObject.shape[0]
        
        if bottomRight[1] > myObject.shape[1]:
            bottomRight[1]= myObject.shape[1]
        
        #Debuging info

        print("CircleLocation: ", circlesLocations[num])
        print("Top Left: +/- " + str(radius) + " ", topLeft)
        print("Bottom Right: +/- " + str(radius) + " ", bottomRight)
        print("CircleRadius: ", circlesLocations[num][2])




    #newFrame=img[bottomRight,topLeft]
        yes=myObject[topLeft[0]:topLeft[1], bottomRight[0]:bottomRight[1]]
        
        #Saving image
        ranNum[0]=ranNum[0]+1
        #imgHSV=cv2.resize(imgHSV, (1080, 720))
        
        
        
        #yes=cv2.resize(yes, (300,300))
        #cv2.rectangle(img, ((circlesY-radius),(circlesY+radius)), ((circleX-radius),(circleX+radius)), (255,0,0), 2)
    cv2.imshow('my RIO', img)
    cv2.waitKey(1)
        #cv2.waitKey(0)


#cv2.moveWindow('camera', 0, 0)

cam.release()








    

    

    


cv2.destroyAllWindows()
