import imghdr
from tkinter import Toplevel
from cv2 import destroyAllWindows, imshow
import numpy as np
import cv2
import os
import time
import random

img = cv2.imread("C:/Users/break/Documents/Python/circleDetectionTest2.jpg")
path = r"C:/Users/break/Documents/Python/Penny"

randomNumbers=random.sample(range(1, 1000), 2)
x=0
y=0
i=0
topLeft =()
bottomRight=()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 35) #cv2.bilateralFilter(gray,10,50,50)
# Hough Circle Prams
minDist = 100
param1 = 30 #500
param2 = 50 #200 #smaller value-> more false circles
minRadius = 5
maxRadius = 500 #10

#Detecting circles
circlesLocations=[]
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)


while x != 1:
    
    #Converting to HSV
    imgHSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #Drawing cicles on the new imgHSV image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]: #i = x,y, Radius
            circlesLocations.append(i[0:3])
            
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.putText(imgHSV, str(y), (i[0]-40, i[1]+40), cv2.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 4)
            y=y+1
    
     #IMPORTANT!!! Line thickness must be -1. To fill in the circle completely
    #Setting HSV Range
    
    lowerBound=np.array([0,0,0])
    upperBound=np.array([1,1,1])
    #Making the mask
    myMask=cv2.inRange(imgHSV,lowerBound, upperBound)
    #Applying the mask
    myObject=cv2.bitwise_and(img, img, mask=myMask)
    print(myObject.shape)
    y=0
    for num in range(0,len(circlesLocations)) :
        print(circlesLocations)
        radius = circlesLocations[num][2]
        circleX = circlesLocations[num][0]
        circlesY = circlesLocations[num][1]
        topLeft=[(circlesY-radius),(circlesY+radius)]
        bottomRight=[(circleX-radius),(circleX+radius)]
        print("")
        print("Circle Number: ", y)
        print("CircleLocation: ", circlesLocations[num])
        print("Top Left: +/- " + str(radius) + " ", topLeft)
        print("Bottom Right: +/- " + str(radius) + " ", bottomRight)
        
        print("CircleRadius: ", circlesLocations[num][2])
        if topLeft[0] > myObject.shape[0]:
            topLeft = 0
        
        if bottomRight[0] > myObject.shape[1]:
            bottomRight[0]= 0
        
        if topLeft[1] > myObject.shape[0]:
            topLeft = myObject.shape[0]
        
        if bottomRight[1] > myObject.shape[1]:
            bottomRight[1]= myObject.shape[1]
        


        print("CircleLocation: ", circlesLocations[num])
        print("Top Left: +/- " + str(radius) + " ", topLeft)
        print("Bottom Right: +/- " + str(radius) + " ", bottomRight)
        print("CircleRadius: ", circlesLocations[num][2])

        
        

    #newFrame=img[bottomRight,topLeft]
        yes=myObject[topLeft[0]:topLeft[1], bottomRight[0]:bottomRight[1]]
        
        gray = cv2.cvtColor(yes, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 35) #cv2.bilateralFilter(gray,10,50,50)
        # Hough Circle Prams
        minDist = 100
        param1 = 30 #500
        param2 = 50 #200 #smaller value-> more false circles
        minRadius = 5
        maxRadius = 500 #10

        #Detecting circles
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        imgHSV=cv2.cvtColor(yes, cv2.COLOR_BGR2HSV)
        
    #Drawing cicles on the new imgHSV image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]: #i = x,y, Radius
            
            
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
            
            y=y+1
    
     #IMPORTANT!!! Line thickness must be -1. To fill in the circle completely
    #Setting HSV Range
    
        lowerBound=np.array([0,0,0])
        upperBound=np.array([1,1,1])
        #Making the mask
        myMask=cv2.inRange(imgHSV,lowerBound, upperBound)
        #Applying the mask
        myObject=cv2.bitwise_and(yes, yes, mask=myMask)
            
        cv2.imwrite(os.path.join(path, "Penny" + str(y)+".png"), yes)
        
        cv2.imshow('my mask', myObject)
        cv2.waitKey(500)
        destroyAllWindows()
        y=y+1
    x=1
    # Show result for testing:
    
    

    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cv2.imshow('imgHSV', imgHSV)
cv2.waitKey(0)
print(bottomRight)
print(len(circlesLocations))
cv2.destroyAllWindows()