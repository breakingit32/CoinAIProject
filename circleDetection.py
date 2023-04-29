import imghdr
from tkinter import Toplevel
from cv2 import destroyAllWindows, imshow
import numpy as np
import cv2
import os
import time
import random

#Progam to make your dataset

#Source images Directory
DIR = "C:/Users/break/Pictures/2023_01_13"
#Loop that goes through each image in DIR
for filename in os.listdir(DIR):
    print(filename)
    #Loading image
    img = cv2.imread(os.path.join(DIR, filename))
    cv2.imshow('ss', img)
    cv2.waitKey(0)
    path = r"C:/Users/break/Pictures/2023_01_11"
    #Generating Random numbers to be used as the save name
    
    topLeft =()
    bottomRight=()
    #setting up for Hough Circle
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 37) #cv2.bilateralFilter(gray,10,50,50)
    # Hough Circle Prams
    #Old Hough circle Prams
    '''
    minDist = 250
    param1 = 30 #500
    param2 = 50 #200 #smaller value-> more false circles
    minRadius = 275
    maxRadius = 300 #10
    '''
    #Current Hough Circle Prams
    #IMPORTANT!!! My input images have been 6000x4000 resolution. With the coins being 950ish pixels wide. 
    #If using different image size, or coin size. Prams will need to be updated.
    minDist = 20000
    param1 = 50
    param2 = 30 
    minRadius = 1500
    maxRadius = 1800
    #Detecting circles
    circlesLocations=[]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)


    largest_circle = max(circles[0], key=lambda x: x[2])
    print(largest_circle)
# Get the coordinates and radius of the largest circle
    x, y, r = largest_circle
    x = int(x)
    y = int(y)
    y = int(r)
    print(x, y, r)
        #Converting to HSV
    imgHSV=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
   
    
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
    print(myObject)
    
    topLeft=[int(y-r),int(y+r)]
    bottomRight=[int(x-r),int(x+r)]
        #Debuging info
    print("")
    print("Circle Number: ", y)
    print("CircleLocation: ")
    print("Top Left: +/- " + str(r) + " ", topLeft)
    print("Bottom Right: +/- " + str(r) + " ", bottomRight)
    
    

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

    




#newFrame=img[bottomRight,topLeft]
    yes=myObject[topLeft[0]:topLeft[1], bottomRight[0]:bottomRight[1]]
    cv2.imshow('s', yes)
    cv2.waitKey(0)
    print(path)
    #Saving image
    
    imgHSV=cv2.resize(imgHSV, (1080, 720))
    
    cv2.imwrite(os.path.join(os.path.join(path, "Test"), filename+".png"), imgHSV)
    
    
    #yes=cv2.resize(yes, (400,400))
    
    
    cv2.imwrite(os.path.join(os.path.join(path, "Front"), filename + ".jpeg"), yes)

    

    


cv2.destroyAllWindows()
