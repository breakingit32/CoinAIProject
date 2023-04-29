
import imghdr
from multiprocessing.resource_sharer import stop
from tkinter import Toplevel
from cv2 import destroyAllWindows, imshow
import numpy as np
import cv2
from asyncio.windows_events import NULL
import os
import time
import random
from pymata4 import pymata4
def contect2Ardunio(board):

    while True:
        try:
            board = pymata4.Pymata4()
            
        except RuntimeError:
            cv2.waitKey(1000)
            print("retrying")
            continue
        else:
            break
        
    return board
board = NULL
board = contect2Ardunio(board=board)
board.set_pin_mode_digital_output(6)
board.set_pin_mode_digital_output(7)
board.set_pin_mode_pwm_output(10)
def myCallBack1(value):
    global minDist
    minDist=value
    
    
    
def myCallBack2(value):
    global param1
    param1 = value

def myCallBack3(value):
    global param2
    param2 = value
 
def myCallBack4(value):
    global minRadius
    minRadius = value

def myCallBack5(value):
    global maxRadius
    maxRadius = value
def myCallBack6(value):
    global blurFactor
    blurFactor = value
def myCallBack7(value):
    global speed
    speed = value
def stopConveyor(board, pin1, pin2, time):
    cv2.waitKey(time*1000)
    board.digital_pin_write(pin1, 0)
    board.digital_pin_write(pin2, 0)
    cv2.waitKey(time*1000)
    board.digital_pin_write(pin1, 0)
    board.digital_pin_write(pin2, 1)
    cv2.waitKey(time*2000)

minDist = 1027
param1 = 184
param2 = 13
minRadius = 84
maxRadius = 91
blurFactor = 15
speed = 65
def detectCircle(frame, noCircles):
    x=0
    y=0
    i=0
    topLeft =()
    bottomRight=()
    #setting up for Hough Circle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 15)
    
    #Detecting circles
    circlesLocations=[]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 1027, param1=184, param2=13, minRadius=84, maxRadius=91)
    imgHSV=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Drawing cicles on the new imgHSV image
    if circles is None:
        noCircles = True
        return noCircles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]: #i = x,y, Radius
            circlesLocations.append(i[0:3])
            isCircle = True
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (255, 0, 0), 2)
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
    myObject=cv2.bitwise_and(frame, frame, mask=myMask)
    print(myObject.shape)
    for num in range(0,len(circlesLocations)) :
        radius = circlesLocations[num][2]
        circleX = circlesLocations[num][0]
        circlesY = circlesLocations[num][1]
        topLeft=[(circlesY-radius),(circlesY+radius)]
        bottomRight=[(circleX-radius),(circleX+radius)]
         #Debuging info
        

        #Checking to see if coin is at edge of the image
        if topLeft[0] > myObject.shape[0]:
            topLeft[0] = 0
        
        if bottomRight[0] > myObject.shape[1]:
            bottomRight[0]= 0
        
        if topLeft[1] > myObject.shape[0]:
            topLeft[1] = myObject.shape[0]
        
        if bottomRight[1] > myObject.shape[1]:
            bottomRight[1]= myObject.shape[1]
    frame = myObject
    noCircles = False
    return frame, noCircles
#Progam to make your dataset
minDist = 1027
param1 = 184
param2 = 13
minRadius = 84
maxRadius = 91
blurFactor = 15
speed = 65
#Source images Directory
width = 1080
height = 720
cam=cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
isCircle = False
cv2.namedWindow('myTrackbars')
cv2.resizeWindow('myTrackbars', 400, 350)
cv2.moveWindow('myTrackbars', width, 0)


cv2.createTrackbar('minDis', 'myTrackbars',minDist ,1980,myCallBack1)

cv2.createTrackbar('param1', 'myTrackbars',param1 ,200,myCallBack2)
cv2.createTrackbar('param2', 'myTrackbars', param2,200,myCallBack3)
cv2.createTrackbar('minRad', 'myTrackbars', minRadius,500,myCallBack4)
cv2.createTrackbar('MaxRad', 'myTrackbars', maxRadius,500,myCallBack5)
cv2.createTrackbar('blurred', 'myTrackbars', blurFactor,50,myCallBack6)
cv2.createTrackbar('SPEED', 'myTrackbars', speed,255,myCallBack7)
cv2.waitKey(1000)
#Loop that goes through each image in DIR
noCircles = False
while True:
    print('Start' + str(noCircles))
    board.pwm_write(10, speed)
    board.digital_pin_write(7, 0)
    board.digital_pin_write(6, 1)
    ignore, frame = cam.read()
   
    #Loading image
    
    x=0
    y=0
    i=0
    topLeft =()
    bottomRight=()
    #setting up for Hough Circle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 15) #cv2.bilateralFilter(gray,10,50,50)
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
   
    #Detecting circles
    circlesLocations=[]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)


    
        
        #Converting to HSV
    imgHSV=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Drawing cicles on the new imgHSV image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]: #i = x,y, Radius
            circlesLocations.append(i[0:3])
            
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (255, 0, 0), 2)
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
    myObject=cv2.bitwise_and(frame, frame, mask=myMask)
    print(myObject.shape)
    for num in range(0,len(circlesLocations)) :
        radius = circlesLocations[num][2]
        circleX = circlesLocations[num][0]
        circlesY = circlesLocations[num][1]
        topLeft=[(circlesY-radius),(circlesY+radius)]
        bottomRight=[(circleX-radius),(circleX+radius)]
        
        print("")
        #Uncommnet for more info of the cicles
        '''
        print("Circle Number: ", y)
        print("CircleLocation: ", circlesLocations[num])
        print("Top Left: +/- " + str(radius) + " ", topLeft)
        print("Bottom Right: +/- " + str(radius) + " ", bottomRight)
        print("CircleRadius: ", circlesLocations[num][2])

        '''
        #Checking to see if coin is at edge of the image
        if topLeft[0] > myObject.shape[0]:
            topLeft[0] = 0
        
        if bottomRight[0] > myObject.shape[1]:
            bottomRight[0]= 0
        
        if topLeft[1] > myObject.shape[0]:
            topLeft[1] = myObject.shape[0]
        
        if bottomRight[1] > myObject.shape[1]:
            bottomRight[1]= myObject.shape[1]
    
    
    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)
    cv2.imshow('blurred', blurred)
    cv2.imshow('mask', myObject)
    

    #cv2.moveWindow('camera', 0, 0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break;
cam.release()

    
'''isCircle == False
    x=0
    y=0
    i=0
    topLeft =()
    bottomRight=()
    #setting up for Hough Circle
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blurFactor)
    
    #Detecting circles
    circlesLocations=[]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    imgHSV=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Drawing cicles on the new imgHSV image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]: #i = x,y, Radius
            circlesLocations.append(i[0:3])
            isCircle = True
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (255, 0, 0), 2)
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
    myObject=cv2.bitwise_and(frame, frame, mask=myMask)
    print(myObject.shape)
    for num in range(0,len(circlesLocations)) :
        radius = circlesLocations[num][2]
        circleX = circlesLocations[num][0]
        circlesY = circlesLocations[num][1]
        topLeft=[(circlesY-radius),(circlesY+radius)]
        bottomRight=[(circleX-radius),(circleX+radius)]
         #Debuging info
        

        #Checking to see if coin is at edge of the image
        if topLeft[0] > myObject.shape[0]:
            topLeft[0] = 0
        
        if bottomRight[0] > myObject.shape[1]:
            bottomRight[0]= 0
        
        if topLeft[1] > myObject.shape[0]:
            topLeft[1] = myObject.shape[0]
        
        if bottomRight[1] > myObject.shape[1]:
            bottomRight[1]= myObject.shape[1]
    
    cv2.imshow('camera', myObject)'''
    

