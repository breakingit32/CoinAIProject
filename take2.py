from inspect import isclass
from re import A, M
import threading
from cv2 import circle
import serial                                 # add Serial library for Serial communication
import cv2 
import time
import queue
import numpy as np
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


minDist = 10
param1 = 184
param2 = 18
minRadius = 84
maxRadius = 91
blurFactor = 15
speed = 65


def mains( frame, arduino):
        minDist = 10
        param1 = 184
        param2 = 18
        minRadius = 84
        maxRadius = 91
        blurFactor = 15
        speed = 65
        width = 1080
        height = 720
        


                  #read the serial data and print it as line



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
        arduino.write(str.encode("On"))
         
            #print('Start' + str(noCircles))
                                                 #infinite loop
            #input_data = input()                  #waits until user enters data
            #print("you entered", input_data)      #prints the data for confirmation
        isCircle = False
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
        print(type(circles))
        #Drawing cicles on the new imgHSV image
        if circles is not None:
            
            circles = np.uint16(np.around(circles))
            #time.sleep(5)
            
            for i in circles[0,:]: #i = x,y, Radius
                circlesLocations.append(i[0:3])
                
                cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
                #cv2.circle(imgHSV, (i[0], i[1]), i[2], (255, 0, 0), 2)
                #cv2.putText(imgHSV, str(y), (i[0]-40, i[1]+40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
                y=y+1
        print(sum(1 for i in circlesLocations))
        if(sum(1 for i in circlesLocations) <= 1):
            isCircle = False
            
            
        if(sum(1 for i in circlesLocations) == 0):
            
            isCircle = True
        if(isCircle == True):
            print("SentOn")
            time.sleep(1)
            arduino.write(str.encode("On"))  
        if(isCircle == False):
            print("SentOff")
            time.sleep(1)
            arduino.write(str.encode("Off"))
            #DO AI HERE
            time.sleep(1)
            arduino.write(str.encode("On"))
            time.sleep(3)
            isCircle = True
                
                
                
                
                
            #IMPORTANT!!! Line thickness must be -1. To fill in the circle completely
            #Setting HSV Range
        y=0
        lowerBound=np.array([0,0,0])
        upperBound=np.array([1,1,1])
        #Making the mask
        myMask=cv2.inRange(imgHSV,lowerBound, upperBound)
        #Applying the mask
        myObject=cv2.bitwise_and(frame, frame, mask=myMask)
        #print(myObject.shape)
        for num in range(0,len(circlesLocations)) :
            radius = circlesLocations[num][2]
            circleX = circlesLocations[num][0]
            circlesY = circlesLocations[num][1]
            topLeft=[(circlesY-radius),(circlesY+radius)]
            bottomRight=[(circleX-radius),(circleX+radius)]
            
            #print("")
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
        
        
        #cv2.imshow('frame', frame)
        #cv2.imshow('gray', gray)
        #cv2.imshow('blurred', blurred)
        #cv2.imshow('mask', myObject)
        
        #cv2.moveWindow('camera', 0, 0)
        if cv2.waitKey(1) & 0xff == ord('q'):
            arduino.write(str.encode("Off"))







width = 1080
height = 720
cam=cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
Arduino_Serial = serial.Serial('com2',9600) 
time.sleep(2) #Create Serial port object called arduinoSerialData
print (Arduino_Serial.readline())
ignore, frame = cam.read()

while True:
    ignore, frame = cam.read()
    mains(frame, Arduino_Serial)
    cv2.imshow("frame", frame)
    
    
    #cv2.moveWindow('camera', 0, 0)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()


    #t2.start()
    
    