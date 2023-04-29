from turtle import color
import cv2
import tensorflow as tf
import os
import random
import argparse
import imutils
import time
from cv2 import Canny
from matplotlib.pyplot import colorbar
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
def prepare(image):
    IMG_SIZE = 400
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
def RotateTillUp(img):
    done = False
    saveLocation = "C:/Users/break/Documents/Python/Penny/Upright"
    template = cv2.imread('C:/Users/break/Documents/Python/Penny/TestPenny00.png', 0)
    template = cv2.resize(template, (300,300))
    h, w = np.shape(template)
    i = 0
    i = i + 1
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    possiblity = []
    possiblityNDegree = []
    degree = []
    x=1
    
    while x < 360:
        
        rotate = imutils.rotate(img, x)
        result = cv2.matchTemplate(rotate, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(max_val)
        possiblity.append(round(max_val*100, 2))
        possiblityNDegree.append([round(max_val*100, 2), x])
        degree.append(x)
        
        threshold = 0.95
        '''
        locations = np.nonzero(result >= threshold)
        print(len(locations))
        
        for top_left in zip(*locations[::-1]):
            rotate = cv2.cvtColor(rotate, cv2.COLOR_GRAY2BGR)
            rotate = cv2.rectangle(img = rotate, pt1= top_left, pt2= (top_left[0] + w, top_left[1] + h), color= (0,255,0), thickness=10)
            x = 359
         '''
        x = x + 1
        
    max = np.amax(possiblity)
    possiblityNDegree = sorted(possiblityNDegree, key= lambda x: x[0])
    print(possiblityNDegree[-1][1])
    img = imutils.rotate(img, possiblityNDegree[-1][1])
    cv2.imwrite(os.path.join(saveLocation, str(time.time())+'.png'), img)
    done = True
    return img, done

thresholds = 1
colors = 1
cap = cv2.VideoCapture(0)
CATEGORIES = [ "NoUpRight","Upright"]
#model = tf.keras.models.load_model("Gen1-1000x1000Rotation-1659975524.609461.model")

cv2.namedWindow('myTrackbars')
cv2.resizeWindow('myTrackbars', 400, 700)
cv2.moveWindow('myTrackbars', 1280, 0)
saveLocation = "C:/Users/break/Documents/Python/Penny/Upright"
minDist = 50
param1 = 50
param2 = 7 
minRadius = 80
maxRadius = 81
cv2.createTrackbar('minDis', 'myTrackbars',minDist ,300,myCallBack1)
cv2.createTrackbar('param1', 'myTrackbars', param1,300,myCallBack2)
cv2.createTrackbar('param2', 'myTrackbars', param2,300,myCallBack3)
cv2.createTrackbar('minRadius', 'myTrackbars', minRadius,300,myCallBack4)
cv2.createTrackbar('maxRadius', 'myTrackbars', maxRadius,300,myCallBack5)

test = cv2.imread('C:/Users/break/Documents/Python/Penny/TestPenny00.png')
cv2.imshow("ddd", test)
while True:
    print("scan now")
    time.sleep(2)
    ignore, frame = cap.read()
    y=0
    belt = frame[0: 1080, 100: 340]
    colorBelt = frame[0: 1080, 100: 340]
    
    gray = cv2.cvtColor(colorBelt, cv2.COLOR_BGR2GRAY)
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
    
    #Detecting circles
    circlesLocations=[]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    
    imgHSV=cv2.cvtColor(colorBelt, cv2.COLOR_BGR2HSV)
    
    #Drawing cicles on the new imgHSV image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]: #i = x,y, Radius
            circlesLocations.append(i[0:3])
            
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (0, 0, 0), -1)
            cv2.circle(imgHSV, (i[0], i[1]), i[2], (255, 0, 0), 2)
            #cv2.putText(imgHSV, str(y), (i[0]-40, i[1]+40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            y=y+1
    else:
        print("no circles")
    #IMPORTANT!!! Line thickness must be -1. To fill in the circle completely
    #Setting HSV Range
    y=0
    lowerBound=np.array([0,0,0])
    upperBound=np.array([1,1,1])
    #Making the mask
    myMask=cv2.inRange(imgHSV,lowerBound, upperBound)
    #Applying the mask
    myObject=cv2.bitwise_and(colorBelt, colorBelt, mask=myMask)
    for num in range(0,len(circlesLocations)) :
        radius = circlesLocations[num][2]
        circleX = circlesLocations[num][0]
        circlesY = circlesLocations[num][1]
        topLeft=[(circlesY-radius),(circlesY+radius)]
        bottomRight=[(circleX-radius),(circleX+radius)]
        
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

        #cv2.rectangle(myObject, (bottomRight[0], bottomRight[1]-radius+10), (topLeft[0]+radius-10, topLeft[1]), (255), 3)
        cv2.imshow('belt', myObject)

#newFrame=img[bottomRight,topLeft]
        yes=myObject[topLeft[0]:topLeft[1], bottomRight[0]:bottomRight[1]]
        yes = cv2.resize(yes, (300,300))
        cv2.imshow('yes', yes)
        cv2.imwrite(os.path.join(saveLocation, str(time.time())+'.png'), yes)
        #res, done = RotateTillUp(yes)
        #while done == True:
            #cv2.imshow('s', res)
            #cv2.waitKey(100)
           # break
    '''
    gray_belt = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray_belt, 3)
    gray_beltBlur = cv2.GaussianBlur(gray_belt, (7, 7), 1)
    ignore, threshol3 = cv2.threshold(median, 150, 255, cv2.THRESH_BINARY)
    medianRound2 = cv2.medianBlur(threshol3, 3)
    deniose = cv2.fastNlMeansDenoising(medianRound2, None, 20, 7, 21)
    ignore, threshol3R2 = cv2.threshold(medianRound2, 133, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgCanny = cv2.Canny(gray_beltBlur, thresholds, colors)
    contours, _ = cv2.findContours(deniose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    '''
    '''
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        width = (x+w)-x
        height = (y+h)-y
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        area = cv2.contourArea(cnt)
        side = area/4
        if area > 2000 and area < 21500 and width > 10 and height > 10 and height < 170 and width <170:
            
            coinPOI = colorBelt[y:(y+h), x:(x+w)]
            #cv2.rectangle(belt, (x, y), (x+w, y+h), (0,255,0),3)
            coinPOI = cv2.resize(coinPOI, (300,300))
            cv2.imshow("CoinPIO", coinPOI)
            
            y=3500
            x=0
            
            while y < 360:
                
                rotate = imutils.rotate(coinPOI, x)
                cv2.imshow('yes', rotate)
                cv2.waitKey(1)
                
                #prediction = model.predict([prepare(coinPOI)])
                    #Saving image depending on model's prediction
                ## print(CATEGORIES[int(prediction[0][0])])
                   # cv2.waitKey(100)
                    
                    
                    
                   # break
                    
               # if CATEGORIES[int(prediction[0][0])] == "NoUpRight":
                    #x=x+1
                   # rotate=cv2.resize(rotate, (400,400))
                    
                   # print(CATEGORIES[int(prediction[0][0])])
                    
                
                y=y+1
            
            print(width)
            print(height)
            print(str(side) + "Sides")
            print(str(area) + "AREA")
        elif area < 2000:
            clearedUp = cv2.fillPoly(threshol3R2, approx, (255,0,0))
            
            
    cv2.imshow("belt", belt)
    
    
    
    
    
    cv2.imshow("80", threshol3R2)
    '''
    
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()