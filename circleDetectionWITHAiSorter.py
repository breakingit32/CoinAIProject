import imghdr
from tkinter import Toplevel
from cv2 import destroyAllWindows, imshow
import numpy as np
import cv2
import os
import time
import random
import tensorflow as tf

#Program that does what CircleDetection.py does, but uses AI model to sort images in correct folder.


def prepare(filepath):
    IMG_SIZE = 300
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (300, 300))
    return new_array.reshape(-1, 300, 300, 1)

#Source images Directory
DIR = "C:/Users/break/Documents/Python/CoinPictures4"
#Class names
CATEGORIES = ["Up", "NotUp"]
#Loop that goes through each image in DIR
for filename in os.listdir(DIR):
    print(filename)
    #Loading image
    img = cv2.imread(os.path.join(DIR, filename))
    path = r"C:/Users/break/Documents/Python/Penny"
    #Generating Random numbers to be used as the save name
    ranNum=random.sample(range(1, 1000), 3)
    x=0
    y=0
    i=0
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
    minDist = 500
    param1 = 50
    param2 = 20
    minRadius = 875
    maxRadius = 1000
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
    myObject=cv2.bitwise_and(img, img, mask=myMask)
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
        
        #Uncommnet for more info of the cicles
        '''
        print("CircleLocation: ", circlesLocations[num])
        print("Top Left: +/- " + str(radius) + " ", topLeft)
        print("Bottom Right: +/- " + str(radius) + " ", bottomRight)
        print("CircleRadius: ", circlesLocations[num][2])
        '''
    
    

#newFrame=img[bottomRight,topLeft]
        yes=myObject[topLeft[0]:topLeft[1], bottomRight[0]:bottomRight[1]]
        #print(path)
        
        ranNum[0]=ranNum[0]+1
        imgHSV=cv2.resize(imgHSV, (1080, 720))
        #Saving result image so the AI has a image to load. (Probably a better way of doing this, but this is what I found that works)
        cv2.imwrite(os.path.join(os.path.join(path, "Test"), filename+".png"), imgHSV)
        
        #Loading model 
        
        model = tf.keras.models.load_model("IsUpRight-1659136094.3350148.model")
        #Saving location Dir
        saveLocation = "C:/Users/break/Documents/Python/Penny/AISortTest"
        cv2.imwrite(os.path.join(saveLocation, "Test.png"), yes)
        #Running Model
        prediction = model.predict([prepare(os.path.join(saveLocation, "Test.png"))])
        #Saving image depending on model's prediction
        if CATEGORIES[int(prediction[0][0])] == "Up":
            yes=cv2.resize(yes, (300,300))
            saveLocation = os.path.join(saveLocation, "Up")
            #print(saveLocation)
            print("Saving")
            
            cv2.imwrite(os.path.join(saveLocation, "Penny" + str(ranNum[0])+str(ranNum[1])+str(ranNum[2])+".png"), yes)
        if CATEGORIES[int(prediction[0][0])] == "NotUp":
            yes=cv2.resize(yes, (300,300))
            saveLocation = os.path.join(saveLocation, "NotUp")
            #print(saveLocation)
            print("Saving")
            
            cv2.imwrite(os.path.join(saveLocation, "Penny" + str(ranNum[0])+str(ranNum[1])+str(ranNum[2])+".png"), yes)
        print(CATEGORIES[int(prediction[0][0])])
        
        
        
    
    

    if cv2.waitKey(1) & 0xff ==ord('q'):
        break

cv2.destroyAllWindows()