
def removeBackGround(imgage):
    img_mask = cv2.imread('C:/Users/break/Documents/Python/Wheat_Reverse_Mask_300px.jpg', 0)
    
    b, g, r = cv2.split(imgage)
    img_output = cv2.merge([b, g, r, img_mask], 4)

    return img_output

def preProcessing(filePath):
    img = cv2.imread(filePath)
    frameHSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    print(img.shape)
    lowerBound=np.array([1,57,48])
    upperBound=np.array([36,255,125])
 
    lowerBound2=np.array([67,57,48])
    upperBound2=np.array([21,255,125])
 
    myMask=cv2.inRange(frameHSV,lowerBound,upperBound)
    myMask2=cv2.inRange(frameHSV,lowerBound2,upperBound2)
 
    myMask=myMask | myMask2
    
    myMask = cv2.resize(myMask, (300,300))
    #myMask=cv2.add(myMask,myMask2)
    #myMask=np.logical_or(myMask,myMask2)
    print(myMask.shape)
    #myMask=cv2.bitwise_not(myMask)
    
    myMaskSmall=cv2.resize(myMask,(int(300),int(300)))
    mySelection=cv2.bitwise_and(img,img, mask=myMask)
    
    mySelection=cv2.resize(mySelection,(int(300),int(300)))
    
    
    
    return myMaskSmall

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import os


saveLocation = "C:/Users/break/Documents/Python/Penny/Upright"
template = cv2.imread('C:/Users/break/Documents/Python/Penny/TestPenny15.png', 0)
#template = cv2.resize(template, (481,475))
h, w = np.shape(template)
i = 0

DIR = 'C:/Users/break/Documents/Python/Penny/MoreNotUpRight 300x300'

#Class names
CATEGORIES = ["Up", "NotUp"]
#Loop that goes through each image in DIR
for filename in os.listdir(DIR):
    print(filename)
    #Loading image
    img = cv2.imread(os.path.join(DIR, filename), 0)
    i = i + 1
    
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
    cv2.imwrite(os.path.join(saveLocation, '{:06d}'.format(i+1)+'.png'), img)
 