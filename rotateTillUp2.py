
import cv2 
import tensorflow as tf
import os
import random
import argparse
from pymata4 import pymata4

import imutils
import time

import tqdm
#stepper motor
def moveCWforDegrees(degree, board):
    spr = 200
    import time
    degrees = spr/360
    
    degrees2drive = degrees * degree
    print(degrees2drive)
    
    
    board.set_pin_mode_digital_output(4) #Drive Pin
    board.set_pin_mode_digital_output(3)
    #Step Pin
    board.digital_write(3, 0)
    board.digital_write(4, 0)
    

    #Step Pin
    board.digital_write(2, 1)
    x=0

    while x <= degrees2drive:
        
        #DirPin High

        board.digital_pin_write(3, 1)
        time.sleep(0.01)
        board.digital_pin_write(3, 0)
        
        x=x+1

#def DCMotorRun(speed, board, pin1, pin2, enablePin):
    


DATADIR = "C:/Users/break/Documents/Python/CoinPictures"
ranNum=random.sample(range(1, 1000), 5)
image = cv2.imread("C:/Users/break/Documents/Python/CoinPictures3/Penny144832.png")
CATEGORIES = [ "NoUpRight", "Upright"]
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
x=0
y=0
model = tf.keras.models.load_model("Gen4.1-1000x1000Rotation-1665967332.2630396.model")
def prepare(filepath):
    IMG_SIZE = 400
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# rotate our image by 45 degrees around the center of the image
start = time.time()


for img in os.listdir(DATADIR):
    y=0
    x=0
    ranNum=random.sample(range(1, 1000), 5)
    image = cv2.imread(os.path.join(DATADIR, img))
    while y < 360:
        
        rotate = imutils.rotate(image, x)
        #cv2.imshow('yes', rotate)
        
        #cv2.waitKey(1)
        
        #print(y)
        

        
            
            #Loading model 
            
        
            #Saving location Dir
        saveLocation = "C:/Users/break/Documents/Python/Penny/"
        cv2.imwrite(os.path.join(saveLocation, "Test.png"), rotate)
            #Running Model
        prediction = model.predict([prepare(os.path.join(saveLocation, "Test.png"))])
            #Saving image depending on model's prediction
        if CATEGORIES[int(prediction[0][0])] == "Upright":
            
            #rotate=cv2.resize(rotate, (400,400))
            saveLocation = os.path.join(saveLocation, "Correct3")
                #print(saveLocation)
            #print("Saving")
            print(CATEGORIES[int(prediction[0][0])])
            cv2.imwrite(os.path.join(saveLocation, "Penny12" + str(ranNum[1])+str(ranNum[4])+str(y)+".png"), rotate)
            #cv2.destroyAllWindows()
            #print(CATEGORIES[int(prediction[0][0])])
            #cv2.waitKey(1000)
            
            
            
            break
            
        if CATEGORIES[int(prediction[0][0])] == "NoUpRight":
            x=x+1
            rotate=cv2.resize(rotate, (400,400))
            saveLocation = os.path.join(saveLocation, "Wrong3")
                #print(saveLocation)
            
                
            cv2.imwrite(os.path.join(saveLocation, "Penny12" + str(ranNum[0])+str(ranNum[4])+str(y)+".png"), rotate)
            #cv2.destroyAllWindows()
            print(CATEGORIES[int(prediction[0][0])])
            
        #print(CATEGORIES[int(prediction[0][0])])
        y=y+1
    
end = time.time()

totaltime = end-start
print(totaltime)