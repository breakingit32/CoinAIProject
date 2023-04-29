
import cv2 
import tensorflow as tf
import os
import random
import argparse
import imutils
import time
import tqdm
DATADIR = "C:/Users/break/Documents/Python/Penny/Rotate Test"
ranNum=random.sample(range(1, 1000), 3)
image = cv2.imread("C:/Users/break/Documents/Python/Penny/Rotate Test/Penny5917483.png")
CATEGORIES = [ "NoUpRight","Upright"]
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
x=0
y=0
model = tf.keras.models.load_model("Gen1-1000x1000Rotation-1659975524.609461.model")
def prepare(filepath):
    IMG_SIZE = 400
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# rotate our image by 45 degrees around the center of the image
start = time.time()


for img in os.listdir(DATADIR):
    y=0
    x=0
    ranNum=random.sample(range(1, 1000), 2)
    image = cv2.imread(os.path.join(DATADIR, img))
    while y < 360:
        
        rotate = imutils.rotate(image, x)
        cv2.imshow("yes", rotate)
        cv2.waitKey(1)
        
        #print(y)
        

        
            
            #Loading model 
            
        
            #Saving location Dir
        saveLocation = "C:/Users/break/Documents/Python/Penny"
        cv2.imwrite(os.path.join(saveLocation, "Test.png"), rotate)
            #Running Model
        print(os.path.join(saveLocation, "Test.png"))
        prediction = model.predict([prepare(os.path.join(saveLocation, "Test.png"))])
            #Saving image depending on model's prediction
        saveLocation = "C:/Users/break/Documents/Python/Penny/RotationTest"
        if CATEGORIES[int(prediction[0][0])] == "Upright":
            rotate=cv2.resize(rotate, (400,400))
            saveLocation = os.path.join(saveLocation, "0")
            print(saveLocation)
            #print("Saving")
                
            cv2.imwrite(os.path.join(saveLocation, "Penny" + str(ranNum[1])+str(y)+".png"), rotate)
            #cv2.destroyAllWindows()
            #print(CATEGORIES[int(prediction[0][0])])
            break
        if CATEGORIES[int(prediction[0][0])] == "NoUpRight":
            x=x+1
            rotate=cv2.resize(rotate, (400,400))
            saveLocation = os.path.join(saveLocation, "Wrong")
                #print(saveLocation)
            #print("Saving")
                
            cv2.imwrite(os.path.join(saveLocation, "Penny" + str(ranNum[0])+str(y)+".png"), rotate)
            #cv2.destroyAllWindows()
            #print(CATEGORIES[int(prediction[0][0])])
            
        #print(CATEGORIES[int(prediction[0][0])])
        y=y+1
    cv2.imshow("", rotate)
end = time.time()

totaltime = end-start
print(totaltime)