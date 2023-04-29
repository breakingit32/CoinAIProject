
import cv2 
import tensorflow as tf
import os
import random
import argparse
import imutils
import time
import tqdm
DATADIR = "C:/Users/break/Documents/Python/CoinPictures3"
ranNum=random.sample(range(1, 1000), 3)
SaveLocation = "C:/Users/break/Documents/Python/Penny/RotationTestShort"

x=0
y=0
CATEGORIES = ["0", "180", "276"]
#Dir for test images

x = 0
'''
while x < 360:
    CATEGORIES.append(str(x))
    x = x + 4
'''
model = tf.keras.models.load_model("Gen1-DegreeDetection-1669008707.5952852.model")
def prepare(filepath):
    IMG_SIZE = 28
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
    for y in CATEGORIES:
        
        rotate = imutils.rotate(image, 180)
        cv2.imshow("yes", rotate)
        cv2.waitKey(10)
        
        cv2.imwrite(os.path.join(DATADIR, "Test.png"), rotate)
            #Running Model
        prediction = model.predict([prepare(os.path.join(DATADIR, "Test.png"))])
        save = os.path.join(SaveLocation, str(CATEGORIES[int(prediction[0][0])]))
        cv2.imwrite(os.path.join(save, "Rotation"+ str(time.time()) + "_Degree_" + str(CATEGORIES[int(prediction[0][0])]) + ".png"), rotate)
        print(CATEGORIES[int(prediction[0][0])])
           
        
        
end = time.time()

totaltime = end-start
print(totaltime)
