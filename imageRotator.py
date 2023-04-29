import numpy as np
import matplotlib.pyplot as plt
import os
import imutils
import cv2
from tqdm import tqdm
import random
import time
DATADIR = "C:/Users/break/Documents/Python/Penny/TestGen3/Upright"
SaveLocation = "C:/Users/break/Documents/Python/Penny/Rotation/"
CATEGORIES = ["NoUpRight", "Upright"]

IMG_SIZE = 28
x = 0
count = os.listdir(DATADIR)
print(count)
for file in os.listdir(DATADIR):  
    image = os.path.join(DATADIR, file)
    x = 0
    while x < 360:
        CorrectFile = os.path.join(SaveLocation, str(x))
        print(SaveLocation)
        img_array = cv2.imread(image)  # convert to array
        rotate = imutils.rotate(img_array, x)
        cv2.imshow("t", rotate)
        saveFileName = os.path.join(CorrectFile, str(time.time())  + "_" + str(x) + ".png")
        cv2.imwrite(saveFileName, rotate)
        x = x + 4
