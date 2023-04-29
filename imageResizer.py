import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
DATADIR = "C:/Users/break/Documents/Python/Penny/Rotation 28x28"
x = 0 
CATEGORIES = []                    #["NoUpRight", "Upright"]
while x < 360:
    CATEGORIES.append(x)
    x = x + 4
training_data = []
IMG_SIZE = 28

for category in CATEGORIES:  

    path = os.path.join(DATADIR,str(category))  # create path to dogs and cats
        # get the classification  (0 or a 1). 0=dog 1=cat

    for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
        
        img_array = cv2.imread(os.path.join(path,img) , 3)  # convert to array
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(path, img), img_array)
