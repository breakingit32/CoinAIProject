from sre_parse import CATEGORIES
import cv2
import tensorflow as tf
import os
import imutils

#Progam to test your AI model from AIModelMaking.py

CATEGORIES = []
#Dir for test images
x = 0
DIR = "C:/Users/break/Documents/Python/CoinPictures3/CoinPictures3_000002.png"
while x < 360:
    CATEGORIES.append(str(x))
    x = x + 4
    


def prepare(filepath):
    IMG_SIZE = 28
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = imutils.rotate(img_array, 12)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("Gen1-DegreeDetection-1668970866.9306254.model")
#running AI on images from DIR then printing results

prediction = model.predict([prepare(DIR)])

img= cv2.imread(DIR)

cv2.imshow('results', img)
print(CATEGORIES[int(prediction[0][0])])
cv2.waitKey(0)
cv2.destroyAllWindows()
