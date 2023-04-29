from sre_parse import CATEGORIES
import cv2
import tensorflow as tf
import os
import imutils

#Progam to test your AI model from AIModelMaking.py


#Dir for test images
DIR = "C:/Users/break/Documents/Python/Penny/Test/NoUpRight/Penny5253934.png"
CATEGORIES = ["NoUpRight", "Upright"]

def prepare(filepath):
    IMG_SIZE = 400
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_array = imutils.rotate(img_array, 28)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("Gen1-1000x1000Rotation-1659913132.087626.model")
#running AI on images from DIR then printing results

prediction = model.predict([prepare(DIR)])

img= cv2.imread(DIR)
img = imutils.rotate(img, 27)
cv2.imshow('results', img)
print(CATEGORIES[int(prediction[0][0])])
cv2.waitKey(0)
cv2.destroyAllWindows()