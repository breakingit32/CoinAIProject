from sre_parse import CATEGORIES
import cv2
import tensorflow as tf
import os
#Progam to test your AI model from AIModelMaking.py


#Dir for test images
DIR = "C:/Users/break/Documents/Python/Penny/Up"
CATEGORIES = ["NoUpRight", "Upright"]

def prepare(filepath):
    IMG_SIZE = 400
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("Gen1-1000x1000Rotation-1659913132.087626.model")
#running AI on images from DIR then printing results
for filename in os.listdir(DIR):
    prediction = model.predict([prepare(str(os.path.join(DIR, filename)))])
    print(str(filename))
    img= cv2.imread(os.path.join(DIR,filename))
    cv2.imshow('results', img)
    print(CATEGORIES[int(prediction[0][0])])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    


