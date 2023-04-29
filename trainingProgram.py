import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random

#This program takes the resulting saved images from circleDetection.py or circleDetectionWithAI.py
#Then makes a dataset that can be used to train your AI

DATADIR = "C:/Users/break/Documents/Python/Penny/Rotation 28x28 Short"

CATEGORIES = {'0':0,'180':1,'276':2}
'''
{'0':0, '4':1, '8':2, '12':3, '16':4, '20':5, '24':6, 
'28':7, '32':8, '36':9, '40':10, '44':11, '48':12, '52':13, 
'56':14, '60':15, '64':16, '68':17, '72':18, '76':19, '80':20,
'84':21, '88':22, '92':23, '96':24, '100':25, '104':26, '108':27,
'112':28, '116':29, '120':30, '124':31, '128':32, '132':33, '136':34,
'140':35, '144':36, '148':37, '152':38, '156':39, '160':40, '164':41,
'168':42, '172':43, '176':44, '180':45, '184':46, '188':47, '192':48,
'196':49, '200':50, '204':51, '208':52, '212':53, '216':54, '220':55,
'224':56, '228':57, '232':58, '236':59, '240':60, '244':61, '248':62,
'252':63, '256':64, '260':65, '264':66, '268':67, '272':68, '276':69, 
'280':70, '284':71, '288':72, '292':73, '296':74, '300':75, '304':76, 
'308':77, '312':78, '316':79, '320':80, '324':81, '328':82, '332':83,
'336':84, '340':85, '344':86, '348':87, '352':88, '356':89}
'''
training_data = []
IMG_SIZE = 28

def create_training_data():
    for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES[category]  # get the classification  (0 or a 1). 0=dog 1=cat
        print(class_num)
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()
print(len(training_data))
#Randomizing your dataset
random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)



X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle
#Saving your dataset and labels
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out, protocol=pickle.HIGHEST_PROTOCOL)
pickle_out.close()
