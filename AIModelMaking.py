import tensorboard

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
import pickle
from keras.callbacks import TensorBoard
import numpy as np
import time
from config import Config
from DataGenerator import Generator


#Making the AI model from the dataset you've created from trainingProgram.py

NAME ="Back-vs-Front-64x2-{}".format(int(time.time()))
dataloader = Generator(root_dir=r'C:\Users\break\Documents\Python')
       
train_data_path = 'rotationTraining.csv'
test_data_path = 'rotationTesting.csv'
train_samples = dataloader.load_samples(train_data_path)
test_samples = dataloader.load_samples(test_data_path)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
#X = pickle.load(open("X.pickle", "rb"))
#y = pickle.load(open("y.pickle", "rb"))
batch_size = Config.batch_size
validation_generator = dataloader.data_generator(test_samples, batch_size=batch_size)
train_generator = dataloader.data_generator(train_samples, batch_size=batch_size)
num_train_samples = len(train_samples)
num_test_samples = len(test_samples)
#X=np.array(X)
#X = X/255
input_shape = (Config.resize,Config.resize,3)
model = Sequential()
#model.add(InputLayer(input_shape=(400,400)))
model.add(Conv2D(64, (3,3), input_shape = input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
class_weight = {0: 1., 1: 318}
model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=Config.num_epochs,
        validation_data=validation_generator,
        validation_steps=num_test_samples // batch_size,
        callbacks=[tensorboard], 
        class_weight=class_weight)
#Saving model

model.save('Gen4.1-1000x1000Rotation-{}'.format(time.time())+".model") #Been manually changing "Genx" to go with the current generation my model is at.

