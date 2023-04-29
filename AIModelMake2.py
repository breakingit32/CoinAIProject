import numpy as np
import time
import logging
from config import Config
from DataGenerator import Generator
import tensorboard
from tensorflow.keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras import backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer, AveragePooling2D
import pickle
import os

errorLog = "C:/Users/break/Documents/Python/Penny/ErrorLogs"
def build(numChannels, imgRows, imgCols, numClasses,
		activation="relu", weightsPath=None):
		# initialize the model
		model = Sequential()
		inputShape = (imgRows, imgCols, numChannels)
		# if we are using "channels first", update the input shape
		if K.image_data_format() == "channels_first":
			inputShape = (numChannels, imgRows, imgCols)
		# define the first set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(20, 5, padding="same",
			input_shape=inputShape))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        		# define the second set of CONV => ACTIVATION => POOL layers
		model.add(Conv2D(50, 5, padding="same"))
		model.add(Activation(activation))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        		# define the first FC => ACTIVATION layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation(activation))
		# define the second FC layer
		model.add(Dense(numClasses))
		# lastly, define the soft-max classifier
		model.add(Activation("softmax"))
        		# if a weights path is supplied (inicating that the model was
		# pre-trained), then load the weights
		if weightsPath is not None:
			model.load_weights(weightsPath)
		# return the constructed network architecture
		return model
#Making the AI model from the dataset you've created from trainingProgram.py
try:
    NAME ="Back-vs-Front-64x2-{}".format(int(time.time()))
    dataloader = Generator(root_dir=r'C:/Users/break/Documents/Python')
        
    #train_data_path = 'rotationTraining.csv'
    #test_data_path = 'rotationTesting.csv'
    #train_samples = dataloader.load_samples(train_data_path)
    #test_samples = dataloader.load_samples(test_data_path)
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))
    X=np.array(X)
    y = np.array(y)
    X = X/255
    batch_size = Config.batch_size
    #validation_generator = dataloader.data_generator(test_samples, batch_size=batch_size)
    #train_generator = dataloader.data_generator(train_samples, batch_size=batch_size)
    #num_train_samples = len(train_samples)
    #num_test_samples = len(test_samples)
    input_shape = (Config.resize,Config.resize,1)
    '''
    model = build(numChannels=1, imgRows=28, imgCols=28,
        numClasses=90)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"])

    model.fit(X, y, batch_size=32, epochs=20, validation_split=0.1,
            verbose=1)
    model.save('Gen1-DegreeDetection-{}'.format(time.time())+".model") #Been manually changing "Genx" to go with the current generation my model is at.
    model.summary()
    '''
    
    CNN = Sequential()

    CNN.add(Conv2D(6, (5,5), activation = 'relu', input_shape = (28,28,1)))
    CNN.add(AveragePooling2D())

    #CNN.add(layers.Dropout(0.2))

    CNN.add(Conv2D(16, (5,5), activation = 'relu'))
    CNN.add(AveragePooling2D())

    CNN.add(Flatten())

    CNN.add(Dense(120, activation = 'relu'))

    CNN.add(Dense(84, activation = 'relu'))

    CNN.add(Dense(43, activation = 'softmax'))
    CNN.summary()
    CNN.compile(optimizer = 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
        # show the accuracy on the testing set
    history = CNN.fit(X,
                 y,
                 batch_size = 500,
                 epochs = 500,
                 verbose = 1,
                 validation_split=0.1)
    CNN.save('Gen1-DegreeDetection-{}'.format(time.time())+".model") #Been manually changing "Genx" to go with the current generation my model is at.
except Exception as Argument:
    f = open(os.path.join(errorLog, str(time.time()) + "error.txt"), "a")
    f.write(str(Argument))
    f.close()

