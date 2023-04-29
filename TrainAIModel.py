import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# config
img_width, img_height = 20,32 #width & height of input image
input_depth = 1 #1: gray image
train_data_dir = 'C:/Users/break/Documents/Python/Penny/DigitDataSet/Train' #data training path
testing_data_dir = 'C:/Users/break/Documents/Python/Penny/DigitDataSet/Test' #data testing path
epochs = 2 #number of training epoch
batch_size = 5 #training batch size

# define image generator for Keras,
# here, we map pixel intensity to 0-1
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# read image batch by batch
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',#inpput iameg: gray
    target_size=(img_width,img_height),#input image size
    batch_size=batch_size,#batch size
    class_mode='categorical')#categorical: one-hot encoding format class label
testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
    color_mode='grayscale',
    target_size=(img_width,img_height),
    batch_size=batch_size,
    class_mode='categorical')

# define number of filters and nodes in the fully connected layer
NUMB_FILTER_L1 = 20
NUMB_FILTER_L2 = 20
NUMB_FILTER_L3 = 20
NUMB_NODE_FC_LAYER = 10

#define input image order shape
if K.image_data_format() == 'channels_first':
    input_shape_val = (input_depth, img_width, img_height)
else:
    input_shape_val = (img_width, img_height, input_depth)

#define the network
model = Sequential()

# Layer 1
model.add(Conv2D(NUMB_FILTER_L1, (5, 5), 
                 input_shape=input_shape_val, 
                 padding='same', name='input_tensor'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 2
model.add(Conv2D(NUMB_FILTER_L2, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool2D((2, 2)))

# Layer 3
model.add(Conv2D(NUMB_FILTER_L3, (5, 5), padding='same'))
model.add(Activation('relu'))

# flattening the model for fully connected layer
model.add(Flatten())

# fully connected layer
model.add(Dense(NUMB_NODE_FC_LAYER, activation='relu'))

# output layer
model.add(Dense(train_generator.num_classes, 
                activation='softmax', name='output_tensor'))

# Compilile the network
model.compile(loss='categorical_crossentropy',
              optimizer='sgd', metrics=['accuracy'])

# Show the model summary
model.summary()

model.fit_generator(
    train_generator,#our training generator
    #number of iteration per epoch = number of data / batch size
    steps_per_epoch=np.floor(train_generator.n/batch_size),
    epochs=epochs,#number of epoch
    validation_data=testing_generator,#our validation generator
    #number of iteration per epoch = number of data / batch size
    validation_steps=np.floor(testing_generator.n / batch_size))
import time
print("Training is done!")
model.save('Gen1-DigitReader-{}'.format(time.time())+".model")
print("Model is successfully stored!")