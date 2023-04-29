from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, MaxPooling2D, Dropout
import time
from config import Config
from DataGenerator import Generator
from keras.callbacks import TensorBoard

def load_model(pretrained_weights=None):
    
    """
    the method to load and returns pretrained or new model
    
    Args:
        pretrained_weights-pretrained weights file
        
    Returns:
        model - the loaded keras model
    """
    
    
    input_shape = (Config.resize,Config.resize,3)
#    print (input_shape)
    model = Sequential()
    
    #filters,kernel_size,strides=(1, 1),padding='valid',data_format=None,dilation_rate=(1, 1),activation=None,use_bias=True,
    #kernel_initializer='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,
    #activity_regularizer=None,kernel_constraint=None,bias_constraint=None,
    
    #pool_size=(2, 2), strides=None, padding='valid',data_format=None
    
    model.add(Conv2D(64, (3,3),padding='same',input_shape=input_shape,name='conv2d_1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_1'))
    
    model.add(Conv2D(64, (3, 3),name='conv2d_2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2d_2'))
    
    #model.add(Dropout(0.5))
    
    #model.add(Convolution2D(64, 3, 3))
    #model.add(Activation('relu'))
    #model.add(Convolution2D(64, 3, 3))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

if __name__=='__main__':
    
    #call the dataloader and create traina nd test dataloader object
    NAME ="Back-vs-Front-64x2-{}".format(int(time.time()))

    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    dataloader = Generator(root_dir=r'C:\Users\break\Documents\Python')
       
    train_data_path = 'rotationTraining.csv'
    test_data_path = 'rotationTesting.csv'
    
    train_samples = dataloader.load_samples(train_data_path)
    test_samples = dataloader.load_samples(test_data_path)
    
    num_train_samples = len(train_samples)
    num_test_samples = len(test_samples)
    
    print ('number of train samples: ', num_train_samples)
    print ('number of test samples: ', num_test_samples) 
       
    # Create generator
    batch_size = Config.batch_size
    train_generator = dataloader.data_generator(train_samples, batch_size=batch_size)
    validation_generator = dataloader.data_generator(test_samples, batch_size=batch_size)
       
    
    model = load_model()
    model.summary()
    
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    class_weight = {0: 318, 1: 1}
    hist=model.fit(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=Config.num_epochs,
        validation_data=validation_generator,
        validation_steps=num_test_samples // batch_size,
        callbacks=[tensorboard], 
        class_weight=class_weight)
    model.save('Gen4.1-1000x1000Rotation-{}'.format(time.time())+".model")