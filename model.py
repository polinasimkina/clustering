import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, MaxPool2D

def centerfinderbuild(drop=0.3):
    '''
    Defines convolutional layers of the network.
    '''
    centerfinder = Sequential()
    centerfinder.add(Conv2D(32, kernel_size=3, input_shape=(171, 360, 1), activation='relu'))  # padding='same'
    centerfinder.add(Conv2D(32, kernel_size=3, activation='relu'))
    centerfinder.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    #centerfinder.add(MaxPool2D(2,2))
    centerfinder.add(Dropout(0.1))
    #centerfinder.add(BatchNormalization())
    
    centerfinder.add(Conv2D(64, kernel_size=3, activation='relu'))
    centerfinder.add(Conv2D(64, kernel_size=3, strides=2, activation='relu'))
    centerfinder.add(Conv2D(64, kernel_size=5, strides=2, activation='relu'))
    #centerfinder.add(MaxPool2D(2,2))
    centerfinder.add(Dropout(0.1))
    #centerfinder.add(BatchNormalization())

    centerfinder.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))
    #centerfinder.add(MaxPool2D(2,2))
    

    centerfinder.add(Flatten())
    centerfinder.add(Dropout(0.1))
    #centerfinder.add(BatchNormalization())
    
    centerfinder.summary()
    return centerfinder

class model(keras.Model):
    '''
    Defines the model.
    '''
    def __init__(self):
        super(model, self).__init__()
        self.centerfinder = centerfinderbuild(0.1) # convolutional layers
        self.dense_en1 = Dense(3200, activation='relu')
        self.dense_en2 = Dense(1600, activation='relu')
        self.dense_en3 = Dense(3*11*11)
        self.reshape_en = Reshape((3,11,11), name='energy')
        
        self.dense1 = Dense(1000, activation='relu')
        self.dense2 = Dense(500, activation='relu')
        self.dense_out = Dense(3 * 2, activation='tanh')
        self.reshape = Reshape((3, 2), name='center')

        self.drop = Dropout(0.3)

    def call(self, inputs):
        x = self.centerfinder(inputs)
        x_en = self.dense_en1(x)
        x_en = self.drop(x_en)
        x_en = self.dense_en2(x_en)
        x_en = self.drop(x_en)
        x_en = self.dense_en3(x_en)
        x_en = self.reshape_en(x_en)
        
        x = self.dense1(x)
        #x = self.drop(x)
        x = self.dense2(x)
        #x = self.drop(x)
        x = self.dense_out(x)
        x = self.reshape(x)
        return x, x_en