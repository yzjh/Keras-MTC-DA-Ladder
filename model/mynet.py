from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Activation,Dense,Dropout,BatchNormalization

def mynet(input_shape=(28, 28, 1)):

    model=Sequential()

    model.add(Conv2D(32, (5,5), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))   
    model.add(MaxPool2D((2,2)))

    model.add(Conv2D(64, (5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPool2D((2,2)))

    model.summary()

    return model
