from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Activation,Dense,BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from sklearn.metrics import recall_score,f1_score,precision_score

def VGG(input_shape=(28, 28, 1)):

    model=Sequential()
    #layer_1
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=input_shape,padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',data_format='channels_last',kernel_initializer='uniform',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))

    #layer_2
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(128,(2,2),strides=(1,1),padding='same',data_format='channels_last',activation='relu',kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))

    #layer_3
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))
    #layer_4
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2)))

    #layer_5
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',data_format='channels_last',activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1,1), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(BatchNormalization())
    # # model.add(MaxPooling2D((2,2)))

    model.summary()

    return model
