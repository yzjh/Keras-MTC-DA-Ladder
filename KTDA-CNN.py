from tensorflow.python.keras.backend import batch_dot
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)
KTF.set_session(sess)


from keras.datasets import mnist
from keras.layers import *
from keras import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
import keras
from keras.models import load_model, Sequential
import tensorflow as tf
import random
from keras import backend as K

import os
import os.path
import struct
import gzip
import numpy as np
from model.mynet import mynet
from model.VGG16 import VGG
from model.resnet50 import ResNet
from model.alexnet import alexnet
from model.ladder_net import get_ladder_network_fc


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def prepare_data(data_path, val_rate=0.2, trans_rate=0.8):
    base_path = '/data1/ningjh/project/transfer0321/mnist'
    data_train, label_train = load_mnist(base_path + data_path)
    data_test, label_test = load_mnist(base_path + data_path, kind='t10k')
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=val_rate,
                                                                    random_state=0)

    if trans_rate == 1.0:
        data_trans = data_train
        label_trans = label_train
    else:
        _, data_trans, _, label_trans = train_test_split(data_train, label_train, test_size=trans_rate, random_state=0)

    # label_train = keras.utils.to_categorical(label_train, num_classes=20)
    label_val = keras.utils.to_categorical(label_val, num_classes=20)
    label_test = keras.utils.to_categorical(label_test, num_classes=20)
    label_trans = keras.utils.to_categorical(label_trans, num_classes=20)

    # data_train = (data_train) / 255
    data_trans = (data_trans) / 255
    data_test = (data_test) / 255
    data_val = (data_val) / 255
    data_val = data_val.reshape(data_val.shape[0], 28, 28, 1)
    # data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
    data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
    data_trans = data_trans.reshape(data_trans.shape[0], 28, 28, 1)

    return data_trans, label_trans, data_val, label_val, data_test, label_test


def mk_mmd_loss(f_src, f_tar,
                kernel_mul=2,
                kernel_num=5,
                fix_sigma=None):

    batch_size = tf.shape(f_src)[0]
    total = tf.concat([f_src, f_tar], axis=0)
    total_n = tf.shape(total)[0]
    total0 = tf.tile(tf.expand_dims(total, axis=0), [total_n, 1, 1])
    total1 = tf.tile(tf.expand_dims(total, axis=1), [1, total_n, 1])
    L2_distance = tf.reduce_sum((total0 - total1) ** 2, axis=2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / tf.cast(batch_size ** 2 - batch_size, tf.float32)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    sigma_list = [bandwidth * (kernel_num ** i) for i in range(kernel_num)]
    kernels_val = [tf.exp(-L2_distance / sigma) for sigma in sigma_list]
    kernels = sum(kernels_val)  # / len(kernels_val)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = tf.reduce_mean(XX + YY - XY - YX)

    return loss


lmd = 0.25
epochs_train = 100
epochs_trans = 200
batch_size = 128
model_name = 'model_train'
# 在这里更换模型
# base_model = ResNet()
base_model = VGG()
# base_model = mynet()
# base_model = alexnet()

classify_1 = Sequential(
    [
        Flatten(),
        # Dense(512),
        # Activation('relu', name='ca11'),
        # Dropout(0.5),
        # Dense(256),
        # Activation('relu', name='ca12'),
        # Dropout(0.5),
        # Dense(128),
        # Activation('relu', name='ca13'),
        # Dropout(0.5),
    ]
)
classify_2 = Sequential(
    [
        Flatten(),
        # Dense(512),
        # Activation('relu', name='ca21'),
        # Dropout(0.5),
        # Dense(256),
        # Activation('relu', name='ca22'),
        # Dropout(0.5),
        # Dense(128),
        # Activation('relu', name='ca23'),
        # Dropout(0.5),
    ]
)

inp_src = Input(shape=(28, 28, 1))
inp_tar = Input(shape=(28, 28, 1))

x_train = base_model(inp_src)
x_train = classify_1(x_train)
y_train = Dense(20, activation='softmax')(x_train)
model_train = Model(inp_src, y_train)
model_train.summary()
model_train.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

x_trans = base_model(inp_tar)
x_trans_src = classify_1(x_trans)
x_trans_tar = classify_2(x_trans)
y_trans = Dense(20, activation='softmax')(x_trans_tar)
model_trans = Model(inp_tar, y_trans)
mmd_loss = mk_mmd_loss(x_trans_src, x_trans_tar)
model_trans.add_loss(mmd_loss * lmd)

# # first dense layer mmd loss
# f_src_1 = classify_1.get_layer('ca11').output
# f_tar_1 = classify_2.get_layer('ca21').output
# mmd_loss1 = mk_mmd_loss(f_src_1, f_tar_1, batch_size)
# model_trans.add_loss(mmd_loss1 * lmd)
# model_trans.add_metric(mmd_loss1, 'mmd_loss')
# # second dense layer mmd loss
# f_src_2 = classify_1.get_layer('ca12').output
# f_tar_2 = classify_2.get_layer('ca22').output
# mmd_loss2 = mk_mmd_loss(f_src_2, f_tar_2, batch_size)
# model_trans.add_loss(mmd_loss2 * lmd)
# model_trans.add_metric(mmd_loss2, 'mmd_loss')
# # third dense layer mmd loss
# f_src_3 = classify_1.get_layer('ca13').output
# f_tar_3 = classify_2.get_layer('ca23').output
# mmd_loss3 = mk_mmd_loss(f_src_3, f_tar_3, batch_size)
# model_trans.add_loss(mmd_loss3 * lmd)
# model_trans.add_metric(mmd_loss3, 'mmd_loss')

model_trans.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#model_trans.add_metric(mmd_loss, 'mmd_loss')

data_src_train, label_src_train, data_src_val, label_src_val, data_src_test, label_src_test \
    = prepare_data('/pre_train', val_rate=0.2, trans_rate=1)
data_tar_train, label_tar_train, data_tar_val, label_tar_val, data_tar_test, label_tar_test \
    = prepare_data('/trans', val_rate=0.2, trans_rate=0.04)

checkpoint = ModelCheckpoint(model_name + str(batch_size) + ".h5",
                             verbose=1,
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True)
tensorboard = TensorBoard(model_name + str(batch_size) + ".log", 0)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=10, mode='auto',
#                                                 verbose=0, min_delta=3e-6, cooldown=0, min_lr=0)

#first stage
history_1 = model_train.fit(data_src_train,
                            label_src_train,
                            batch_size=batch_size,
                            epochs=epochs_train,
                            validation_data=(data_src_val, label_src_val),
                            callbacks=[checkpoint, tensorboard])
#model_train.load_weights('model_train128.h5')
score_src = model_train.evaluate(data_src_test, label_src_test, batch_size=batch_size, verbose=0)
print('Total loss on src Test Set:', score_src[0])
print('Accuracy of src Test Set:', score_src[1])

base_model.trainable = False
classify_1.trainable = False

# second stage
history_2 = model_trans.fit(data_tar_train,
                            label_tar_train,
                            batch_size=batch_size,
                            epochs=epochs_trans,
                            validation_data=(data_tar_val, label_tar_val),
                            callbacks=[tensorboard])
score_tar = model_trans.evaluate(data_tar_test, label_tar_test, batch_size=batch_size, verbose=0)
print('Total loss on tar Test Set:', score_tar[0])
print('Accuracy of tar Test Set:', score_tar[1])
