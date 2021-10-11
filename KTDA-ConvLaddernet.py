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


def label_split(data_train, label_train, sample_rate=0.2):

    sample_num = int(data_train.shape[0] * sample_rate)
    random.seed(0)
    idxs_annot = np.random.choice(data_train.shape[0], sample_num)

    x_train_unlabeled = data_train
    x_train_labeled = data_train[idxs_annot]
    y_train_labeled = label_train[idxs_annot]

    n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
    x_train_labeled = np.concatenate([x_train_labeled] * n_rep)
    y_train_labeled = np.concatenate([y_train_labeled] * n_rep)

#    rmd = 1.0 % sample_rate
#    if rmd is not 0:
#        add_num = int(data_train.shape[0] * rmd)
#        x_temp = x_train_labeled[:add_num, :]
#        y_temp = y_train_labeled[:add_num]
#        x_train_labeled = np.append(x_train_labeled, x_temp, axis=0)
#        y_train_labeled = np.append(y_train_labeled, y_temp, axis=0)

    if x_train_labeled.shape[0] < x_train_unlabeled.shape[0]:
        x_train_unlabeled = x_train_unlabeled[:int(x_train_labeled.shape[0])]

    return x_train_labeled, x_train_unlabeled, y_train_labeled


def prepare_data(data_path, val_rate=0.2, trans_rate=0.8, is_semi=False):

    base_path = '/data1/ningjh/project/transfer0321/mnist'
    data_train, label_train = load_mnist(base_path + data_path)
    data_test, label_test = load_mnist(base_path + data_path, kind='t10k')
    # 这个是取验证集（val）的比例，改test_size即可
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=val_rate,
                                                                   random_state=0)
    data_trans_u = None
    data_val_u = None
    if trans_rate == 1.0:
        data_trans = data_train
        label_trans = label_train
    else:
        if is_semi:
            data_trans, data_trans_u, label_trans = label_split(data_train, label_train, sample_rate=trans_rate)
            data_val, data_val_u, label_val = label_split(data_val, label_val, sample_rate=trans_rate)
        else:
            _, data_trans, _, label_trans = train_test_split(data_train, label_train, test_size=trans_rate,
                                                             random_state=0)

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
    if data_trans_u is not None:
        data_trans_u = data_trans_u / 255
        data_trans_u = data_trans_u.reshape(data_trans_u.shape[0], 28, 28, 1)
        data_val_u = data_val_u / 255
        data_val_u = data_val_u.reshape(data_val_u.shape[0], 28, 28, 1)

    return data_trans, label_trans, data_val, label_val, data_test, label_test, data_trans_u, data_val_u


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


lmd = 0.5
epochs_train = 100
epochs_trans = 200
batch_size = 128
model_name1 = 'model_train'
model_name2 = 'model_new_train'

# 在这里更换模型
base_model = ResNet()
# base_model = VGG()
# base_model = mynet()
# base_model = alexnet()

Inp1 = Input(shape=(28, 28, 1))
Inp2 = Input(shape=(28, 28, 1))
Inp_l = Input(shape=(28, 28, 1))
Inp_u = Input(shape=(28, 28, 1))

x = base_model(Inp1)
x = Flatten()(x)
r1 = Dense(20, activation='softmax')(x)
model_train = Model(inputs=Inp1, outputs=r1)
model_train.compile(loss="categorical_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])

# 在这里更换模型
new_base_model = ResNet()
# new_base_model = VGG()
# new_base_model = mynet()
# new_base_model = alexnet()

x1 = new_base_model(Inp2)
f1 = base_model(Inp2)
x1 = Flatten()(x1)
f1 = Flatten()(f1)
mmd_loss = mk_mmd_loss(x1, f1)
r2 = Dense(20, activation='softmax')(x1)
model_new_train = Model(inputs=Inp2, outputs=r2)
model_new_train.add_loss(mmd_loss * lmd)
model_new_train.compile(loss="categorical_crossentropy",
                    optimizer="sgd",
                    metrics=["accuracy"])


y_l = base_model(Inp_l)
y_u = base_model(Inp_u)
y_l = Flatten()(y_l)
y_u = Flatten()(y_u)
f_l = Model(Inp_l, y_l)
f_u = Model(Inp_u, y_u)
y_c_l, y_n_l, u_cost = get_ladder_network_fc(inputs_l=f_l.output, inputs_u=f_u.output, layer_sizes=[f_l.output_shape[1], 1000, 500, 250, 250, 250, 20])
model_trans = Model([f_l.input, f_u.input], y_c_l)
model_trans.add_loss(u_cost)
#model_trans.add_metric(u_cost, name="den_loss")
te_m = Model(f_l.input, y_n_l)
model_trans.test_model = te_m

data_src_train, label_src_train, data_src_val, label_src_val, data_src_test, label_src_test, _, _ \
    = prepare_data('/pre_train', val_rate=0.2, trans_rate=1)
data_tar_train, label_tar_train, data_tar_val, label_tar_val, data_tar_test, label_tar_test,\
    data_tar_train_u, data_tar_val_u = prepare_data('/trans', val_rate=0.2, trans_rate=0.1, is_semi=True)

checkpoint1 = ModelCheckpoint(model_name1 + str(batch_size) + ".h5",
                             verbose=1,
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True)
checkpoint2 = ModelCheckpoint(model_name2 + str(batch_size) + ".h5",
                             verbose=1,
                             monitor='val_loss',
                             save_best_only=True,
                             save_weights_only=True)
tensorboard = TensorBoard(model_name1 + str(batch_size) + ".log", 0)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=10, mode='auto',
#                                                 verbose=0, min_delta=3e-6, cooldown=0, min_lr=0)

# first stage
history_1 = model_train.fit(data_src_train,
                            label_src_train,
                            batch_size=batch_size,
                            epochs=epochs_train,
                            validation_data=(data_src_val, label_src_val),
                            callbacks=[checkpoint1, tensorboard])
# model_train.load_weights('model_train128.h5')
score_src = model_train.evaluate(data_src_test, label_src_test, batch_size=batch_size, verbose=0)
print('Total loss on src Test Set:', score_src[0])
print('Accuracy of src Test Set:', score_src[1])

base_model.trainable = False
model_trans.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(lr=0.02),
              metrics=["accuracy"])

model_trans.test_model.compile(loss="categorical_crossentropy",
                         optimizer=keras.optimizers.Adam(lr=0.02),
                         metrics=["accuracy"])


# second stage
history_2 = model_train.fit(data_tar_train,
                            label_tar_train,
                            batch_size=batch_size,
                            epochs=epochs_train,
                            validation_data=(data_tar_val, label_tar_val),
                            callbacks=[checkpoint2, tensorboard])
score_new = model_train.evaluate(data_tar_test, label_tar_test, batch_size=batch_size, verbose=0)
print('Total loss on new Test Set:', score_new[0])
print('Accuracy of new Test Set:', score_new[1])

new_base_model.trainable = False

# Third stage
history_3 = model_trans.fit([data_tar_train, data_tar_train_u],
                            label_tar_train,
                            batch_size=batch_size,
                            epochs=epochs_trans,
                            validation_data=([data_tar_val, data_tar_val_u], label_tar_val),
                            callbacks=[tensorboard])
score_tar = model_trans.test_model.evaluate(data_tar_test, label_tar_test, batch_size=batch_size, verbose=0)
print('Total loss on tar Test Set:', score_tar[0])
print('Accuracy of tar Test Set:', score_tar[1])
