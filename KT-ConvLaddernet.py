from keras.datasets import mnist
from mlxtend.data import loadlocal_mnist 
from keras.layers import *
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import keras
from keras.models import load_model, Sequential, Model
import tensorflow as tf
import random
import os
import os.path
import struct
import gzip
import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Conv2D,MaxPool2D,Flatten,Activation,Dense,Dropout,BatchNormalization


from model.mynet import mynet
from model.VGG16 import VGG
from model.resnet50 import ResNet
from model.alexnet import alexnet
from model.ladder_net import get_ladder_network_fc
import itertools

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


def label_split(data_train, label_train, sample_rate = 0.2):
    
    sample_num = int(data_train.shape[0] * sample_rate)
    random.seed(0)
    idxs_annot = np.random.choice(data_train.shape[0], sample_num)

    x_train_unlabeled = data_train
    x_train_labeled = data_train[idxs_annot]
    y_train_labeled = label_train[idxs_annot]

    n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
    x_train_labeled = np.concatenate([x_train_labeled]*n_rep)
    y_train_labeled = np.concatenate([y_train_labeled]*n_rep)

#    rmd = 1.0 % sample_rate
#    if rmd is not 0:
#        add_num = (int)(data_train.shape[0] * rmd)
#        x_temp = x_train_labeled[:add_num, :]
#        y_temp = y_train_labeled[:add_num]
#        x_train_labeled = np.append(x_train_labeled, x_temp, axis=0)
#        y_train_labeled = np.append(y_train_labeled, y_temp, axis=0)


    x_train_unlabeled = x_train_unlabeled.reshape(x_train_unlabeled.shape[0], 28, 28, 1)
    x_train_labeled = x_train_labeled.reshape(x_train_labeled.shape[0], 28, 28, 1)

    return x_train_labeled, x_train_unlabeled, y_train_labeled


def prepare_data(data_path, val_rate=0.2, trans_rate=1.0):

    base_path = '/data1/ningjh/project/transfer0321/'
    data_train, label_train = load_mnist(base_path + data_path)
    data_test, label_test = load_mnist(base_path + data_path, kind='t10k')
    # 这个是取验证集（val）的比例，改test_size即可
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=val_rate, random_state=0)
    
    if trans_rate == 1.0:
        data_trans = data_train
        label_trans = label_train
    else:
        _, data_trans, _, label_trans = train_test_split(data_train, label_train, test_size=trans_rate, random_state=0)

    # def classify(labels):
    #     l = [0 for i in range(20)]
    #     for i in labels:
    #         l[i] = l[i] + 1
    #     print(l)
    # classify(label_train)
    # classify(label_val)
    # classify(label_test)

    #label_train = keras.utils.to_categorical(label_train, num_classes=20)
    label_val = keras.utils.to_categorical(label_val, num_classes=20)
    label_test = keras.utils.to_categorical(label_test, num_classes=20)
    label_trans = keras.utils.to_categorical(label_trans, num_classes=20)

    #data_train = (data_train) / 255
    data_trans = (data_trans) / 255
    data_test = (data_test) / 255
    data_val = (data_val) / 255
    data_val = data_val.reshape(data_val.shape[0], 28, 28, 1)
    #data_train = data_train.reshape(data_train.shape[0], 28, 28, 1)
    data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)
    data_trans = data_trans.reshape(data_trans.shape[0], 28, 28, 1)
    # data_val = np.resize(data_val, (data_val.shape[0], 224, 224, 3))
    # #data_train = np.resize(data_train, (data_train.shape[0], 32, 32, 3))
    # data_test = np.resize(data_test, (data_test.shape[0], 224, 224, 3))
    # data_trans = np.resize(data_trans, (data_trans.shape[0], 224, 224, 3))
    
    return data_trans, label_trans, data_val, label_val, data_test, label_test


def train_model(model, 
                data_train, label_train, 
                data_val, label_val, 
                data_test, label_test,
                model_name,
                epochs=100,
                batch_size=128):

    if model_name == 'model_trans_ladder':
        model.compile(loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.02),
            metrics=["accuracy"])
        model.test_model.compile(loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(lr=0.02),
            metrics=["accuracy"])
        model.metrics_names.append("den_loss")
        model.metrics_tensors.append(u_cost)
        pass
    else:
        model.compile(loss="categorical_crossentropy",
            optimizer="sgd",
            metrics=["accuracy"])
    model.summary()

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6, patience=10, mode='auto',
                                                  verbose=0, min_delta=3e-6, cooldown=0, min_lr=0)

    checkpoint = ModelCheckpoint(model_name + str(batch_size) + ".hdf5",
                                verbose=1,
                                monitor='val_loss',
                                save_best_only=True)
    tensorboard = TensorBoard(model_name + str(batch_size) + ".log", 0)
    
    # 在这改epoch数
    history = model.fit(data_train,label_train,batch_size=batch_size,epochs=epochs,validation_data=(data_val,label_val),
                callbacks=[checkpoint, tensorboard, reduce_lr])
    if model_name == 'model_trans_ladder':
        score = model.test_model.evaluate(data_test,label_test, batch_size=batch_size, verbose=0)
    else:
        score = model.evaluate(data_test,label_test, batch_size=batch_size, verbose=0)
    print('Total loss on Test Set:', score[0])
    print('Accuracy of Testing Set:', score[1])

    return history 


def test_model(model, data_test, label_test, batch_size=128):
    Y_pred = model.predict(data_test, batch_size=batch_size, verbose=0)
    def oh2ori(Y_pred):
        max = np.max(Y_pred, axis=1)
        y_pred = np.zeros(shape=(Y_pred.shape[0], 1))
        for i in range(Y_pred.shape[0]):
            for j in range(20):
                if (Y_pred[i][j] == max[i]):
                    y_pred[i] = j
        return y_pred
    y_pred = oh2ori(Y_pred)
    label_test = oh2ori(label_test)
    # y_pred = model.predict(data_test, batch_size=100)
    ans = metrics.classification_report(label_test, y_pred, digits=5)
    print(ans)


def draw_plot(history):
    acc = history.history['acc']    
    val_acc = history.history['val_acc']    
    loss = history.history['loss']      
    val_loss = history.history['val_loss']     

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# 在这里更换模型
# base_model = ResNet()
base_model = VGG()
# base_model = mynet()
# base_model = alexnet()

Inp = Input((28,28,1))
Inp_l = Input((28,28,1))
Inp_u = Input((28,28,1))

# 分类1
x = base_model(Inp)
x = BatchNormalization()(x)
x = Flatten()(x)
r1 = Dense(20, activation='softmax')(x)
model_train = Model(inputs=Inp, outputs=r1)

# 分类2
y_l = base_model(Inp_l)
y_u = base_model(Inp_u)
y_l = BatchNormalization()(y_l)
y_u = BatchNormalization()(y_u)
y_l = Flatten()(y_l)
y_u = Flatten()(y_u)
y_l = Model(Inp_l, y_l)
y_u = Model(Inp_u, y_u)
# y = Model([y_l.input, y_u.input], [y_l.output, y_u.output])
y_c_l, y_n_l, u_cost = get_ladder_network_fc(inputs_l=y_l.output, inputs_u=y_u.output, layer_sizes=[y_l.output_shape[1], 1000, 500, 250, 250, 250, 20])
model_trans = Model([y_l.input, y_u.input], y_c_l)
model_trans.add_loss(u_cost)
# model_trans.compile(keras.optimizers.Adam(lr=0.02), 'categorical_crossentropy', metrics=['accuracy'])

# model_trans.metrics_names.append("den_loss")
# model_trans.metrics_tensors.append(u_cost)

te_m = Model(y_l.input, y_n_l)
model_trans.test_model = te_m
# model_trans.test_model.compile(keras.optimizers.Adam(lr=0.02), 'categorical_crossentropy', metrics=['accuracy'])
# r2 = ladder(y.output)
# model_trans = Model(inputs=[y_l.input, y_u.input], outputs=r2)
    
# 训练和测试
batch_size = 64
base_model.trainable = True
data_train, label_train, data_val, label_val, data_test, label_test = \
    prepare_data('/mnist/pre_train', val_rate=0.2, trans_rate=0.8)    # 准备预训练数据
history_train = train_model(model_train,               
                            data_train, label_train, 
                            data_val, label_val, 
                            data_test, label_test,
                            model_name='model_train', epochs=50, batch_size=batch_size)
test_model(model_train, data_test, label_test, batch_size)

data_train, label_train = load_mnist('/data1/ningjh/project/transfer0321' + "/mnist/trans")
data_test, label_test = load_mnist('/data1/ningjh/project/transfer0321' + "/mnist/trans", kind='t10k')

data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, train_size=120000, random_state=0)
data_val = data_val[0:8000, :]
label_val = label_val[0:8000]

label_train = keras.utils.to_categorical(label_train, num_classes=20)
label_val = keras.utils.to_categorical(label_val, num_classes=20)
label_test = keras.utils.to_categorical(label_test, num_classes=20)

data_train = data_train.reshape(-1, 784).astype('float32')/255
data_val = data_val.reshape(-1, 784).astype('float32')/255
data_test  = data_test.reshape(-1, 784).astype('float32')/255

data_test = data_test.reshape(data_test.shape[0], 28, 28, 1)

data_train_l, data_train_u, label_train = label_split(data_train, label_train, 0.04)
data_val_l, data_val_u, label_val = label_split(data_val, label_val, 0.04)

#冻结前一个卷积层
#for l in range(len(base_model.layers)):
#      if l < 5:
#         base_model.layers[l].trainable = False
#      else:
#         base_model.layers[l].trainable = True
#冻结后一个卷积层
#for l in range(len(base_model.layers)):
#     if l < 4:
#         base_model.layers[l].trainable = True
#     else:
#         base_model.layers[l].trainable = False
#全冻
base_model.trainable = False
#全不冻
#base_model.trainable = True

history_trans = train_model(model_trans,               
                            [data_train_l, data_train_u], label_train, 
                            [data_val_l, data_val_u], label_val, 
                            data_test, label_test,
                            model_name='model_trans_ladder', epochs=50, batch_size=batch_size)
test_model(model_trans.test_model, data_test, label_test, batch_size)

# 画图
draw_plot(history_train)
draw_plot(history_trans)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
        dig = np.diag(cm)
        acc = dig.mean()
        acc = format(acc, '.2%')
        print("acc:", acc)
    else:
        # print('Confusion matrix, without normalization')
        cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        dig = np.diag(cm1)
        acc = dig.mean()
        acc = format(acc, '.2%')
        print("acc:", acc)
    # print(cm)

    thresh = cm.max() / 2.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.colorbar()
    labels = ['BitTorrent','Facetime','FTP','Gmail','MySQL','Outlook','Skype','SMB','Weibo','WorldOfWarcraft','Cridex','Geodo','Htbot','Miuref','Neris','Nsis-ay','Shifu','Tinba','Virut','Zeus']
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)

x, y_true = load_mnist('/data1/ningjh/project/transfer0317/mnist/trans/', kind='t10k')
classes = ['BitTorrent','Facetime','FTP','Gmail','MySQL','Outlook','Skype','SMB','Weibo','WorldOfWarcraft','Cridex','Geodo','Htbot','Miuref','Neris','Nsis-ay','Shifu','Tinba','Virut','Zeus']
proba = model_trans.predict([data_test,data_test], batch_size=batch_size, verbose=1)

max = np.max(proba, axis=1)
y_pred = np.zeros(shape=(x.shape[0], 1))
for i in range(x.shape[0]):
    for j in range(20):
        if (proba[i][j] == max[i]):
            y_pred[i] = j
cm = confusion_matrix(y_true, y_pred)

plt.figure()
plot_confusion_matrix(cm, classes, normalize=True, cmap=plt.cm.Blues)
plt.show()
