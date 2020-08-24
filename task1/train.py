# -*- coding: utf-8 -*-

import numpy as np
from keras.optimizers import SGD, Adam
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU
from keras.models import Sequential, Model
from keras import backend as K
from keras.regularizers import l2
import os.path
import csv
import cv2
import glob
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json
from keras import callbacks
import math
from matplotlib import pyplot

# 对于摄像头拍摄的图片，进行转动角度的预测

SEED = 13

# 模型
def get_model(shape):
    '''
    预测方向盘角度: 以图像为输入, 预测方向盘的转动角度
    shape: 输入图像的尺寸, 例如(128, 128, 3)
    '''

    model = Sequential()
    
    model.add(Conv2D(8, (5, 5), strides=(1, 1), padding="valid", activation='relu', input_shape=shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    sgd = SGD(lr=0.01)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    return model

# 读入路径为img_address的图片并将图片转为RGB格式
def image_transformation(img_address, degree, data_dir):
    # print(data_dir + img_address)  # E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/贪心学院project3dataset/IMG/center_2016_12_01_13_34_10_600.jpg
    img = cv2.imread(data_dir + img_address)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return (img_RGB, degree)

# 传入的参数为：X_train, y_train, batch_size = 32, shape = (128, 128, 3), training=True
def batch_generator(x, y, batch_size, shape, training=True, data_dir=r'E:/tanxinxueyuan/tanxin/ziliao/project3dataset/', monitor=True, yieldXY=True, discard_rate=0.95):
    """
    产生批处理的数据的generator
    x: 文件路径list
    y: 方向盘的角度
    training: 值为True时产生训练数据
              值为True时产生validation数据
    batch_size: 批处理大小
    shape: 输入图像的尺寸(高, 宽, 通道)
    data_dir: 数据目录, 包含一个IMG文件夹（真实图片路径）
    monitor: 保存一个batch的样本为 'X_batch_sample.npy‘ 和'y_bag.npy’
    yieldXY: 为True时, 返回(X, Y)
             为False时, 只返回 X only
    discard_rate: 随机丢弃角度为零的训练数据的概率
    """
    
    if training:  # True
        y_bag = []
        x, y = shuffle(x, y)  # 随机打乱训练数据
        new_x = x  # 赋值
        new_y = y  # 赋值
    else:
        new_x = x
        new_y = y

    # print("new_x的长度为：")
    # print(len(new_x))  # 6428
    
    offset = 0
    while True:  # 进入循环
        X = np.empty((batch_size, *shape))  # 转为（batch_size，height,width,channel）格式
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):  # 循环32次每张图片,0.1....31
            # img_address：图片路径，img_steering：转动的角度
            img_address, img_steering = new_x[example + offset], new_y[example + offset]
            # print("图片地址为：")  # IMG/center_2016_12_01_13_32_58_012.jpg
            # print(img_address)
            
            if training:
                # print(data_dir)  # E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/贪心学院project3dataset/
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                img = cv2.imread(data_dir + img_address)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            X[example,:,:,:] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1]) ) / 255 - 0.5
            
            Y[example] = img_steering
            if training:
                y_bag.append(img_steering)
            
            '''
             到达原来数据的结尾, 从头开始
            '''
            if (example + 1) + offset > len(new_y) - 1:
                x, y = shuffle(x, y)
                new_x = x
                new_y = y
                offset = 0
        if yieldXY:
            yield (X, Y)
        else:
            yield X

        offset = offset + batch_size
        if training:
            np.save('y_bag.npy', np.array(y_bag) )
            np.save('Xbatch_sample.npy', X ) 


if __name__ == '__main__':

    # CSV文件路径
    data_path = r'E:/tanxinxueyuan/tanxin/ziliao/course/kejian/Project3-Self-Driving-Car/Project3-Self-Driving-Car/data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array(log)
    # print(log.shape)  # (8037, 7)

    # 去掉文件第一行
    log = log[1:,:]   # (8036, 7)

    # 判断图像文件数量是否等于csv日志文件中记录的数量
    drive_data_path = r'E:/tanxinxueyuan/tanxin/ziliao/project3dataset/IMG/*.jpg'
    # 真实图像路径
    drive_data = r'E:/tanxinxueyuan/tanxin/ziliao/project3dataset/IMG/'
    ls_imgs = glob.glob(drive_data_path)
    # print(len(ls_imgs))  24108
    # print(len(log)) 8036
    assert len(ls_imgs) == len(log)*3, 'number of images does not match'

    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size =32
    nb_epoch = 5

    x_ = log[:, 0]
    # print(len(x_))  # IMG/center_2016_12_01_13_30_48_287.jpg 8036
    y_ = log[:, 3].astype(float)
    # print(len(y_))  # 0.0 8036
    x_, y_ = shuffle(x_, y_)  # 打乱位置
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ratio, random_state=SEED)

    print('batch size: {}'.format(batch_size))  # 32
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))  # 6428 | 1608

    samples_per_epoch = batch_size  # 32
    # 使得validation数据量大小为batch_size的整数陪
    nb_val_samples = len(y_val) - len(y_val)%batch_size

    model = get_model(shape)
    print(model.summary())

    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1,
                                         save_best_only=True, mode='min')

    # 如果训练持续没有validation loss的提升, 提前结束训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15,
                                         verbose=1, mode='auto')
    callbacks_list = [early_stop, save_best]

    # 训练网络
    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True),
                                  steps_per_epoch = samples_per_epoch,
                                  validation_steps = nb_val_samples // batch_size,
                                  validation_data = batch_generator(X_val, y_val, batch_size, shape,
                                                                  training=False, monitor=False),
                                  epochs=nb_epoch, verbose=1, callbacks=callbacks_list)


    with open('./trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.savefig('train_val_loss.jpg')

    # 保存模型
    with open('./model.json', 'w') as f:
            f.write(model.to_json())
    model.save('model.h5')
    print('Done!')