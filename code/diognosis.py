import pandas as pd
import tensorflow as tf
import glob
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, LSTM, Conv2D,BatchNormalization,GlobalMaxPooling2D,MaxPooling2D,Activation
import os
import cv2
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import os
import itertools
import matlab.engine
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from pyts.image import GramianAngularField
from matplotlib import image

eng = matlab.engine.start_matlab()  # 为读取matlab文件做准备


# 将信号分开后进行单支重构
def idatahandle(data, col, Q, r, num, N):
    J = int((math.log10((N / (4 * (Q + 1)))) / (math.log10(Q + 1) / ((Q + 1) - 2 / r))))  # 定义J
    J = float(J)
    y = eng.single_itqwt(data, col, Q, r, J, num)
    return y

def CNN_LSTMdata(data,col,Q, r, num, N):
    one_line = pd.DataFrame()
    single_data = idatahandle(data, col, Q, r, num, N) #单支重构函数（总数据,第几列,Q,r,重构支数）
    single_data = [single_data]
    for x in single_data:
        one_data = pd.DataFrame(x)  # 第一行特征值
        one_line = pd.concat([one_line, one_data], axis=1)
    return one_line

def CNNLSTM_model(trainX):
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(trainX.shape[1],1)))
    model.add(MaxPooling1D(pool_size=2))

    # 添加第二个卷积层及池化层
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # 添加LSTM层
    model.add(LSTM(64, return_sequences=False))

    # 添加全连接层
    model.add(Dense(256, activation='relu'))
    model.add(Dense(512, activation='softmax'))

    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练和评估模型
    #model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    #score = model.evaluate(X_test, y_test, batch_size=32)
    '''
    model.add(Conv1D(filters=128, kernel_size=3, strides=2, input_shape=(trainX.shape[1],1), activation='relu'))
    model.add(MaxPooling1D(pool_length=4))
    model.add(LSTM(units=128, return_sequences=False))
    #model.add(Flatten())
    #model.add(MaxPooling1D(1))
    model.add(Dense(units=20, activation='relu'))
    # model.add(Dense(units=1, activation='sigmoid'))
    # model.add(Dense(units=6, activation='linear'))
    # start = time.time()
    model.compile(loss='mean_squared_error', optimizer='Adam')'''
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')
    return model

def feature_extraction(data, col,Q,r,N):
    # ===================使用CNN-LSTM网络===================
    # ===================子带1
    num = 1
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature1 = CNN_LSTM.predict(X_train)
    # print(feature1)
    feature1 = pd.DataFrame(feature1)
    # ===================子带2
    num = 2
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature2 = CNN_LSTM.predict(X_train)
    # print(feature2)
    feature2 = pd.DataFrame(feature2)
    # ===================子带3
    num = 3
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature3 = CNN_LSTM.predict(X_train)
    # print(feature3)
    feature3 = pd.DataFrame(feature3)
    # ===================子带4
    num = 4
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature4 = CNN_LSTM.predict(X_train)
    # print(feature4)
    feature4 = pd.DataFrame(feature4)
    # ===================子带5
    num = 5
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature5 = CNN_LSTM.predict(X_train)
    # print(feature5)
    feature5 = pd.DataFrame(feature5)
    # ===================子带6
    num = 6
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature6 = CNN_LSTM.predict(X_train)
    # print(feature6)
    feature6 = pd.DataFrame(feature6)
    # ===================子带7
    num = 7
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature7 = CNN_LSTM.predict(X_train)
    # print(feature7)
    feature7 = pd.DataFrame(feature7)
    # ===================子带8
    num = 8
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature8 = CNN_LSTM.predict(X_train)
    # print(feature8)
    feature8 = pd.DataFrame(feature8)
    # ===================子带9
    num = 9
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature9 = CNN_LSTM.predict(X_train)
    # print(feature9)
    feature9 = pd.DataFrame(feature9)
    # ===================子带10
    num = 10
    num = float(num)
    beforefea = CNN_LSTMdata(data, col, Q, r, num, N)
    X_train = beforefea
    X_train = X_train.values
    CNN_LSTM = CNNLSTM_model(X_train)
    CNN_LSTM.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    X_train = X_train.reshape((1, 1000, 1))
    feature10 = CNN_LSTM.predict(X_train)
    # print(feature10)
    feature10 = pd.DataFrame(feature10)
    all_feature = pd.DataFrame()
    all_feature= pd.concat([all_feature, feature1], axis=0)
    all_feature = pd.concat([all_feature, feature2], axis=0)
    all_feature = pd.concat([all_feature, feature3], axis=0)
    all_feature = pd.concat([all_feature, feature4], axis=0)
    all_feature = pd.concat([all_feature, feature5], axis=0)
    all_feature = pd.concat([all_feature, feature6], axis=0)
    all_feature = pd.concat([all_feature, feature7], axis=0)
    all_feature = pd.concat([all_feature, feature8], axis=0)
    all_feature = pd.concat([all_feature, feature9], axis=0)
    all_feature = pd.concat([all_feature, feature10], axis=0)
    return all_feature



def diagnosis_model():
    '''

    :param m: CNN-LSTM网络个数
    :param len_classes: 滤波器数
    :param dropout_rate:
    '''
    model = tf.keras.models.Sequential()

    model.add(Conv2D(512, kernel_size=(2, 2), activation='relu',input_shape=(10, 1000, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    #model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #model.add(Dropout(0.2))

    # Fully connected layer 1
    model.add(Conv2D(filters=64, kernel_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Fully connected layer 2
    model.add(Conv2D(filters=32, kernel_size=(2, 2), strides=2))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling2D())



    model.add(Dense(10, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    print(model.summary())
    print(f'Total number of layers: {len(model.layers)}')
    return model

# 混淆矩阵定义
def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    plt.yticks(tick_marks, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('真实类别')
    plt.xlabel('预测类别')
    plt.show()

# 显示混淆矩阵
def plot_confuse(predictions, real):
    #truelabel = real.argmax(axis=-1)   # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=real, y_pred=predictions)
    print(conf_mat)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(real)+1))
    return conf_mat




result = pd.read_excel('data2.xlsx')
train_Y = result.iloc[0:648, :]
test_Y = result.iloc[648:1295,:]
trainY = tf.keras.utils.to_categorical(train_Y, num_classes=10)
testY = tf.keras.utils.to_categorical(test_Y, num_classes=10)
path = 'feature.xlsx'
Q = 1.34
r = 3
Q = float(Q)
r = float(r)
data, df = datahandle(path)
x, w, fea, N = TQWT(data, df, Q, r)
COLS = df.shape[1]
col = 1
kong = pd.DataFrame()
while col <= COLS:
    feature_one = feature_extraction(data, col, Q, r, N)
    kong = pd.concat([kong, feature_one], axis=0)
    # print(feature_one )
    col = col + 1
print(kong)
#kong.to_excel('isingle.xlsx')

trainX = kong.iloc[0:6480, :]
testX = kong.iloc[6480:12950, :]


trainX = (trainX.T.values).reshape(6480, 1, 1000, 1)
testX = (testX.T.values).reshape(6470, 1, 1000, 1)


# ====================分类====================
# 训练分类器
estimator = KerasClassifier(build_fn=diagnosis_model, epochs=150, batch_size=1, verbose=1)  # 模型，轮数，每次数据批数，显示进度条
estimator.fit(trainX, trainY)  # 训练模型
# 将其模型转换为json
model_json = estimator.model.to_json()
with open(r"model.json", 'w')as json_file:
    json_file.write(model_json)  # 权重不在json中,只保存网络结构
estimator.model.save_weights('model.h5')
# 加载模型用做预测
json_file = open(r"model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 输出预测类别
predicted = loaded_model.predict(testX)  # 返回对应概率值
print(predicted)
predicted_label = np.argmax(loaded_model.predict(testX), axis=-1)
print(predicted_label)
# 分类准确率
#print("The accuracy of the classification model:")
#scores = loaded_model.evaluate(testX, testY, verbose=0)
#print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))
test_Y = test_Y.values.tolist()
# print("\npredicted label: " + str(predicted_label))
# print(11111111111111111111111111111111111)
# # 显示混淆矩阵
#
confmat = plot_confuse(predicted_label, test_Y)  # sklearn画，模型，测试集，测试集标签





Accuracy = (confmat[0][0] + confmat[1][1] + confmat[2][2] + confmat[3][3] + confmat[4][4] + confmat[5][5]) / 60
print("准确率A：", Accuracy)
Precision0 = confmat[0][0] / (confmat[0][0] + confmat[0][1] + confmat[0][2] + confmat[0][3] + confmat[0][4] + confmat[0][5])
print("类别0精确率P：", Precision0)
Recall0 = confmat[0][0] / (confmat[1][0] + confmat[2][0] + confmat[3][0] + confmat[4][0] + confmat[0][0] + confmat[5][0])
print("类别0召回率R：", Recall0)

Precision1 = confmat[1][1] / (confmat[1][0] + confmat[1][1] + confmat[1][2] + confmat[1][3] + confmat[1][4] + confmat[0][5])
print("类别1精确率P：", Precision0)
Recall1 = confmat[1][1] / (confmat[1][1] + confmat[2][1] + confmat[3][1] + confmat[4][1] + confmat[0][1] + confmat[5][1])
print("类别1召回率R：", Recall1)

Precision2 = confmat[2][2] / (confmat[2][0] + confmat[2][1] + confmat[2][2] + confmat[2][3] + confmat[2][4] + confmat[2][5])
print("类别2精确率P：", Precision2)
Recall2 = confmat[2][2] / (confmat[1][2] + confmat[2][2] + confmat[3][2] + confmat[4][2] + confmat[0][2] + confmat[5][2])
print("类别2召回率R：", Recall1)

Precision3 = confmat[3][3] / (confmat[3][0] + confmat[3][1] + confmat[3][2] + confmat[3][3] + confmat[3][4] + confmat[3][5])
print("类别3精确率P：", Precision0)
Recall3 = confmat[3][3] / (confmat[1][3] + confmat[2][3] + confmat[3][3] + confmat[4][3] + confmat[0][3] + confmat[5][3])
print("类别3召回率R：", Recall1)


Precision4 = confmat[4][4] / (confmat[4][0] + confmat[4][1] + confmat[4][2] + confmat[4][3] + confmat[4][4] + confmat[4][5])
print("类别4精确率P：", Precision0)
Recall4 = confmat[4][4] / (confmat[1][4] + confmat[2][4] + confmat[3][4] + confmat[4][4] + confmat[0][4] + confmat[5][4])
print("类别4召回率R：", Recall4)

Precision5 = confmat[5][5] / (confmat[5][0] + confmat[5][1] + confmat[5][2] + confmat[5][3] + confmat[5][4] + confmat[5][5])
print("类别5精确率P：", Precision5)
Recall5 = confmat[5][5] / (confmat[1][5] + confmat[2][5] + confmat[3][5] + confmat[4][5] + confmat[0][5] + confmat[5][5])
print("类别5召回率R：", Recall5)


