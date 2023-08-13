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
# 定义RMSE
def RMS(records):
    root_mean = math.sqrt(sum([x ** 2 for x in records]) / len(records))
    return root_mean

# 数据处理，将matlab.double转化为dataframe
def datahandle(path):
    data = eng.xlsread(path)  # 读取数据
    data_base = data  # 读取完整数据
    df = pd.DataFrame(data_base)  # 将完整数据转化为dataframe
    return data, df


# 将数据进行TQWT特征提取
def TQWT(data, df, Q, r):
    N = df.shape[0]  # 读取每种数据长度
    N = float(N)
    M = df.shape[1]
    M = float(M)
    Q = float(Q)
    r = float(r)  # 固定r
    J = int((math.log10((N / (4 * (Q + 1)))) / (math.log10(Q + 1) / ((Q + 1) - 2 / r))))  # 定义J
    J = float(J)
    Feature = pd.DataFrame()
    da_a = pd.DataFrame()
    col = 1
    while col <= M:
        x = eng.read_data(data, col)  # 读取某一列数据
        d = eng.tqwt_radix2(x, Q, r, J)  # 对某一列数据进行TQWT变换
        for x in d:
            da = pd.DataFrame(x)    #第一行特征值
            da_a = pd.concat([da_a, da], axis=0)
        col = col+1
        Feature = da_a  #维度为（1210,1024）的特征矩阵
    return x, d, Feature, N



path = 'data.xlsx'
Q = 1.34
r = 3
Q = float(Q)
r = float(r)
data, df = datahandle(path)
x, w, fea, N = TQWT(data, df, Q, r)
fea.to_excel('feature_data.xlsx')
print(N)




