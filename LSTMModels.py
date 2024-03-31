import tensorflow as tf 
import Preprocess
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import TimeDistributed
MultiToOneParams={'n_traits':5,'n_neurons':128, 'n_output':50, 'n_epoches':300, 'n_timeOrder':59, 'n_dropoutrate':0.2,'n_testDataAmount':50}
MultiToMultiParams={'n_traits':5,'n_neurons':128, 'n_output':50, 'n_epoches':300, 'n_timeOrder':59, 'n_dropoutrate':0.2,'n_testDataAmount':50}
# transform the time series data into supervised learning
scaler=MinMaxScaler()
#MultiToMulti的参数
# closedscaler=MinMaxScaler()
def getStockDataMultiToOne(name='FTI'):
    data=Preprocess.load_data()[name][:,1:]
    data=data.astype('float32')
    data=np.array(data)
    #正则化
    scaler.fit(data)
    # closedscaler.fit(data[:, 3].reshape(-1,1))
    data=scaler.transform(data)
    
    supervised_data=Preprocess.series_to_supervised(data, n_in=MultiToOneParams['n_timeOrder'])
    #删除当前的四个特征， 只保留closedvalue
    supervised_data=supervised_data.drop(columns=['var1(t)', 'var2(t)', 'var3(t)', 'var5(t)'])
    supervised_data=supervised_data.values
    return supervised_data

def getStockDataMultiToMulti(name='FIT'):
    data=Preprocess.load_data()[name][:,1:]
    data=data.astype('float32')
    data=np.array(data)
    #正则化
    scaler.fit(data)
    # closedscaler.fit(data[:, 3].reshape(-1,1))
    data=scaler.transform(data)
    supervised_data=Preprocess.series_to_supervised(data, n_in=1)
    supervised_data=supervised_data.drop(columns=['var1(t)', 'var2(t)', 'var3(t)', 'var5(t)'])
    supervised_data=Preprocess.series_to_supervised(supervised_data, n_in=MultiToMultiParams['n_timeOrder'])
    return supervised_data.values
    

def splitDataMultiToOne(data):
    x_train, y_train, x_test, y_test=data[:-MultiToOneParams['n_testDataAmount'],:-1], data[:-MultiToOneParams['n_testDataAmount'],1], data[-MultiToOneParams['n_testDataAmount']:,:-1], data[-MultiToOneParams['n_testDataAmount']:,1]
    trainBatchNum=len(x_train)
    testBatchNum=len(x_test)
    x_train=x_train.reshape(trainBatchNum, MultiToOneParams['n_timeOrder'], MultiToOneParams['n_traits'])
    y_train=y_train.reshape(trainBatchNum, 1)
    x_test=x_test.reshape(testBatchNum, MultiToOneParams['n_timeOrder'], MultiToOneParams['n_traits'])
    y_test=y_test.reshape(testBatchNum, 1)
    return x_train, y_train, x_test, y_test

def splitDataMultiToMulti(data):
    batchnum=len(data)
    data=data.reshape(batchnum, MultiToMultiParams['n_timeOrder'], MultiToMultiParams['n_traits'])
    x_train, y_train, x_test, y_test=data[:-MultiToMultiParams['n_testDataAmount'],:,:-1], data[:-MultiToMultiParams['n_testDataAmount'],:,-1], data[-MultiToMultiParams['n_testDataAmount']:,:,:-1], data[-MultiToMultiParams['n_testDataAmount']:,:,-1]
    return x_train, y_train, x_test, y_test
    

def MultiToOneModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(MultiToOneParams['n_neurons'], input_shape=(MultiToOneParams['n_timeOrder'],MultiToOneParams['n_traits']), return_sequences=True),
        tf.keras.layers.Dropout(MultiToOneParams['n_dropoutrate']),
        tf.keras.layers.LSTM(MultiToOneParams['n_neurons'], input_shape=(MultiToOneParams['n_timeOrder'],MultiToOneParams['n_traits']), return_sequences=True),
        tf.keras.layers.Dropout(MultiToOneParams['n_dropoutrate']),
        tf.keras.layers.LSTM(MultiToOneParams['n_neurons'], input_shape=(MultiToOneParams['n_timeOrder'],MultiToOneParams['n_traits'])),
        tf.keras.layers.Dropout(MultiToOneParams['n_dropoutrate']),
        tf.keras.layers.Dense(32,kernel_initializer="uniform",activation='relu'),      
        tf.keras.layers.Dense(1,kernel_initializer="uniform",activation='linear')
        ],name="LSTM")
    # 模型编译
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    model.summary()
    return model

def MultiToMultiModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(n_neurons, input_shape=(n_timeOrder,n_traits), return_sequences=True),
        tf.keras.layers.Dropout(n_dropoutrate),
        tf.keras.layers.LSTM(n_neurons, input_shape=(n_timeOrder,n_traits), return_sequences=True),
        tf.keras.layers.Dropout(n_dropoutrate),
        tf.keras.layers.LSTM(n_neurons, input_shape=(n_timeOrder,n_traits), return_sequences=True),
        tf.keras.layers.Dropout(n_dropoutrate),
        TimeDistributed(tf.keras.layers.Dense(32,kernel_initializer="uniform",activation='relu')),      
        TimeDistributed(tf.keras.layers.Dense(1,kernel_initializer="uniform",activation='linear'))
        ],name="LSTM")
    # 模型编译
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    model.summary()
    return model

def train(x_train, y_train):
    model=MultiToOneModel()
    model.fit(x_train, y_train, epochs=MultiToOneParams['n_epoches'],batch_size=512,
    validation_split=0.1,
    verbose=1)
    model.save('LSTMMultiToOne.h5') 
    
    
def predict(x_test, y_test):
    model=tf.keras.models.load_model('LSTMMultiToOne.h5')
    result=model.predict(x_test)
    # result=closedscaler.inverse_transform(result)
    plt.plot(y_test)
    plt.plot(result)
    plt.show()
    


data=getStockDataMultiToOne()
x_train, y_train, x_test, y_test=splitDataMultiToOne(data)
train(x_train, y_train)
predict(x_test, y_test)
    
    
# train()
# data=getStockData()
# x_train, y_train, x_test, y_test=splitData(data)
# plt.plot(y_test)
# plt.show()