import tensorflow as tf 
import Preprocess
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import TimeDistributed
MTOP={'n_traits':5,'n_neurons':10, 'n_output':100, 'n_epoches':2000, 'n_timeOrder':173, 'n_dropoutrate':0.2,'n_testDataAmount':300 }
MTMP={'n_traits':5,'n_neurons':200, 'n_output':100, 'n_epoches':300, 'n_timeOrder':59, 'n_dropoutrate':0.2,'n_testDataAmount':50}
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
    
    supervised_data=Preprocess.series_to_supervised(data, n_in=MTOP['n_timeOrder'])
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
    supervised_data=Preprocess.series_to_supervised(supervised_data, n_in=MTMP['n_timeOrder'])
    return supervised_data.values
    

def splitDataMultiToOne(data):
    x_train, y_train, x_test, y_test=data[:-MTOP['n_testDataAmount'],:-1], data[:-MTOP['n_testDataAmount'],-1], data[-MTOP['n_testDataAmount']:,:-1], data[-MTOP['n_testDataAmount']:,-1]
    trainBatchNum=len(x_train)
    testBatchNum=len(x_test)
    x_train=x_train.reshape(trainBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
    y_train=y_train.reshape(trainBatchNum, 1)
    x_test=x_test.reshape(testBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
    y_test=y_test.reshape(testBatchNum, 1)
    return x_train, y_train, x_test, y_test

def splitDataMultiToMulti(data):
    batchnum=len(data)
    data=data.reshape(batchnum, MTMP['n_timeOrder'], MTMP['n_traits'])
    x_train, y_train, x_test, y_test=data[:-MTMP['n_testDataAmount'],:,:-1], data[:-MTMP['n_testDataAmount'],:,-1], data[-MTMP['n_testDataAmount']:,:,:-1], data[-MTMP['n_testDataAmount']:,:,-1]
    return x_train, y_train, x_test, y_test
    

def MultiToOneModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'), ),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'), ),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'), ),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'), ),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'), ),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        # tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        # tf.keras.layers.TimeDistributed( tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'), ),
        # tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits'])),
        tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']), 
        tf.keras.layers.Dense(1,kernel_initializer="uniform",activation='linear')
        ],name="LSTM")
    # 模型编译
    model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
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
    model.fit(x_train, y_train, epochs=MTOP['n_epoches'],
    validation_split=0.1,
    verbose=1)
    model.save('LSTMMultiToOne.h5') 
    
    
def predict(x_test, y_test):
    model=tf.keras.models.load_model('LSTMMultiToOne.h5')
    result=model.predict(x_test)
    print(model.evaluate(x_test,y_test))
    # result=closedscaler.inverse_transform(result)
    plt.plot(y_test)
    plt.plot(result)
    plt.show()
    
data=getStockDataMultiToOne('AAL')
# print(data[-1,:])
x_train, y_train, x_test, y_test=splitDataMultiToOne(data)
# plt.plot(y_test)
# plt.show()
# train(x_train, y_train)
predict(x_test, y_test)
    
    
# train()
# data=getStockDataMultiToOne()
# x_train, y_train, x_test, y_test=splitDataMultiToOne(data)
# plt.plot(results)
# plt.plot(y_test)
# plt.show()