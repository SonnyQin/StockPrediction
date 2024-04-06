import tensorflow as tf 
import Preprocess
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import TimeDistributed
MTOP={'n_traits':5,'n_neurons':200, 'n_output':100, 'n_epoches':400, 'n_timeOrder':73, 'n_dropoutrate':0.01,'n_testDataAmount':300 }
MTMP={'n_traits':5,'n_neurons':200, 'n_output':100, 'n_epoches':300, 'n_timeOrder':59, 'n_dropoutrate':0.2,'n_testDataAmount':50}
# transform the time series data into supervised learning
scaler=MinMaxScaler()
#MultiToMulti的参数
# closedscaler=MinMaxScaler()
def getStockDataMultiToOne(name='FTI'):
    data=Preprocess.load_data()[name][:,1:]
    data=data.astype('float64')
    data=np.array(data)
    #正则化
    scaler.fit(data)
    # closedscaler.fit(data[:, 3].reshape(-1,1))
    data=scaler.transform(data)
    
    supervised_data=Preprocess.series_to_supervised(data, n_in=MTOP['n_timeOrder'])
    #删除当前的四个特征， 只保留closedvalue
    #supervised_data=supervised_data.drop(columns=['var1(t)', 'var2(t)', 'var3(t)', 'var5(t)'])
    supervised_data=supervised_data.values
    return supervised_data

def getStockDataMultiToOneT(name='FTI'):
    data=Preprocess.load_data()[name][:,1:]
    data=data.astype('float64')
    data=np.array(data)
    #正则化
    scaler.fit(data)
    # closedscaler.fit(data[:, 3].reshape(-1,1))
    data=scaler.transform(data)
    
    supervised_data=Preprocess.series_to_supervised(data, n_in=MTOP['n_timeOrder'])
    #删除当前的四个特征， 只保留closedvalue
    supervised_data=supervised_data.drop(columns=['var1(t)', 'var2(t)', 'var3(t)', 'var5(t)'])
    closed=np.array(supervised_data['var4(t)'])
    supervised_data=supervised_data.values

    #one-hot 编码， 分别是涨跌平
    #代码不是很好看，懒得改了
    target=[[0,0,0]]
    for i in range(1,supervised_data.shape[0]):
        if closed[i]>closed[i-1]:
            target.append([1,0,0])
        elif closed[i]<closed[i-1]:
            target.append([0,1,0])
        else:
            target.append([0,0,1])

    return supervised_data, target

def getStockDataMultiToMulti(name='FIT'):
    data=Preprocess.load_data()[name][:,1:]
    data=data.astype('float64')
    data=np.array(data)
    #正则化
    scaler.fit(data)
    # closedscaler.fit(data[:, 3].reshape(-1,1))
    data=scaler.transform(data)
    supervised_data=Preprocess.series_to_supervised(data, n_in=1)
    #=supervised_data.drop(columns=['var1(t)', 'var2(t)', 'var3(t)', 'var5(t)'])
    supervised_data=Preprocess.series_to_supervised(supervised_data, n_in=MTMP['n_timeOrder'])
    return supervised_data.values
    

def splitDataMultiToOne(data):
    x_train, y_train, x_test, y_test=data[:-MTOP['n_testDataAmount'],:-MTOP['n_traits']], data[:-MTOP['n_testDataAmount'],-MTOP['n_traits']:], data[-MTOP['n_testDataAmount']:,:-MTOP['n_traits']], data[-MTOP['n_testDataAmount']:,-MTOP['n_traits']:]
    trainBatchNum=len(x_train)
    testBatchNum=len(x_test)
    x_train=x_train.reshape(trainBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
    y_train=y_train.reshape(trainBatchNum, MTOP['n_traits'])
    x_test=x_test.reshape(testBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
    y_test=y_test.reshape(testBatchNum, MTOP['n_traits'])
    return x_train, y_train, x_test, y_test

def splitDataMultiToOneT(dat, target):
    x_train, y_train, x_test, y_test=data[:-MTOP['n_testDataAmount'],:-1], target[:-MTOP['n_testDataAmount']], data[-MTOP['n_testDataAmount']:,:-1], target[-MTOP['n_testDataAmount']:]
    trainBatchNum=len(x_train)
    testBatchNum=len(x_test)
    x_train=x_train.reshape(trainBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
    y_train=np.array(y_train)
    x_test=x_test.reshape(testBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
    y_test=np.array(y_test)
    return x_train, y_train, x_test, y_test

def splitDataMultiToMulti(data):
    batchnum=len(data)
    data=data.reshape(batchnum, MTMP['n_timeOrder'], MTMP['n_traits'])
    x_train, y_train, x_test, y_test=data[:-MTMP['n_testDataAmount'],:,:-1], data[:-MTMP['n_testDataAmount'],:,-1], data[-MTMP['n_testDataAmount']:,:,:-1], data[-MTMP['n_testDataAmount']:,:,-1]
    return x_train, y_train, x_test, y_test
    

def MultiToOneModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits'])),
        tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'),
        tf.keras.layers.Dense(MTOP['n_traits'],kernel_initializer="uniform",activation='linear')
        ],name="LSTM")
    # 模型编译
    model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
    model.summary()
    return model

def MultiToOneModelT():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits'])),
        tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'),
        tf.keras.layers.Dense(3)
        ],name="LSTM")
    # 模型编译
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
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
    history=model.fit(x_train, y_train, epochs=MTOP['n_epoches'],
    validation_split=0.1,
    verbose=1)
    plt.plot(history.history['loss'])
    plt.show()
    model.save('LSTMMultiToOne.h5')

def trainT(x_train, y_train):
    model=MultiToOneModelT()
    history=model.fit(x_train, y_train, epochs=MTOP['n_epoches'],
    batch_size=10,
    validation_split=0.1,
    verbose=1)
    plt.plot(history.history['loss'])
    plt.show()
    model.save('LSTMMultiToOne.h5')     
    
    
def predict(x_test, y_test):
    model=tf.keras.models.load_model('LSTMMultiToOne.h5')
    result=model.predict(x_test)
    print(model.evaluate(x_test,y_test))
    # result=closedscaler.inverse_transform(result)
    plt.plot(y_test[:,3])
    #单步预测的结果
    plt.plot(result[:,3])
    #递归单步预测
    results=[]
    last=x_test[0].reshape(1,x_test[0].shape[0],-1)
    for i in range(0,300):
        last=model.predict(last)
        results.append(last[0][3])
        last=x_test[0].reshape(1,x_test[0].shape[0],-1)
    plt.plot(results)
    plt.show()

def predictT(x_test, y_test):
    model=tf.keras.models.load_model('LSTMMultiToOne.h5')
    print(model.evaluate(x_test, y_test))
    results=model.predict(x_test)
    print('Results:')
    print(results)
    print('Y_test')
    print(y_test)

    
data=getStockDataMultiToOne()
# print(data[-1,:])
x_train, y_train, x_test, y_test=splitDataMultiToOne(data)
# plt.plot(y_test)
# plt.show()
# train(x_train, y_train)
predict(x_train, y_train)
    
    
# train()
# data=getStockDataMultiToOne()
# x_train, y_train, x_test, y_test=splitDataMultiToOne(data)
# plt.plot(results)
# plt.plot(y_test)
# plt.show()