import tensorflow as tf 
import Preprocess
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import TimeDistributed
#n_timeOrder在这里是每个batch中有多少个时间步
MTOP={'n_traits':5,'n_neurons':200, 'n_output':100, 'n_epoches':100, 'n_timeOrder':5, 'n_dropoutrate':0.01,'n_testDataAmount':300 }
MTMP={'n_traits':5,'n_neurons':200, 'n_output':100, 'n_epoches':300, 'n_timeOrder':59, 'n_dropoutrate':0.2,'n_testDataAmount':50}
# transform the time series data into supervised learning
#MultiToMulti的参数
# closedscaler=MinMaxScaler()
    

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
        TimeDistributed(tf.keras.layers.Dense(MTOP['n_output']),input_shape=(MTOP['n_timeOrder'],3*MTOP['n_traits'])),
        # tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],3*MTOP['n_traits']), return_sequences=True),
        # tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],3*MTOP['n_traits'])),
        tf.keras.layers.Dropout(MTOP['n_dropoutrate']),
        tf.keras.layers.Dense(MTOP['n_output'],kernel_initializer="uniform",activation='relu'),
        tf.keras.layers.Dense(3)
        ],name="LSTM")
    # 模型编译
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def MultiToMultiModel():
    model = tf.keras.models.Sequential([
        TimeDistributed(tf.keras.layers.Dense(MTMP['n_output'],kernel_initializer="uniform",activation='relu')),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.dropout(MTMP['n_dropoutrate']),
        tf.keras.layers.LSTM(MTOP['n_neurons'], input_shape=(MTOP['n_timeOrder'],MTOP['n_traits']), return_sequences=True),
        tf.keras.layers.dropout(MTMP['n_dropoutrate']),
        TimeDistributed(tf.keras.layers.Dense(MTMP['n_output'],kernel_initializer="uniform",activation='relu')),      
        TimeDistributed(tf.keras.layers.Dense(MTMP['n_traits'],kernel_initializer="uniform",activation='linear'))
        ],name="LSTM")
    # 模型编译
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
    model.summary()
    return model

def train(x_train, y_train):
    model=MultiToOneModel()
    history=model.fit(x_train, y_train, epochs=MTOP['n_epoches'],
    verbose=1)
    plt.plot(history.history['loss'])
    plt.show()
    model.save('LSTMMultiToOne.h5')
def trainMultiple(x_train, y_train):
    model=MultiToMultiModel()
    history=model.fit(x_train, y_train, epochs=MTMP['n_epoches'],
    verbose=1)
    plt.plot(history.history['loss'])
    plt.show()
    model.save('LSTMMultiToOne.h5')

def trainT(x_train, y_train):
    model=MultiToOneModelT()
    history=model.fit(x_train, y_train, epochs=MTOP['n_epoches'],
    batch_size=100,
    verbose=1)
    plt.plot(history.history['loss'])
    plt.show()
    model.save('LSTMMultiToOne.h5')     
    
    
def predict(x_test, y_test):
    model=tf.keras.models.load_model('LSTMMultiToOne.h5')
    result=model.predict(x_test)
    #print(model.evaluate(x_test,y_test))
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

    
# data=getStockDataMultiToOne(t=True)
# x_train, y_train, x_test, y_test=splitDataMultiToOne(data,t=True)
# data,target=getStockDataMultiToMulti()
# x_train, y_train, x_test, y_test=splitDataMultiToMulti(data,target)
# trainMultiple(x_train, y_train)
# predict(x_test, y_test)
    
# train()
# data=getStockDataMultiToOne()
# x_train, y_train, x_test, y_test=splitDataMultiToOne(data)
# plt.plot(results)
# plt.plot(y_test)
# plt.show()