import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

def load_data():
    rawdata = np.loadtxt(r'./all_stocks_5yr.csv',str, delimiter=',',skiprows=1, encoding='utf-8')#第82951，165735， 205078，239834，434381，434504，478594，558215，581908，598238数据缺失， 我先使用了前一项数据弥补确实的数据
    dictdata={}
    for item in rawdata:
        if not item[6] in dictdata.keys():
            dictdata[item[6]]=[]
        dictdata[item[6]].append(item[:6])
    for key in dictdata.keys():
        dictdata[key]=np.mat(dictdata[key])#ToDo:
    return dictdata
def DrawClosePrice(name, data):
    plt.yticks(np.arange(0, 100, 1))
    y=data[name][:,4]
    dates=data[name][:, 0]
    dates=dates.reshape(-1)
    xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates.A1]
    plt.plot(xs,y)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title(name+' Closing Price Change Chart')
    plt.tick_params(axis='both',which='both',labelsize=5)
    plt.show()
def DrawOpenPrice(name, data):
    plt.yticks(np.arange(0, 100, 1))
    y=data[name][:,1]
    dates=data[name][:, 0]
    dates=dates.reshape(-1)
    xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates.A1]
    plt.plot(xs,y)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title(name+' Opening Price Change Chart')
    plt.tick_params(axis='both',which='both',labelsize=5)
    plt.show()
def DrawMaximumPrice(name, data):
    plt.yticks(np.arange(0, 100, 1))
    y=data[name][:,2]
    dates=data[name][:, 0]
    dates=dates.reshape(-1)
    xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates.A1]
    plt.plot(xs,y)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title(name+' Maximum Price Change Chart')
    plt.tick_params(axis='both',which='both',labelsize=5)
    plt.show()
def DrawMinimumPrice(name, data):
    plt.yticks(np.arange(0, 100, 1))
    y=data[name][:,3]
    dates=data[name][:, 0]
    dates=dates.reshape(-1)
    xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates.A1]
    plt.plot(xs,y)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title(name+' Minimum Price Change Chart')
    plt.tick_params(axis='both',which='both',labelsize=5)
    plt.show()
def DrawVolume(name, data):
    plt.yticks(np.arange(0, 100, 1))
    y=data[name][:,5]
    dates=data[name][:, 0]
    dates=dates.reshape(-1)
    xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in dates.A1]
    plt.plot(xs,y)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('Value', fontsize=10)
    plt.title(name+' CVolume Change Chart')
    plt.tick_params(axis='both',which='both',labelsize=5)
    plt.show()
    
if __name__=='__main__':
    data=load_data()
    print(data['AAL'])
    DrawClosePrice('AAL', data=data)
    # DrawOpenPrice('AAL', data=data)
    # DrawMaximumPrice('AAL', data=data)
    # DrawMinimumPrice('AAL', data=data)
    # DrawVolume('AAL', data=data)
    
def arrange(data):
    data=np.array(data)
    x=np.delete(data, 0, 1)
    x=x.astype(np.float64)
    return x

def MSEw(x, y):
    return np.sum((x-y)*(x-y))#/(x.shape[0]*x.shape[1])

def MSE(x, y):
    return np.sum((x-y)*(x-y))/x.shape[0]

#梯度下降
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad

from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def gaussian(dist, sigma = 10.0):
    """ Input a distance and return it`s weight"""
    weight = np.exp(-dist**2/(2*sigma**2))
    return weight

def changeToLabels(data):
    labels=[[1,1,1,1,1]]
    for i in range(1,len(data)):
        r=[]
        for j in range(0,5):
            if data[i][j]<data[i-1][j]:
                r.append(0)
            elif data[i][j]==data[i-1][j]:
                r.append(1)
            else:
                r.append(2)
        labels.append(r)
    return np.array(labels)




def getStockData(MTOP, name='PXD', t=False):
    data=load_data()[name][:,1:]
    data=data.astype('float64')
    data=np.array(data)
    if(t==False):
        #正则化
        scaler.fit(data)
        # closedscaler.fit(data[:, 3].reshape(-1,1))
        data=scaler.transform(data)
    else:
        data=changeToLabels(data)
    
    supervised_data=series_to_supervised(data, n_in=MTOP['n_timeOrder'])
    #删除当前的四个特征， 只保留closedvalue
    #supervised_data=supervised_data.drop(columns=['var1(t)', 'var2(t)', 'var3(t)', 'var5(t)'])
    supervised_data=supervised_data.values
    return supervised_data

def getStockDataMultiToMulti(MTMP, name='PXD'):
    data=load_data()[name][:,1:]
    data=data.astype('float64')
    data=np.array(data)
    #正则化
    scaler.fit(data)
    # closedscaler.fit(data[:, 3].reshape(-1,1))
    data=scaler.transform(data)
    supervised_data=series_to_supervised(data, n_in=MTMP['n_timeOrder'])
    supervised_data=supervised_data.values
    target=supervised_data[1:]
    supervised_data=supervised_data[:-1]
    return supervised_data, target

def splitDataMultiToOne(data, MTOP,  t=False, supervised=False):
    x_train, y_train, x_test, y_test=data[:-MTOP['n_testDataAmount'],:-MTOP['n_traits']], data[:-MTOP['n_testDataAmount'],-MTOP['n_traits']:], data[-MTOP['n_testDataAmount']:,:-MTOP['n_traits']], data[-MTOP['n_testDataAmount']:,-MTOP['n_traits']:]
    trainBatchNum=len(x_train)
    testBatchNum=len(x_test)
    #如果需要的是监督数据
    if supervised==False:
        x_train=x_train.reshape(trainBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
        x_test=x_test.reshape(testBatchNum, MTOP['n_timeOrder'], MTOP['n_traits'])
    y_test=y_test.reshape(testBatchNum, MTOP['n_traits'])
    y_train=y_train.reshape(trainBatchNum, MTOP['n_traits'])
    if t==False:
        return x_train, y_train, x_test, y_test
    else:
        import tensorflow as tf
        #只需要一个特征作为结果
        y_train=y_train[:,3]
        y_test=y_test[:,3]
        x_train=np.array(tf.one_hot(x_train,3))
        y_train=np.array(tf.one_hot(y_train,3))
        x_test=np.array(tf.one_hot(x_test,3))
        y_test=np.array(tf.one_hot(y_test,3))
        return x_train.reshape(trainBatchNum, MTOP['n_timeOrder'],-1),y_train.reshape(trainBatchNum,-1),x_test.reshape(testBatchNum, MTOP['n_timeOrder'],-1),y_test.reshape(testBatchNum,-1)

def splitDataMultiToMulti(data, MTMP, target):
    x_train, y_train, x_test, y_test=data[:-MTMP['n_testDataAmount']], target[:-MTMP['n_testDataAmount']], data[-MTMP['n_testDataAmount']:], target[-MTMP['n_testDataAmount']:]
    trainBatchNum=len(x_train)
    testBatchNum=len(x_test)
    x_train=x_train.reshape(trainBatchNum, MTMP['n_timeOrder'], MTMP['n_traits'])
    y_train=y_train.reshape(trainBatchNum, MTMP['n_timeOrder'],MTMP['n_traits'])
    x_test=x_test.reshape(testBatchNum, MTMP['n_timeOrder'], MTMP['n_traits'])
    y_test=y_test.reshape(testBatchNum, MTMP['n_timeOrder'],MTMP['n_traits'])
    return x_train, y_train, x_test, y_test