import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime
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
    DrawClosePrice('FLS', data=data)
    DrawOpenPrice('AAL', data=data)
    DrawMaximumPrice('AAL', data=data)
    DrawMinimumPrice('AAL', data=data)
    DrawVolume('AAL', data=data)
    
def arrange(data):
    data=np.array(data)
    x=np.delete(data, 0, 1)
    x=x.astype(np.float64)
    return x

def MSEw(x, y):
    return np.sum((x-y)*(x-y))#/(x.shape[0]*x.shape[1])

def MSE(x, y):
    return np.sum((x-y)*(x-y))/(x.shape[0]*x.shape[1])

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