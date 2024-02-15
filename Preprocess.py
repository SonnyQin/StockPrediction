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
        dictdata[key]=np.mat(dictdata[key])
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
data=load_data()
print(data['AAL'])
DrawClosePrice('AAL', data=data)
DrawOpenPrice('AAL', data=data)
DrawMaximumPrice('AAL', data=data)
DrawMinimumPrice('AAL', data=data)
DrawVolume('AAL', data=data)