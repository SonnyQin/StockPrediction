import Preprocess
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

#标注缺陷， 特征的权重目前没有调， 各个特征视为相同地位, 有点可惜， 放神经网络里做吧

a=np.array([1,2,3,4,5,6])
b=np.array([6,5,4,3,2,1])

#依次训练各个特征, data是np.mat
def kNN_Prediction(data, k=5, predictNum=200):
    data=np.array(data)
    
    #整理特征数据
    #去除日期
    x=np.delete(data, 0, 1)
    x=x.astype(np.float64)
    #将最后一项特征删除，因为无法推断
    x=x[:-1]
    x=pd.DataFrame(x)
    raw=x.copy()

    #添加时间序列
    timeOrder=np.linspace(1, x.shape[0], x.shape[0])
    x['TimeOrder']=timeOrder

    #将时间序列转换为监督数据, 不需要加一， 后面会从原始数据中找下一个数据
    x=Preprocess.series_to_supervised(x, 30, 1)
    
    #记录最后的时间序号
    LastIndex=x.shape[0]
    
    #后续50个数据的存储, 记录未归一化的数据
    predict=[x.iloc[-1],]
    #print(predict)
    
    #记录特征数
    traitAmount=x.shape[1]
    
    #归一化数据
    preX=x.copy()
    scaler=MinMaxScaler()
    scaler.fit(x)
    x=scaler.transform(x)
    
    #print(x[:,-1])
    
    #设置分类器
    kNN=KNeighborsClassifier()
    paras={'n_neighbors':k}
    kNN.set_params(**paras)
    y=np.ones((x.shape[0], x.shape[1]))
    kNN.fit(x,y)
    
    #推测数据
    for i in range(1, predictNum):
        distance, indices=kNN.kneighbors(scaler.transform([predict[i-1]]))
        #反函数距离权重
        distance=distance+0.00000001
        distance=1/distance
        sum=distance.sum(axis=1)
        #计算权重
        distance=distance/sum
        indices=indices.reshape(-1)
        #根据符合条件的下一个点
        indices=indices+1
        #neighbours=raw[indices]
        neighbours=raw.iloc[indices]
        results=[]
        #找到点， 并根据权重计算, 行末几位是当前时间的数据
        neighbourArray=np.array(neighbours)
        for j in range(0,5):
            #预测出几个特征值
            results.append((neighbourArray[:,j]*distance).sum(axis=1))
        LastIndex+=1
        
        #将预测好的数据重放回
        #raw 没有时间序列
        results=[x[0] for x in results]
        cresults=pd.DataFrame([results])
        raw=raw._append(cresults)
        #raw=raw.reset_index()
        #print(raw.tail(5))
        
        #更新预测的值
        results.append(LastIndex)
        new=predict[-1]
        new=new.shift(-6)
        count=0
        for index in new.tail(6).index:
             new[index]=results[count]
             count+=1
        predict.append(new)

        #重新训练
        new=pd.DataFrame(new).T
        pd.concat([preX, new], axis=0)
        x=scaler.transform(preX)
        kNN.fit(x, y)
        
    return np.array(predict).astype(np.float64)

def loss(x, y):
    return Preprocess.MSE(x,y)
    

#全局最佳
def paraAdjustW(data, t, kMax=20, amount=200):
    minn=np.inf
    k=0
    t=Preprocess.arrange(t)    #t现在是np.array
    outResults=[]
    for i in range(1, kMax):
        results=kNN_Prediction(data, i,amount)
        l=loss(results, t)
        if  l< minn:
            minn=l
            k=i
            outResults=results
        print(k)
        print(minn)
    
    return k, minn

#指定最佳
def paraAdjustA(data, t, a=3,kMax=20, amount=200):
    minn=np.inf
    k=0
    t=Preprocess.arrange(t)    #t现在是np.array
    outResults=[]
    for i in range(10, kMax):
        results=kNN_Prediction(data, i,amount)
        l=loss(results[:,a], t[:,a+1])
        if  l< minn:
            minn=l
            k=i
            outResults=results
        print(k)
        print(minn)
    
    return k, minn
#预测
def predict(data,name, save):
    
    x=data[name][:-201].copy()
    y=data[name][-200:].copy()
    k,min=paraAdjustA(x,y,3)
    
    results=kNN_Prediction(data[name][:-51],k,50)
    plt.plot(data[name][-50:,4],linewidth=0.5)
    plt.plot(results[:,3],linewidth=1)
    plt.title(name)
    plt.legend(['Real','Predict'])
    #输出均方差
    print(min)
    if save:
        plt.savefig(r'C:/Users/15105/Desktop/Predict/'+name)
    else:
        plt.show()
    plt.close()
     
# if __name__=='__main__':
#     data=Preprocess.load_data()
#     count=0
#     for key in data.keys():
#         count+=1
#         # try:
#         predict(data,key, True)
#         # except:
#         #     pass
#     print(count)

data=Preprocess.load_data()
predict(data, 'AAL', False)