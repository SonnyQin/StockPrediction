import Preprocess
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

a=np.array([1,2,3,4,5,6])
b=np.array([6,5,4,3,2,1])

def arrange(data):
    data=np.array(data)
    x=np.delete(data, 0, 1)
    x=x.astype(np.float64)
    return x

#依次训练各个特征, data是np.mat
def kNN_Prediction(data, k=5, predictNum=50):
    data=np.array(data)
    #后续50个数据的存储, 记录未归一化的数据
    predict=[data[-1][1:],]
    
    #整理特征数据
    #去除日期
    x=np.delete(data, 0, 1)
    x=x.astype(np.float64)
    raw=x.copy()
    #将最后一项特征删除，因为无法推断
    x=x[:-1]
    #归一化数据
    scaler=MinMaxScaler()
    scaler.fit(x)
    x=scaler.transform(x)
    
    #设置分类器
    kNN=KNeighborsClassifier()
    paras={'n_neighbors':k}
    kNN.set_params(**paras)
    y=np.ones((x.shape[0], x.shape[1]))
    kNN.fit(x,y)
    
    #推测数据
    for i in range(1, predictNum):
        distance, indices=kNN.kneighbors(scaler.transform([predict[i-1]]))
        distance=distance+0.00000001
        distance=1/distance
        sum=distance.sum(axis=1)
        #计算权重
        distance=distance/sum
        indices=indices.reshape(-1)
        #根据符合条件的下一个点
        indices=indices+1
        neighbours=raw[indices]
        results=[]
        #找到点， 并根据权重计算
        for j in range(0,5):
            results.append((neighbours[:,j]*distance).sum(axis=1))
        results=np.array(results)
        results=results.reshape(-1)
        #results=[[d[0] for d in results]]
        predict.append(results)
        #print(results)
    return np.array(predict).astype(np.float64)

def loss(x, y):
    return np.sum((x-y)*(x-y))
    

#全局最佳
def paraAdjustW(data, t, kMax=100):
    minn=np.inf
    k=0
    t=arrange(t)    #t现在是np.array
    outResults=[]
    for i in range(1, kMax):
        results=kNN_Prediction(data, i)
        l=loss(results, t)
        if  l< minn:
            minn=l
            k=i
            outResults=results
        print(k)
        print(minn)
    
    return k, minn
        
     
if __name__=='__main__':
    name='MCD'
    
    data=Preprocess.load_data()

    x=data[name][:-51].copy()
    y=data[name][-50:].copy()
    k,min=paraAdjustW(x,y)
    results=kNN_Prediction(data[name][:-51],k,50)
    plt.plot(results[:,3],linewidth=1, color='red')
    plt.show()