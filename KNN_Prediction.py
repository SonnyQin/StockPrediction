import Preprocess
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

a=np.array([1,2,3,4,5,6])
b=np.array([6,5,4,3,2,1])

#依次训练各个特征, data是np.mat
def kNN_Prediction(data, k=5, predictNum=200):
    data=np.array(data)
    
    #整理特征数据
    #去除日期
    x=np.delete(data, 0, 1)
    x=x.astype(np.float64)
    raw=x.copy()
    #将最后一项特征删除，因为无法推断
    x=x[:-1]
    
    #添加时间序列
    timeOrder=np.linspace(1, x.shape[0], x.shape[0])
    x=np.insert(x, x.shape[1], timeOrder, axis=1)
    
    #记录最后的时间序号
    LastIndex=x.shape[0]
    
    #后续50个数据的存储, 记录未归一化的数据
    predict=[x[-1][:],]
    
    
    
    #归一化数据
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
        LastIndex+=1
        results.append(LastIndex)
        results=np.array(results)
        results=results.reshape(-1)
        predict.append(results)
        #将预测好的数据重放回
        
    return np.array(predict).astype(np.float64)

def loss(x, y):
    return Preprocess.MSEw(x,y)
    

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
def paraAdjustA(data, t, a=3,kMax=50, amount=200):
    minn=np.inf
    k=0
    t=Preprocess.arrange(t)    #t现在是np.array
    outResults=[]
    for i in range(1, kMax):
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
    k,min=paraAdjustA(x,y,3,amount=200)
    
    results=kNN_Prediction(data[name][:-51],k,50)
    plt.plot(data[name][-50:,4],linewidth=0.5)
    plt.plot(results[:,3],linewidth=1)
    if save:
        plt.savefig(r'C:/Users/A/Desktop/Predict/'+name)
    else:
        plt.show()
    plt.close()
     
if __name__=='__main__':
    data=Preprocess.load_data()
    count=0
    for key in data.keys():
        # count+=1
        # if(count>0):
            # try:
                predict(data,key, True)
            # except:
            #     pass
    print(count)