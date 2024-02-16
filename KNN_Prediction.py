import Preprocess
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#依次训练各个特征, data是np.mat
def kNN_Prediction(data):
    #后续50个数据的存储, 记录未归一化的数据
    predict=[[data[-1][1:]],]
    #遍历五个特征, 并获取到5个特征的分类器
    classifiers=[]
    #用于记录各个特征的标签
    targets=[]
    
    #整理特征数据
    #去除日期
    x=np.delete(data, 0, 1)
    #将最后一项特征删除，因为无法推断
    x=x[:-1]
    x=x.astype(np.float64)
    #归一化数据
    scaler=MinMaxScaler()
    scaler.fit(x)
    x=scaler.transform(x)
    print(x)
    
    for i in range(1,6):
        #通过前一天的特征推断出当前天的选定特征
        #数据第一个删除, 无法作为当前标签
        y=data[1:,i]
        y=y.astype(np.float64)
        #y可能不能使用浮点, 先将浮点值乘以100待会再除以100
        targets.append(y)
        kNN=KNeighborsClassifier()
        kNN.fit(x,y.astype(np.int32))
        paras={'n_neighbors':1}
        kNN.set_params(**paras)
        classifiers.append(kNN)
    #推测后续50个数据
    for i in range(1, 100):
        results=[]
        for j in range(0, 5):
            distance,indices=classifiers[j].kneighbors(scaler.transform(predict[i-1]))
            distance=distance+0.00000001
            distance=1/distance
            sum=distance.sum(axis=1)
            distance=distance/sum
            print(distance)
            results.append((targets[j][indices]*distance).sum(axis=1))

        # results=np.array(results)
        # results=results.reshape(-1)
        results=[[d[0] for d in results]]
        # print(predict[-1])
        # print(results)
        predict.append(results)
    predict=np.array(predict)
    predict = np.delete(predict, 0, axis=0)
    predict=predict.reshape(-1, 5)
    return predict
            
data=Preprocess.load_data()
results=kNN_Prediction(data['FLS'][:-50])
print(results)
x=results[:,3]
x=x.astype(np.float64)
plt.plot(x,linewidth=1, color='red')
plt.show()