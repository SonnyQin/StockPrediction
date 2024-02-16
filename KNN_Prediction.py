from Preprocess import load_data
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
#依次训练各个特征, data是np.mat
def kNN_Prediction(data):
    #后续50个数据的存储
    predict=[[data[-1][1:]],]
    #遍历五个特征, 并获取到5个特征的分类器
    classifiers=[]
    for i in range(1,6):
        #通过前一天的特征推断出当前天的选定特征
        #去除日期
        x=np.delete(data, 0, 1)
        #将最后一项特征删除，因为无法推断
        x=x[:-1]
        #数据第一个删除, 无法作为当前标签
        y=data[1:,i]
        x=x.astype(np.float32)
        y=y.astype(np.float32)
        #y可能不能使用浮点, 先将浮点值乘以100待会再除以100
        y=y*100
        kNN=KNeighborsClassifier()
        kNN.fit(x,y.astype(np.int32))
        classifiers.append(kNN)
    #推测后续50个数据
    for i in range(1, 51):
        
data=load_data()
kNN_Prediction(data['AAL'])