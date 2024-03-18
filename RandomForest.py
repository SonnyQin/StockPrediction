import Preprocess
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def RandomForest_Prediction(closed):
    closed=np.array(closed)
    train=Preprocess.series_to_supervised(closed, 29, 2)
    x_train=train.iloc[:,:-1]
    y_train=train['var1(t+1)']
    print(y_train)
    
    #会少一个特征
    RandomForest=RandomForestRegressor()
    results=[]
    
    for i in range(0, 20):
        #选取的特征来预测
        x_Test=closed[-30:]
        x_Test=[x_Test.reshape(-1)]
        RandomForest.fit(x_train, y_train)
        result=RandomForest.predict(x_Test)
        closed=np.append(closed, result)
        results.append(result)
        result=pd.Series(result)
        y_train=y_train._append(result)
        x_Test=pd.DataFrame(x_Test, columns=x_train.columns)
        x_train=x_train._append(x_Test)
        print(x_train)
    
    return results

if __name__=='__main__':
    stockName='AAL'
    #所有结束时的价格
    data=Preprocess.load_data()[stockName]
    data=data[:,4]
    closed=data[:-51]
    
    results=RandomForest_Prediction(closed)

    #真实值
    t=data[-50:]
    print(t)
    plt.plot(t)
    plt.plot(results)
    
    plt.show()