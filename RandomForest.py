import Preprocess
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def RandomForest_Prediction(closed,t):
    closed=np.array(closed)
    train=Preprocess.series_to_supervised(closed, 6, 2)
    x_train=train.iloc[:,:-1]
    y_train=train['var1(t+1)']
    print(y_train)
    
    #会少一个特征
    RandomForest=RandomForestRegressor(n_estimators=1000)
    results=[]
    
    for i in range(0, 10):
        print(i)
        #选取的特征来预测
        x_Test=closed[-7:]
        x_Test=[x_Test.reshape(-1)]
        RandomForest.fit(x_train.values, y_train.values)
        result=RandomForest.predict(x_Test)
        closed=np.append(closed, result)
        results.append(result)
        result=pd.Series(result)
        y_train=y_train._append(result)
        x_Test=pd.DataFrame(x_Test, columns=x_train.columns)
        x_train=x_train._append(x_Test)
        print(result)
        print(t[i])
        #print(x_train)
    
    return results

if __name__=='__main__':
    # # load the dataset
    # series = pd.read_csv(r'C:/Users/A/Desktop/Birth/daily-total-female-births.csv', header=0, index_col=0)
    # values = series.values
    # results=RandomForest_Prediction(values[:-49])
    # #真实值
    # plt.plot(values[-50:])
    # plt.plot(results)
    # plt.show()
    
    # plt.show()
    
    #错位了
    
    stockName='CCI'
    #所有结束时的价格
    data=Preprocess.load_data()[stockName]
    data=data[:,4]
    closed=data[:-50]
    #真实值
    t=data[-50:]
    
    results=RandomForest_Prediction(closed,t)
    print(results)

    plt.plot(t)
    plt.plot(results)
    
    plt.show()