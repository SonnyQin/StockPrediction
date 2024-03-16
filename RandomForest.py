import Preprocess
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
def RandomForest_Prediction(x, y):
    RandomForest=RandomForestRegressor()
    x=x.reshape(-1,1)
    y=np.array(y)
    y=y.reshape(-1,1)
    count=[[x.shape[0]]]
    results=[]
    for i in range(0, 50):
        RandomForest.fit(x,y)
        count[0][0]+=1
        result=RandomForest.predict(count)
        results.append(result)
        x=np.append(x, np.array(count[0]))
        y=np.append(y, np.array(result[0]))
        x=x.reshape(-1,1)
        y=y.reshape(-1,1)
        print(result)
    return results

if __name__=='__main__':
    stockName='MDT'
    #所有结束时的价格
    data=Preprocess.load_data()[stockName]
    data=data[:,4]
    t=data[-51:]
    print(t)
    plt.plot(t)
    
    y=data[:-51]
    # x=data[-20:]
    timeOrder=np.linspace(1, y.shape[0], y.shape[0])
    
    results=RandomForest_Prediction(timeOrder, y)
    plt.plot(results)
    
    plt.show()