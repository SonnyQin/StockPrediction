import Preprocess
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
MTOP={'n_traits':5,'n_neurons':200, 'n_output':100, 'n_epoches':100, 'n_timeOrder':10, 'n_dropoutrate':0.01,'n_testDataAmount':300 }
class KNNPredictionLabel():
    def __init__(self):
        paras={'n_neighbors':100}
        self.classifier=KNeighborsClassifier()
        self.classifier.set_params(**paras)
    def train(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)
    def predict(self,x_test,y_test):
        print(self.classifier.score(x_test,y_test))
    
data=Preprocess.getStockData(MTOP, 'NAVI',t=True)
x=data[:,:-5]
y=data[:,-3]
x_train,x_test=x[:-300],x[-300:]
y_train,y_test=y[:-300],y[-300:]
num=len(y_test)
count=0
for i in range(num):
    if y_test[i]==2:
        count+=1
        
print('Guess accuracy:')
print(count/num)
Knn=KNNPredictionLabel()
Knn.train(x_train, y_train)
Knn.predict(x_test, y_test)