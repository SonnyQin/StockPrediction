# from statsmodels.tsa.statespace.sarimax import SARIMAX
# import Preprocess
# import matplotlib.pyplot as plt
# import numpy as np
# if __name__=='__main__':
#     data=Preprocess.load_data()['AAL']
#     closed=np.array(data[:,4])
#     closed=closed.reshape(-1)
#     closed=closed.astype('float64')
#     # fit model
#     model = SARIMAX(closed, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
#     model_fit = model.fit(disp=False)
#     print(len(closed))
#     # make prediction
#     yhat = model_fit.predict(len(closed), len(closed))
#     print(yhat)
#     plt.plot(yhat)
#     plt.show()

# SARIMA example
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
data = [x + random() for x in range(1, 100)]
# fit model
model = SARIMAX()
model_fit = model.fit(data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
# make prediction
yhat = mode.predict(len(data), len(data))
print(yhat)