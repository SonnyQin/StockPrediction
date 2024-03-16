import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error

#获取数据
df = pd.read_csv(r'./all_stocks_5yr.csv', parse_dates=["date"], index_col=[0])
df = df.head(2000)
# print(df.head())

cols = ['open', 'high', 'low', 'close', 'volume']
data = df[cols]

for i in range(1, 30):
    data['R_%d' % i] = df.close.shift(i)
#第二收盘价作为目标
print(data.close)
data['target'] = data.close.shift(-1)
print(data['target'])
#删除空缺值#重要
data = data.dropna()

train = data[data.index <= '2016/1/4']
valid = data[data.index > '2016/1/4'][:-30]
test = data[-30:]
x_train, x_test, y_train, y_test = train.iloc[:, :-1], valid.iloc[:, :-1], train.target, valid.target

print(x_train)
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
y_pred = rfr.predict(x_train)
print(x_test)
y_valpred = rfr.predict(x_test)
#predict = np.sqrt(np.mean(y_valpred - y_test)**2)  #方差没用的东西
print('MSE:', mean_squared_error(y_train, y_pred), mean_squared_error(y_test, y_valpred))
print('MAE:', mean_absolute_error(y_train, y_pred), mean_absolute_error(y_test, y_valpred))

plt.figure(figsize=(10, 8))
plt.title("股票收盘价格")
plt.xticks(y_test.index)
plt.plot(y_test.values, label="真实")
plt.plot(y_valpred, label="预测")

plt.legend()
plt.show()