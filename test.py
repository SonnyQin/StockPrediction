from sklearn.preprocessing import MinMaxScaler
import numpy as np
# 创建数据
data = np.array([[1, 2], [3, 4], [5, 6]])
# 创建MinMaxScaler对象
scaler = MinMaxScaler()
# 对数据进行缩放处理
scaled_data = scaler.fit_transform(data)
mydata=np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]])
print(scaler.inverse_transform(mydata))