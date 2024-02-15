import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

# 从已经写好的csv文件中读取数据
mydata = pd.read_csv("data.csv")
print(data) # 查看数据

# 将数据提取出作为坐标，将数值转化为int型，datetime类型转化为string类型
count = np.array(mydata['Count'].astype(str).astype(int))
date = np.array(mydata['Time'].astype(str))
# X坐标，将str类型的数据转换为datetime.date类型的数据，作为x坐标
xs = [datetime.strptime(d, '%Y-%m-%d').date() for d in date]
 
# 图表格式
# 设置图形格式

plt.title('地铁人数',fontsize=25)  # 字体大小设置为25
plt.xlabel('日期',fontsize=10)   # x轴显示“日期”，字体大小设置为10
plt.ylabel('人数',fontsize=10)  # y轴显示“人数”，字体大小设置为10
plt.plot(xs, count, 'o-',label='客流量')
plt.tick_params(axis='both',which='both',labelsize=10)
 
# 显示折线图
plt.gcf().autofmt_xdate()  # 自动旋转日期标记
plt.show()