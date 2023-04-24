from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 练习的数据：
data = pd.read_csv('rfeout.csv',header=0)

a=data.corr(method='pearson')

# 绘制热度图：
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['axes.unicode_minus'] = False

plot = sns.heatmap(a)

plt.title('皮尔逊热力图')

plt.show()
