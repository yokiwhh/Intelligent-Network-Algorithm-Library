import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
data1 = np.random.normal(loc=10, scale=2, size=50)
data2 = np.random.normal(loc=20, scale=3, size=50)

# 转换数据格式
df = pd.DataFrame({'data': np.concatenate([data1, data2]),
                   'group': ['data1']*len(data1) + ['data2']*len(data2)})

# 绘制箱型图
fig, ax = plt.subplots(figsize=(8,6))
sns.boxplot(x='group', y='data', data=df, ax=ax, palette='pastel', dodge=True)

# 添加分布密度图
sns.stripplot(x='group', y='data', data=df, ax=ax, color='gray', alpha=0.5)

# 添加图例
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['data1', 'data2'], loc='upper left')

# 添加网格线
ax.grid(axis='y')

# 添加坐标轴标签
ax.set_xlabel('')
ax.set_ylabel('')

# 设置右侧y轴
ax2 = ax.twinx()
y2_min = np.min(data2) - 1.5*(np.percentile(data2, 75) - np.percentile(data2, 25))
y2_max = np.max(data2) + 1.5*(np.percentile(data2, 75) - np.percentile(data2, 25))
ax2.set_ylim([y2_min, y2_max])
ax2.set_ylabel('data2', rotation=270, labelpad=15)

# 设置左侧y轴
y1_min = np.min(data1) - 1.5*(np.percentile(data1, 75) - np.percentile(data1, 25))
y1_max = np.max(data1) + 1.5*(np.percentile(data1, 75) - np.percentile(data1, 25))
ax.set_ylim([y1_min, y1_max])
ax.set_ylabel('data1')

# 设置图表标题
ax.set_title('Comparison of Two Data Distributions')

plt.show()