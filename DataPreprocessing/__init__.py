import pandas as pd
from sklearn import preprocessing
import numpy as np
# 输出不省略
np.set_printoptions(threshold=np.inf)
# test.xls 用于练习处理缺失值
# test1.xls 用于练习特征编码，数据规范化
data = pd.read_excel("data/test.xls")
data1 = pd.read_excel("data/test1.xls")


# 查看数据
# 获取表头
print(data.columns)
# 统计每列缺失值情况
print(data.isnull().sum())
# 查看每列数据值数目
print(data.info())
# 统计每行缺失值情况
print(data.isnull().sum(axis=1))


# 简单方法：直接删除缺失值得数据，但是造成数据的浪费。
# 删除缺失值的行 dropna(inplace=True) 直接修改data的值
print(data.dropna())
# 删除缺失值得列
print(data.dropna(axis=1))
# 删除全为空值的行
print(data.dropna(how="all"))
# 删除全为空值的列
print(data.dropna(axis=1,how="all"))
# 保留至少4个非空值得数据
print(data.dropna(thresh=4))
# 删除B列包含缺失值的行
print(data.dropna(subset=["B"]))
# 注意:dropna有时候删除不掉空值
# test[data.isnull(['A']) == True]


# 均值插补
# 平均值
# print(data.mean())
print(data.mean())
# 中位数
print(data.median())
# 众数
print(data['A'].mode())
# 将各列空值插入平均值/中位数/众数
# 数据是定量数据所以采用平均值插补缺失值
# 定量数据一般由测量或计数、统计所得到的量（长度，温度）
# 定性数据一般是性质上的差异（品质）
# 如果缺失值是定量的，就以该字段存在值的平均值来插补缺失的值。
# 如果缺失值是定性的，就以该字段存在值得众数来插补缺失的值。
for x in data.columns:
    data[x].fillna(data.mean()[x], inplace=True)
print(data)

# 利用同类均值插补 用层次聚类模型预测缺失变量的类型，再以该类型的均值插补
# 极大似然估计（前提大样本）


# 特征编码
# One-Hot编码（处理非数值属性）
enc = preprocessing.OneHotEncoder()
enc.fit(data1[['color', 'kind']])
tempdata = enc.transform(data1[['color', 'kind']]).toarray()


# 数据标准化
# [0, 1] 归一化
print(data1['weight'])
max_weight = data1['weight'].max()
min_weight = data1['weight'].min()
for x in data1['weight']:
    x = (x - min_weight) / (max_weight - min_weight)
    print(x)
# Z-score标准化(原始数据的分布可以近似为高斯分布,不然效果较差)
mean_weight = data1['weight'].mean()
std_weight = data1['weight'].std()
for x in data1['weight']:
    x = (x - mean_weight) / std_weight
    print(x)
# 正则化
