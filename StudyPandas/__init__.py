import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# 第一步加载数据
# pandas读取数据
data = pd.read_csv('data/2012_Federal_Election_Commission_Database.csv')
'''
    cand_nm – 接受捐赠的候选人姓名
    contbr_nm – 捐赠人姓名
    contbr_st – 捐赠人所在州
    contbr_employer – 捐赠人所在公司
    contbr_occupation – 捐赠人职业
    contb_receipt_amt – 捐赠数额（美元）
    contb_receipt_dt – 收到捐款的日期
'''
# 查看前五行数据
# data.head()
# print(data.head())

# 查看数据的信息，包括每个字段的名称、非空数量、字段的数据类型
# print(data.info())
# contbr_employer,contbr_occupation 缺失部分数据

# 统计信息概要
# print(data.describe())

# 第二步数据清洗
# 缺失值处理
# 将空数据填充NOT PROVIDED
data['contbr_employer'].fillna('NOT PROVIDED', inplace=True)
data['contbr_occupation'].fillna('NOT PROVIDED', inplace=True)

# fomat格式化输出
# cnad_nm 候选人名字
# unique() 去重
# print('共有{}位候选人'.format(len(data['cand_nm'].unique())))
# 查看候选人名字，查询他们的党派
# print(data['cand_nm'].unique())
parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

# map映射函数，增加一列party存储党派信息
data['party'] = data['cand_nm'].map(parties)
data['party'].value_counts()
data.groupby('contbr_occupation')['contb_receipt_amt'].sum().sort_values(ascending=False)[:20]
# 数据中存在一些相同职业不同称呼，将他们进行转化
occupation_map = {
  'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
  'INFORMATION REQUESTED': 'NOT PROVIDED',
  'SELF': 'SELF-EMPLOYED',
  'SELF EMPLOYED': 'SELF-EMPLOYED',
  'C.E.O.': 'CEO',
  'LAWYER': 'ATTORNEY',
}
# 如果不在字典中,返回x
f = lambda x: occupation_map.get(x, x)
data.contbr_occupation = data.contbr_occupation.map(f)


# 我们限定数据集只有正出资额
data = data[data['contb_receipt_amt'] > 0]

# 候选人瞎选
# data.groupby('cand_nm')['contb_receipt_amt'].sum().sort_values(ascending=False)
# 候选人前两位以及他们的赞助金额:Obama, Barack:1.358776e+08,Romney, Mitt:8.833591e+07
# 一二与后面差距较大，所以我们选取候选人为Obama、Romney的子集数据
data_vs = data[data['cand_nm'].isin(['Obama, Barack', 'Romney, Mitt'])].copy()
bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
# cut将数据分段
labels = pd.cut(data_vs['contb_receipt_amt'], bins)
# 按照党派、职业对赞助金额进行汇总，类似excel中的透视表操作，聚合函数为sum
by_occupation = data.pivot_table('contb_receipt_amt', index='contbr_occupation', columns='party', aggfunc='sum')
# 过滤掉赞助金额小于200W的数据
over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
# bar绘制条形图
over_2mm.plot(kind='bar')
grouped_bins = data_vs.groupby(['cand_nm', labels])
grouped_bins.size().unstack(0)
# 统计区间的赞助金额
bucket_sums = grouped_bins['contb_receipt_amt'].sum().unstack(0)
# Obama、Romney各区间赞助总金额
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)
# 百分比堆积图
plt.title('Democrat和Republican获得赞助金比较')
normed_sums[:-2].plot(kind='bar', stacked=True)
# 指定默认字体 SimHei为黑体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 使用to_datetime将str转换成time
data_vs['time'] = pd.to_datetime(data_vs['contb_receipt_dt'])
data_vs.set_index('time', inplace=True)
# 把频率从日变成月，高频转化成低频的降采样
vs_time = data_vs.groupby('cand_nm').resample('M')['cand_nm'].count()
fig1, ax1 = plt.subplots(figsize=(32, 8))
# print(vs_time.unstack(0))
# 绘制面积图
plt.title('两位总统候选人接受的赞助笔数')
vs_time.unstack(0).plot(kind='area', ax=ax1, alpha=0.6)
plt.show()
