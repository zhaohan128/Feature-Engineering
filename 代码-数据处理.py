from collections import Counter
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np
def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.):
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    boundary = []  # 待return的分箱边界值列表
    clf = DecisionTreeClassifier(criterion='entropy',    #“信息熵”最小化准则划分
                                 max_leaf_nodes=6,       # 最大叶子节点数
                                 min_samples_leaf=0.05)  # 叶子节点样本数量最小占比
    x = np.array(x)
    clf.fit(x.reshape(-1, 1), y)  # 训练决策树
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])
    boundary.sort()
    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    boundary = [min_x] + boundary + [max_x]
    return boundary
def feature_woe_iv(x, y):
    boundary = optimal_binning_boundary(x, y)        # 获得最优分箱边界值列表
    df = pd.concat([x, y], axis=1)                        # 合并x、y为一个DataFrame，方便后续计算
    df.columns = ['x', 'y']                               # 特征变量、目标变量字段的重命名
    df['bins'] = pd.cut(x=x, bins=boundary, right=False)  # 获得每个x值所在的分箱区间
    grouped = df.groupby('bins')['y']                     # 统计各分箱区间的好、坏、总客户数量
    result_df = grouped.agg([('good',  lambda y: (y == 1).sum()),
                             ('bad',   lambda y: (y == 0).sum()),
                             ('total', 'count')])
    result_df['good_pct'] = result_df['good'] / result_df['good'].sum()       # 好客户占比
    result_df['bad_pct'] = result_df['bad'] / result_df['bad'].sum()          # 坏客户占比
    result_df['total_pct'] = result_df['total'] / result_df['total'].sum()    # 总客户占比
    result_df['bad_rate'] = result_df['bad'] / result_df['total']             # 坏比率
    result_df['woe'] = np.log(result_df['good_pct'] / result_df['bad_pct'])              # WOE
    result_df['iv'] = (result_df['good_pct'] - result_df['bad_pct']) * result_df['woe']  # IV
    #print(f"该变量IV = {result_df['iv'].sum()}")
    return result_df['iv'].sum(),boundary
def discrete(x,boundary):
    n = len(boundary)
    for i in range(n):
        if x < boundary[i]:
            return i
    return n
#------------读取数据----------------------
train_data1 = pd.read_csv("out_sample_0221_Apr.csv")
train_data2 = pd.read_csv("out_sample_0221_Aug.csv")
train_data3 = pd.read_csv("out_sample_0221_Feb.csv")
train_data4 = pd.read_csv("out_sample_0221_Jan.csv")
train_data5 = pd.read_csv("out_sample_0221_July.csv")
train_data6 = pd.read_csv("out_sample_0221_Jun.csv")
train_data7 = pd.read_csv("out_sample_0221_Mar.csv")
train_data8 = pd.read_csv("out_sample_0221_May.csv")
train_data9 = pd.read_csv("out_sample_0221_Oct.csv")
train_data10 = pd.read_csv("out_sample_0221_Sep.csv")
train_data = pd.concat([train_data1,train_data2,train_data3,train_data4,train_data5,train_data6,train_data7,train_data8,
                       train_data9,train_data10],axis=0)
val_data = pd.read_csv("out_sample_0221_Nov.csv")
test_data = pd.read_csv("out_sample_0221_Dec.csv")

#----------确定下正负比例
train_y = train_data['Y']
label_count = Counter(train_y)
val_y = val_data['Y']
test_y = test_data['Y']
print(label_count)
#-----删除不要的列
del_columns = ['Unnamed: 0','StockCode','CustomerID','Country','InvoiceNo','order_month']
train_data_1 = copy.deepcopy(train_data)
val_data_1 = copy.deepcopy(val_data)
test_data_1 = copy.deepcopy(test_data)
train_data_1.drop(del_columns,axis=1,inplace=True)
val_data_1.drop(del_columns,axis=1,inplace=True)
val_data_1.drop(del_columns,axis=1,inplace=True)
# -----数据分桶
spase_columns = ['category','price_trend','is_developed']
continuous_columns  = list(set(train_data.columns)-set(spase_columns)-set('Y'))
new_continuous_columns = []
for i in continuous_columns:
    iv , boundary = feature_woe_iv(train_data_1[i],train_y)
    print(iv,len(boundary))
    if iv > 0.1:
        train_data_1[i] = train_data_1[i].apply(discrete,boundary=boundary)
        val_data_1[i] = val_data_1[i].apply(discrete, boundary=boundary)
        test_data_1[i] = test_data_1[i].apply(discrete, boundary=boundary)
        spase_columns.append(i)
    else:
        new_continuous_columns.append(i)
#-----------对离散变量做独热编码
for i in spase_columns:
    tmp = pd.get_dummies(train_data_1[i],prefix = i)
    train_data = pd.concat([train_data,tmp],axis=1)
    train_data.drop(i,aixs=1,inplace=True)

    tmp = pd.get_dummies(val_data_1[i], prefix=i)
    val_data = pd.concat([val_data, tmp], axis=1)
    val_data.drop(i, aixs=1, inplace=True)

    tmp = pd.get_dummies(test_data_1[i], prefix=i)
    test_data = pd.concat([test_data, tmp], axis=1)
    test_data.drop(i, aixs=1, inplace=True)
train_data.to_csv("train.csv")
val_data.to_csv("val.csv")
test_data.to_csv("test.csv")