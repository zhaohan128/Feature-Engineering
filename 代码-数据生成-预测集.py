# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 11:03:54 2022

@author: 19105
"""

import pandas as pd                                                     
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta


def todatetime(x):
    return datetime.strptime(x,'%y-%b')

def todatetime2(x):
    return datetime.strptime(x,'%b-%y')

def y_map(x):
    if x=='null':
        y = 0   ##负样本
    elif (x[0]=='C') or (x[0]=='c') :
        y = 0   ##退货
    else:
        y = 1   ##正样本
    return y

def times2days(x):
    #把时间转换为距离下个月15号的天数
     x = x.to_pydatetime()
     t = x + relativedelta(months=+1)
     y = (datetime(t.year,t.month,15) - x).days
     return y

def strplace(x):
    return x.replace('\r','')

def countryflag(x):
    #x=str(x)
    if x in 'Poland|Lithuania|Channel Islands|Lebanon|United Arab Emirates|Saudi Arabia|Unspecified|Brazil|European Community|Bahrai|RSA|nan':
        y = 0
    else:
        y = 1
    return y

def countrystr(x):
    x=str(x)
    return x

##导入数据，修改文件路径
#导入订单表
df_ord = pd.read_csv('Merge_test/ecommercedata-old.csv',parse_dates=['InvoiceDate'])
#导入特征表
df_pro = pd.read_csv('Merge_test/product_feature.csv',
                     parse_dates=['date'], encoding='utf-8')
df_cus = pd.read_csv('Merge_test/customer_feature.csv',encoding='gbk')
df_cpjh = pd.read_csv('Merge_test/cp_jiaohufeature.csv',
                      parse_dates=['purchase_recentdays_UI'], encoding='gbk')
df_proclass =  pd.read_csv('Merge_test/class_feature.csv',encoding='gbk')
df_cpclass =  pd.read_csv('Merge_test/cpclass_jiaohufeature.csv',
                     parse_dates=['purchase_recentdays_UC'], encoding='gbk')
df_prolabel =  pd.read_csv('Merge_test/ecommercedata-label3.csv',encoding='gbk')
#导入预测
df_pre = pd.read_csv('Merge_test/ecommercedata-pre.csv')
#提示列名
#cols = ['InvoiceNo','StockCode','Description','Quantity',
#        'InvoiceDate','UnitPrice','CustomerID','Country']

#修改指标
df_ord['order_month'] = df_ord['InvoiceDate'].dt.month
df_prolabel['category']=df_prolabel['label']
#print(df_ord[0:5])
df_pro['order_month'] = df_pro['date'].dt.month
df_pro = df_pro.drop(['date','category'],axis=1) #删去没有用的列
#print(df_pro[0:5])
df_cus['交易日期'] = df_cus['交易日期'].apply(todatetime)
df_cus['特征日期'] = df_cus['特征日期'].apply(todatetime)
df_cus['order_month'] = df_cus['交易日期'].dt.month
df_cus['Country'] = df_cus['Country'].apply(strplace)
df_cus = df_cus.drop(['交易日期', '特征日期'],axis=1) #删去没有用的列，交易日期', '特征日期'
#print(df_cus[0:5])
df_cpjh['交易日期'] = df_cpjh['交易日期'].apply(todatetime)
df_cpjh['特征日期'] = df_cpjh['特征日期'].apply(todatetime)
df_cpjh['order_month'] = df_cpjh['交易日期'].dt.month
df_cpjh['purchase_recentdays_UI'] = df_cpjh['purchase_recentdays_UI'].apply(times2days)
df_cpjh['Country'] = df_cpjh['Country'].apply(strplace)
df_cpjh = df_cpjh.drop(['交易日期', '特征日期'],axis=1) #删去没有用的列，交易日期', '特征日期'
#print(df_cpjh[0:5])
df_proclass['交易日期'] = df_proclass['交易日期'].apply(todatetime)
df_proclass['特征日期'] = df_proclass['特征日期'].apply(todatetime)
df_proclass['order_month'] = df_proclass['交易日期'].dt.month
df_proclass = df_proclass.drop(['交易日期', '特征日期'],axis=1) #删去没有用的列，交易日期', '特征日期'
#print(df_proclass[0:5])
df_cpclass['交易日期'] = df_cpclass['交易日期'].apply(todatetime)
df_cpclass['特征日期'] = df_cpclass['特征日期'].apply(todatetime)
df_cpclass['order_month'] = df_cpclass['交易日期'].dt.month
df_cpclass['purchase_recentdays_UC'] = df_cpclass['purchase_recentdays_UC'].apply(times2days)
df_cpclass['Country'] = df_cpclass['Country'].apply(strplace)
df_cpclass = df_cpclass.drop(['交易日期', '特征日期'],axis=1) #删去没有用的列，交易日期', '特征日期'
#print(df_cpclass[0:5])

#取出某个月的订单数据，例如6月
#mon=1 ######################修改月份
#df_month = df_ord.query('order_month==@mon') # 选择所有月则，
df_month = df_ord
#选出去重商品和顾客
#df_1 = df_month[['CustomerID','Country','order_month']].drop_duplicates(subset=['CustomerID','Country','order_month'])
#df_2 = df_month[['StockCode','order_month']].drop_duplicates(subset=['StockCode','order_month'])
df_1 = df_pre[['CustomerID']].drop_duplicates(subset=['CustomerID'])
df_2 = df_month[['StockCode']].drop_duplicates(subset=['StockCode'])
df_1 = df_1.merge(df_cus[['CustomerID','Country']],on=['CustomerID'],how='left')
df_1['Country']=df_1['Country'].apply(countrystr)
df_1 = df_1[['CustomerID','Country']].drop_duplicates(subset=['CustomerID','Country'])
#查看冷启动用户数
print(df_1['Country'].isin(['nan']).sum())
df_1['order_month']=12
df_2['order_month']=12
##组合所有可能性，连接特征，构造样本
df = pd.merge(df_1,df_2,on=['order_month'],how='outer')
df = df.merge(df_prolabel,on=['StockCode'],how='left')
#df = df.merge(df_ord,on=['StockCode','CustomerID','Country','order_month'],how='left')
#df = df[['StockCode', 'CustomerID', 'Country', 'InvoiceNo', 'order_month', 'category']]
df = df[['StockCode', 'CustomerID', 'Country','order_month', 'category']]
df = df.merge(df_pro,on=['StockCode','order_month'],how='left')
df = df.merge(df_cus,on=['CustomerID','Country','order_month'],how='left')
df = df.merge(df_cpjh,on=['StockCode','CustomerID','Country','order_month'],how='left')
df = df.merge(df_proclass,on=['category','order_month'],how='left')
df = df.merge(df_cpclass,on=['category','CustomerID','Country','order_month'],how='left')
#print(df[0:5])

cols = df.columns
#打印没有匹配上的行数
print(len(df)) #所有可能性行数
#print(df['InvoiceNo'].isnull().sum()/len(df)) #正样本+退货样本行数，百分之99
print(df['sales_last'].isnull().sum()/len(df)) #商品特征没有匹配上的行数，0.7
print(df['transaction_amount_last'].isnull().sum()/len(df)) #用户特征没有匹配上的行数，二分之一
print(df['sales_last_category'].isnull().sum()/len(df)) #商品类特征没有匹配上的行数，0
print(df['sales_last_UI'].isnull().sum()/len(df)) #交互特征没有匹配上的行数,千分之999
print(df['sales_last_UC'].isnull().sum()/len(df)) #类交互特征没有匹配上的行数,93%

#缺失值填充
print(df.isnull().sum().sum())
#df['InvoiceNo']=df['InvoiceNo'].fillna('null')
df['is_developed']=df['Country'].apply(countryflag)
df['latest_order_time']=df['latest_order_time'].fillna(99) 
df['repurchase_day']=df['repurchase_day'].fillna(365)
df['transaction_frequency_last']=df['transaction_frequency_last'].fillna(31536000)
df['transaction_frequency_as_of_last']=df['transaction_frequency_as_of_last'].fillna(31536000)
df['purchase_recentdays_UI']=df['purchase_recentdays_UI'].fillna(99)
df['repurchase_day_UI']=df['repurchase_day_UI'].fillna(365)
df['repurchase_day_category']=df['repurchase_day_category'].fillna(365)
df['purchase_recentdays_UC']=df['purchase_recentdays_UC'].fillna(99)
df['repurchase_day_UC']=df['repurchase_day_UC'].fillna(365)
df=df.fillna(0)
print(df.isnull().sum().sum())

#构造Y
#df['Y'] = df['InvoiceNo'].apply(y_map)
df['Y'] = pd.Series()
#查看正负样本行数
#temp = df.groupby('Y').count()
#查看列名
cols = df.columns
# #删去没有用的列，'StockCode', 'CustomerID', 'Country', 'order_month', 'InvoiceNo','label'
#df = df.drop(['StockCode', 'CustomerID', 'Country', 'order_month', 'InvoiceNo','label'],axis=1)
#保存数据
df.to_csv('Merge_test/out_sample_0226_Pre_2.csv',encoding='utf-8_sig') ##输出样本
